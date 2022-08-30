import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import crf
import numpy as np

conv1d = tf.layers.conv1d

def dropout(input_tensor, dropout_prob):
    """Perform dropout.

    Args:
    input_tensor: float Tensor.
    dropout_prob: Python float. The probability of dropping out a value (NOT of
      *keeping* a dimension as in `tf.nn.dropout`).

    Returns:
    A version of `input_tensor` with dropout applied."""
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
    return output

def layer_norm(input_tensor, name=None):
    """Run layer normalization on the last dimension of the tensor."""
    return tf.contrib.layers.layer_norm(inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)

def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
    """Runs layer normalization followed by dropout."""
    output_tensor = layer_norm(input_tensor, name)
    output_tensor = dropout(output_tensor, dropout_prob)
    return output_tensor

class GAT_BILSTM_CRF(object):
    def __init__(self, embedded_chars, hidden_unit, cell_type, num_layers, dropout_rate,
                 initializers, num_labels, seq_length, labels, lengths, is_training,
                 bias_matrix, syntactic_feature_matrix, num_gat_heads, num_gat_units, dropout_rate_of_gat, num_gat_layers):
        """
        BLSTM-CRF 网络
        :param embedded_chars: Fine-tuning embedding input
        :param hidden_unit: LSTM的隐含单元个数
        :param cell_type: RNN类型（LSTM OR GRU DICNN will be add in feature）
        :param num_layers: RNN的层数
        :param droupout_rate: droupout rate
        :param initializers: variable init class
        :param num_labels: 标签数量
        :param seq_length: 序列最大长度
        :param labels: 真实标签
        :param lengths: [batch_size] 每个batch下序列的真实长度
        :param is_training: 是否是训练过程
        """
        self.hidden_unit = hidden_unit
        self.dropout_rate = dropout_rate
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.embedded_chars = embedded_chars
        self.initializers = initializers
        self.seq_length = seq_length
        self.num_labels = num_labels
        self.labels = labels
        self.lengths = lengths
        self.embedding_dims = embedded_chars.shape[-1].value
        self.is_training = is_training
        self.syntactic_feature_matrix = syntactic_feature_matrix
        self.bias_matrix = bias_matrix
        self.num_gat_heads = num_gat_heads
        self.num_gat_units = num_gat_units
        self.dropout_rate_of_gat = dropout_rate_of_gat
        self.num_gat_layers = num_gat_layers

    def add_gat_blstm_crf_layer(self, flag = 0):
        """
        blstm-crf网络
        :return:
        """        
        hid_units = [1024]
        n_heads = [1, 1]

        if self.is_training:
            # lstm input dropout rate i set 0.9 will get best score
            self.embedded_chars = tf.nn.dropout(self.embedded_chars, self.dropout_rate)
        
        embeddings = tf.concat([self.embedded_chars, self.syntactic_feature_matrix], axis = -1)
    
        print("embeddings type:", type(embeddings))
        print("embeddings shape:", embeddings.shape)

        if flag == 2:
            logits = self.project_crf_layer(embeddings)
        elif flag == 1:
            # blstm
            lstm_output = self.blstm_layer(embeddings)
            # project
            logits = self.project_bilstm_layer(lstm_output)
        else:
            # gat layer
            gat_output = self.gat_layer(embedding_chars = embeddings, nb_classes = self.num_labels, nb_nodes = self.seq_length,
                                        training = True, attn_drop=0.0, ffd_drop = 0.0, bias_mat = self.bias_matrix, 
                                        hid_units = hid_units, n_heads = n_heads, residual = False, activation = tf.nn.elu)
            # bi_lstm layer
            lstm_output = self.blstm_layer(gat_output)
            # project
            logits = self.project_bilstm_layer(lstm_output)
        # crf
        loss, trans = self.crf_layer(logits)
        # CRF decode, pred_ids 是一条最大概率的标注路径
        pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=trans, sequence_length=self.lengths)

        # 模仿biobert构建predict
        predict = {"predict": pred_ids}
        return (loss, logits, trans, predict)

        # return (loss, logits, trans, pred_ids)
        
    def get_gat_embeddings(self):
        hid_units = [self.num_gat_units]
        n_heads = [self.num_gat_heads, 1]
        
        if self.is_training:
            attn_drop = self.dropout_rate_of_gat
            ffd_drop = self.dropout_rate_of_gat
        else:
            attn_drop = 0.0
            ffd_drop = 0.0
        
        dropout_prob = 0.0
        
        # if self.is_training:
        #     # lstm input dropout rate i set 0.9 will get best score
        #     self.embedded_chars = tf.nn.dropout(self.embedded_chars, self.dropout_rate)
        
        embeddings = tf.concat([self.embedded_chars, self.syntactic_feature_matrix], axis = -1)
        embedding_dims = embeddings.shape[-1].value
        
        flag = self.num_gat_layers
        while flag:
            flag -= 1
            # gat_output = self.gat_layer(is_training = self.is_training, embedding_chars = embeddings, nb_classes = self.num_labels, nb_nodes = self.seq_length,
            #                                 training = True, attn_drop=self.dropout_rate_of_gat, ffd_drop = self.dropout_rate_of_gat, bias_mat = self.bias_matrix, 
            #                                 hid_units = hid_units, n_heads = n_heads, residual = False, activation = tf.nn.elu, num_layer = flag)
            gat_output = self.gat_layer(embedding_chars = embeddings, nb_classes = self.num_labels, nb_nodes = self.seq_length,
                                            training = True, attn_drop = attn_drop, ffd_drop = ffd_drop, bias_mat = self.bias_matrix, 
                                            hid_units = hid_units, n_heads = n_heads, residual = False, activation = tf.nn.elu, num_layer = flag)
            gat_output = layer_norm_and_dropout(gat_output, dropout_prob)
            embeddings = gat_output
        return gat_output

    def _witch_cell(self):
        """
        RNN 类型
        :return:
        """
        cell_tmp = None
        if self.cell_type == 'lstm':
            cell_tmp = rnn.LSTMCell(self.hidden_unit)
        elif self.cell_type == 'gru':
            cell_tmp = rnn.GRUCell(self.hidden_unit)
        return cell_tmp

    def _bi_dir_rnn(self):
        """
        双向RNN
        :return:
        """
        cell_fw = self._witch_cell()
        cell_bw = self._witch_cell()
        if self.dropout_rate is not None:
            cell_bw = rnn.DropoutWrapper(cell_bw, output_keep_prob=self.dropout_rate)
            cell_fw = rnn.DropoutWrapper(cell_fw, output_keep_prob=self.dropout_rate)
        return cell_fw, cell_bw

    def blstm_layer(self, embedding_chars):
        """

        :return:
        """
        with tf.variable_scope('rnn_layer'):
            cell_fw, cell_bw = self._bi_dir_rnn()
            if self.num_layers > 1:
                cell_fw = rnn.MultiRNNCell([cell_fw] * self.num_layers, state_is_tuple=True)
                cell_bw = rnn.MultiRNNCell([cell_bw] * self.num_layers, state_is_tuple=True)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, embedding_chars,
                                                         dtype=tf.float32)
            outputs = tf.concat(outputs, axis=2)
        return outputs

    def project_bilstm_layer(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.hidden_unit * 2, self.hidden_unit],
                                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.hidden_unit], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.hidden_unit * 2])
                hidden = tf.nn.xw_plus_b(output, W, b)

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.hidden_unit, self.num_labels],
                                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.num_labels], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W, b)
            return tf.reshape(pred, [-1, self.seq_length, self.num_labels])

    def project_crf_layer(self, embedded_chars, name=None):
        """
        hidden layer between input layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.embedding_dims, self.num_labels],
                                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.num_labels], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(self.embedded_chars,
                                    shape=[-1, self.embedding_dims])  # [batch_size, embedding_dims]
                pred = tf.tanh(tf.nn.xw_plus_b(output, W, b))
            return tf.reshape(pred, [-1, self.seq_length, self.num_labels])

    def crf_layer(self, logits):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"):
            trans = tf.get_variable(
                "transitions",
                shape=[self.num_labels, self.num_labels],
                initializer=self.initializers.xavier_initializer())
            if self.labels is None:
                return None, trans
            else:
                log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
                    inputs=logits,
                    tag_indices=self.labels,
                    transition_params=trans,
                    sequence_lengths=self.lengths)
                return tf.reduce_mean(-log_likelihood), trans

    def gat_attn_head_simple(self, seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):
        with tf.name_scope('my_gat_attn'):
            if in_drop != 0.0:
                seq = tf.nn.dropout(seq, 1.0 - in_drop)
            seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)
            # simplest self-attention possible
            f_1 = tf.layers.conv1d(seq_fts, 1, 1)
            f_2 = tf.layers.conv1d(seq_fts, 1, 1)
            logits = f_1 + tf.transpose(f_2, [0, 2, 1])
            coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)
            if coef_drop != 0.0:
                coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
            if in_drop != 0.0:
                seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)
            vals = tf.matmul(coefs, seq_fts)
            ret = tf.contrib.layers.bias_add(vals)
            # residual connection
            if residual:
                if seq.shape[-1] != ret.shape[-1]:
                   ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
                else:
                    ret = ret + seq
            return activation(ret)
    
    def gat_attn_head(self, seq, out_sz, bias_mat, activation, head_num, num_layer, in_drop=0.0, coef_drop=0.0, residual=False):
        batch_size = seq.shape.as_list()[0]
        embedding_dim = seq.shape[-1].value
        with tf.name_scope('my_gat_attn'):
            with tf.variable_scope("logits_head%s_layer%s" % (head_num, num_layer)):
                W = tf.get_variable("W", shape = [embedding_dim, out_sz], dtype=tf.float32, initializer=self.initializers.xavier_initializer())
                a = tf.get_variable("a", shape = [out_sz*2, 1], dtype=tf.float32, initializer=self.initializers.xavier_initializer())
                # a = tf.get_variable("a", shape = [out_sz, 1], dtype=tf.float32, initializer=self.initializers.xavier_initializer())
                if in_drop != 0.0:  
                    seq = tf.nn.dropout(seq, 1.0 - in_drop)
                seq_fts = tf.matmul(seq, W)
                seq_fts_intermediate = tf.repeat(seq_fts, repeats=self.seq_length, axis=1)
                seq_fts_repeat = tf.reshape(seq_fts_intermediate, [-1, self.seq_length, self.seq_length, out_sz])
                seq_fts_repeat_transpose = tf.transpose(seq_fts_repeat, [0,2,1,3])
                
                # seq_fts_add = seq_fts_repeat + seq_fts_repeat_transpose
                # logits_unreshaped = tf.matmul(seq_fts_add, a)
                # logits = tf.reshape(logits_unreshaped, [-1, self.seq_length, self.seq_length])
                
                seq_fts_concat = tf.concat([seq_fts_repeat, seq_fts_repeat_transpose], axis = -1)
                logits_unreshaped = tf.matmul(seq_fts_concat, a)
                logits = tf.reshape(logits_unreshaped, [-1, self.seq_length, self.seq_length])
                
                coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)
                if coef_drop != 0.0:
                    coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
                if in_drop != 0.0:
                    seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)
                vals = tf.matmul(coefs, seq_fts)
                ret = tf.contrib.layers.bias_add(vals)
                # residual connection
                if residual:
                    if seq.shape[-1] != ret.shape[-1]:
                        ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
                    else:
                        ret = ret + seq
                return activation(ret)

    def gat_layer(self, embedding_chars, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
            bias_mat, hid_units, n_heads, num_layer, activation=tf.nn.elu, residual=False):
        attns = []
        for _ in range(n_heads[0]):
            attns.append(self.gat_attn_head(embedding_chars, bias_mat=bias_mat,
                out_sz=hid_units[0], activation=activation,
                in_drop=ffd_drop, coef_drop=attn_drop, num_layer= num_layer, residual=False, head_num = _))
        h_1 = tf.concat(attns, axis=-1)
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            for _ in range(n_heads[i]):
                attns.append(self.gat_attn_head(h_1, bias_mat=bias_mat,
                    out_sz=hid_units[i], activation=activation,
                    in_drop=ffd_drop, coef_drop=attn_drop, num_layer= num_layer, residual=False))
            h_1 = tf.concat(attns, axis=-1)
        gat_outputs = h_1
        
        # if is_training:
        #     dropout_prob = 1 - self.dropout_rate_of_gat
        # else:
        #     dropout_prob = 1.0
        # gat_outputs = layer_norm_and_dropout(gat_outputs, dropout_prob)
    
        return gat_outputs
