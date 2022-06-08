#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Copyright 2018 The Google AI Language Team Authors.
BASED ON Google_BERT.
@Author:zhoukaiyin
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from enum import Flag
from itertools import accumulate
import os
from types import prepare_class

# from spacy.util import get_package_path
from lstm_gat_crf_layer_wordpiece import GAT_BILSTM_CRF
import modeling
import optimization
import tokenization
import tensorflow as tf
from tensorflow.python.ops import math_ops
import tf_metrics
import pickle
flags = tf.flags

from tensorflow.contrib.layers.python.layers import initializers
import numpy as np
import spacy
spacy_nlp = spacy.load('en_core_web_trf')

global_input_error = 0
Flag_wordpiece = 3  # Flag_wordpiece  0:normal_tag, 1:split_tag, 2:duplicate_tag, 3:duplicate_pos_low_dim
Flag_network = 0  # Flag_network  0:bert-gat-bilstm-crf, 1:bert-bilstm-gat-crf, 2:bert-gat-crf, 3:bert-bilstm-crf, 4:bert-crf
Flag_init = 2 # Flag_init  0:bert-gat-bilstm-crf, 1:bert-gat-softmax, 2:bert_syntactic_concat
pos_list_low_dim = ["ADJ",
    "ADP",
    "CCONJ",
    "NOUN",
    "NUM",
    "PART",
    "PROPN",
    "PUNCT",
    "SYM",
    "VERB",
    "OTHERS",
    "X",
    "CLS",
    "SEP"]
tag_list = [".",
    ",",
    "-LRB-",
    "-RRB-",
    "``",
    '""',
    "''",
    ":",
    "$",
    "#",
    "AFX",
    "CC",
    "CD",
    "DT",
    "EX",
    "FW",
    "HYPH",
    "IN",
    "JJ",
    "JJR",
    "JJS",
    "LS",
    "MD",
    "NIL",
    "NN",
    "NNP",
    "NNPS",
    "NNS",
    "PDT",
    "POS",
    "PRP",
    "PRP$",
    "RB",
    "RBR",
    "RBS",
    "RP",
    "TO",
    "UH",
    "VB",
    "VBD",
    "VBG",
    "VBN",
    "VBP",
    "VBZ",
    "WDT",
    "WP",
    "WP$",
    "WRB",
    "SP",
    "SYM",
    "ADD",
    "NFP",
    "GW",
    "XX",
    "BES",
    "HVS",
    "_SP",
    "X",
    "CLS",
    "SEP"]
dependency_list_low_dim = ["compound",
    "punct",
    "nmod",
    "amod",
    "pobj",
    "conj",
    "appos",
    "det",
    "nummod",
    "npadvmod",
    "cc",
    "nsubj",
    "dobj",
    "prep",
    "nsubjpass",
    "acl",
    "root",
    "case",
    "others",
    "next"]
dependency_list = ["acl",
    "acomp",
    "advcl",
    "advmod",
    "agent",
    "amod",
    "appos",
    "attr",
    "aux",
    "auxpass",
    "case",
    "cc",
    "ccomp",
    "clf",
    "complm",
    "compound",
    "conj",
    "cop",
    "csubj",
    "csubjpass",
    "dative",
    "dep",
    "det",
    "discourse",
    "dislocated",
    "dobj",
    "expl",
    "fixed",
    "flat",
    "goeswith",
    "hmod",
    "hyph",
    "infmod",
    "intj",
    "iobj",
    "list",
    "mark",
    "meta",
    "neg",
    "nmod",
    "nn",
    "npadvmod",
    "nsubj",
    "nsubjpass",
    "nounmod",
    "npmod",
    "num",
    "number",
    "nummod",
    "oprd",
    "obj",
    "obl",
    "orphan",
    "parataxis",
    "partmod",
    "pcomp",
    "pobj",
    "poss",
    "possessive",
    "preconj",
    "predet",
    "prep",
    "prt",
    "punct",
    "quantmod",
    "rcmod",
    "relcl",
    "reparandum",
    "root",
    "vocative",
    "xcomp",
    "next"]
pos_map_low_dim = {}
for i, pos in enumerate(pos_list_low_dim):
    pos_map_low_dim[pos] = i
tag_map = {}
for i, tag in enumerate(tag_list):
    tag_map[tag] = i
dependency_map = {}
for i, dependency in enumerate(dependency_list):
    dependency_map[dependency] = i
dependency_map_low_dim = {}
for i, dependency in enumerate(dependency_list_low_dim):
    dependency_map_low_dim[dependency] = i

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "task_name", "NER", "The name of the task to train."
)

flags.DEFINE_string(
    "data_dir", None,
    "The input datadir.",
)

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written."
)

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model."
)

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model)."
)

flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text."
)

flags.DEFINE_integer(
    "max_seq_length", 128, # 384 recommended for longer sentences
    "The maximum total input sequence length after WordPiece tokenization."
)

flags.DEFINE_bool(
    "do_train", True,
    "Whether to run training."
)
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", True,"Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 16, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 50.0, "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, 
        bias_matrix, syntactic_feature_matrix):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.bias_matrix = bias_matrix
        self.syntactic_feature_matrix = syntactic_feature_matrix

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """Reads a BIO data."""
        inpFilept = open(input_file)
        lines = []
        words = []
        labels = []
        for lineIdx, line in enumerate(inpFilept):
            contents = line.splitlines()[0]
            lineList = contents.split()
            if len(lineList) == 0: # For blank line
                assert len(words) == len(labels), "lineIdx: %s,  len(words)(%s) != len(labels)(%s) \n %s\n%s"%(lineIdx, len(words), len(labels), " ".join(words), " ".join(labels))
                if len(words) != 0:
                    wordSent = " ".join(words)
                    labelSent = " ".join(labels)
                    lines.append((labelSent, wordSent))
                    words = []
                    labels = []
                else: 
                    print("Two continual empty lines detected!")
            else:
                words.append(lineList[0])
                labels.append(lineList[-1])
        if len(words) != 0:
            wordSent = " ".join(words)
            labelSent = " ".join(labels)
            lines.append((labelSent, wordSent))
            words = []
            labels = []

        inpFilept.close()
        return lines

class NerProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train_dev.tsv")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "devel.tsv")), "dev"
        )

    def get_test_examples(self,data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.tsv")), "test")


    def get_labels(self):
        return ["[PAD]", "B", "I", "O", "X", "[CLS]", "[SEP]"] 

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples

def adj_to_bias(adj, sizes, nhood=1):
        nb_graphs = adj.shape[0]
        mt = np.empty(adj.shape)
        for g in range(nb_graphs):
            mt[g] = np.eye(adj.shape[1])
            for _ in range(nhood):
                mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
            for i in range(sizes[g]):
                for j in range(sizes[g]):
                    if mt[g][i][j] > 0.0:
                        mt[g][i][j] = 1.0
        return -1e9 * (1.0 - mt)

def matrix_generation_tag(sentence, ntokens, max_seq_length):
    adjacent_matrix = np.zeros((max_seq_length, max_seq_length), dtype = np.float32)
    syntactic_feature_matrix = np.zeros((max_seq_length, max_seq_length), dtype = np.float32)
    sentence_dependency_parse = spacy_nlp(sentence)
    pos = []
    tag = []
    dependency = []
    tokens = []
    for token in sentence_dependency_parse:
        pos.append(token.pos_)
        tag.append(token.tag_)
        dependency.append((token.i, token.head.i, token.dep_))
        tokens.append(token.text)

    bool_list = [0] * max_seq_length
    for i,item in enumerate(ntokens):
        if item[0:2] != "##" and item != "[CLS]" and item != "[SEP]" and item != "[PAD]":
            bool_list[i] = 1
    word_index = [i for i,x in enumerate(bool_list) if x==1]
    tem = 0
    for i, item in enumerate(ntokens):
        if item == "[CLS]":
            syntactic_feature_matrix[i][tag_map["CLS"]] = 1.0
        elif item == "[SEP]":
            syntactic_feature_matrix[i][tag_map["SEP"]] = 1.0
        elif item[0:2] == "##":
            syntactic_feature_matrix[i][tag_map["X"]] = 1.0
        elif item == "[PAD]":
            pass
        else:
            syntactic_feature_matrix[i][tag_map[tag[tem]]] = 1.0
            tem += 1
    # for i, item in enumerate(pos):
    #     syntactic_feature_matrix[i][pos_map[item]] = 1.0
    for tail, head, rel in dependency:
        rel = rel.lower()
        try:
            adjacent_matrix[word_index[tail]][word_index[head]] = 1.0
            adjacent_matrix[word_index[head]][word_index[tail]] = 1.0
            syntactic_feature_matrix[word_index[tail]][len(tag_list)+2*dependency_map[rel]] = 1.0
            syntactic_feature_matrix[word_index[head]][len(tag_list)+2*dependency_map[rel]+1] = 1.0
        except:
            global global_input_error
            global_input_error += 1
            print(tokens, word_index)
            print(ntokens)
    adjacent_matrix = np.mat(adjacent_matrix)
    adjacent_matrix = adjacent_matrix[np.newaxis]
    bias_matrix = adj_to_bias(adjacent_matrix, [max_seq_length], nhood=1)
    bias_matrix = np.reshape(bias_matrix, [max_seq_length, max_seq_length])
    # adjacent_matrix = np.mat(adjacent_matrix)
    # syntactic_feature_matrix = np.mat(syntactic_feature_matrix)
    # adjacent_matrix = adjacent_matrix[np.newaxis]
    # syntactic_feature_matrix = syntactic_feature_matrix[np.newaxis]
    return bias_matrix, syntactic_feature_matrix

def matrix_generation_split_tag(sentence, ntokens, max_seq_length):
    adjacent_matrix = np.zeros((max_seq_length, max_seq_length), dtype = np.float32)
    syntactic_feature_matrix = np.zeros((max_seq_length, max_seq_length), dtype = np.float32)
    sentence_dependency_parse = spacy_nlp(sentence)
    pos = []
    tag = []
    dependency = []
    tokens = []
    for token in sentence_dependency_parse:
        pos.append(token.pos_)
        tag.append(token.tag_)
        dependency.append((token.i, token.head.i, token.dep_))
        tokens.append(token.text)

    bool_list = [0] * max_seq_length
    for i,item in enumerate(ntokens):
        if item[0:2] != "##" and item != "[CLS]" and item != "[SEP]" and item != "[PAD]":
            bool_list[i] = 1
    word_index = [i for i,x in enumerate(bool_list) if x==1]

    word_length = []
    for i,item in enumerate(word_index):
        if i < len(word_index)-1:
            word_length.append(word_index[i+1]-word_index[i])
        else:
            word_length.append(ntokens.index("[SEP]")-word_index[len(word_index)-1])
    
    tem = 0
    for i, item in enumerate(ntokens):
        if item == "[CLS]":
            syntactic_feature_matrix[i][tag_map["CLS"]] = 1.0
        elif item == "[SEP]":
            syntactic_feature_matrix[i][tag_map["SEP"]] = 1.0
        elif item[0:2] == "##":
            syntactic_feature_matrix[i][tem_tag] = 1.0 / tem_len
        elif item == "[PAD]":
            pass
        else:
            tem_tag = tag_map[pos[tem]]
            tem_len = word_length[tem]
            syntactic_feature_matrix[i][tem_tag] = 1.0 / tem_len
            tem += 1
    
    for i, item in enumerate(ntokens):
        if item[0:2] == "##":
            adjacent_matrix[i][i-1] = 1.0
            adjacent_matrix[i-1][i] = 1.0
            syntactic_feature_matrix[i-1][len(tag_list)+2*dependency_map["next"]] = 1.0
            syntactic_feature_matrix[i][len(tag_list)+2*dependency_map["next"]+1] = 1.0
        else:
            pass

    for tail, head, rel in dependency:
        rel = rel.lower()
        try:
            adjacent_matrix[word_index[tail]][word_index[head]] = 1.0
            adjacent_matrix[word_index[head]][word_index[tail]] = 1.0
            syntactic_feature_matrix[word_index[tail]][len(tag_list)+2*dependency_map[rel]] = 1.0
            syntactic_feature_matrix[word_index[head]][len(tag_list)+2*dependency_map[rel]+1] = 1.0
        except:
            global global_input_error
            global_input_error += 1
            print(tokens, word_index)
            print(ntokens)
    adjacent_matrix = np.mat(adjacent_matrix)
    adjacent_matrix = adjacent_matrix[np.newaxis]
    bias_matrix = adj_to_bias(adjacent_matrix, [max_seq_length], nhood=1)
    bias_matrix = np.reshape(bias_matrix, [max_seq_length, max_seq_length])
    # adjacent_matrix = np.mat(adjacent_matrix)
    # syntactic_feature_matrix = np.mat(syntactic_feature_matrix)
    # adjacent_matrix = adjacent_matrix[np.newaxis]
    # syntactic_feature_matrix = syntactic_feature_matrix[np.newaxis]
    return bias_matrix, syntactic_feature_matrix

def matrix_generation_duplicate_tag(sentence, ntokens, max_seq_length):
    adjacent_matrix = np.zeros((max_seq_length, max_seq_length), dtype = np.float32)
    syntactic_feature_matrix = np.zeros((max_seq_length, max_seq_length), dtype = np.float32)
    sentence_dependency_parse = spacy_nlp(sentence)
    pos = []
    tag = []
    dependency = []
    tokens = []
    for token in sentence_dependency_parse:
        pos.append(token.pos_)
        tag.append(token.tag_)
        dependency.append((token.i, token.head.i, token.dep_))
        tokens.append(token.text)

    bool_list = [0] * max_seq_length
    for i,item in enumerate(ntokens):
        if item[0:2] != "##" and item != "[CLS]" and item != "[SEP]" and item != "[PAD]":
            bool_list[i] = 1
    word_index = [i for i,x in enumerate(bool_list) if x==1]
    
    tem = 0
    for i, item in enumerate(ntokens):
        if item == "[CLS]":
            syntactic_feature_matrix[i][tag_map["CLS"]] = 1.0
        elif item == "[SEP]":
            syntactic_feature_matrix[i][tag_map["SEP"]] = 1.0
        elif item[0:2] == "##":
            syntactic_feature_matrix[i][tem_tag] = 1.0
        elif item == "[PAD]":
            pass
        else:
            tem_tag = tag_map[tag[tem]]
            syntactic_feature_matrix[i][tem_tag] = 1.0
            tem += 1
    
    for i, item in enumerate(ntokens):
        if item[0:2] == "##":
            adjacent_matrix[i][i-1] = 1.0
            adjacent_matrix[i-1][i] = 1.0
            syntactic_feature_matrix[i-1][len(tag_list)+2*dependency_map["next"]] = 1.0
            syntactic_feature_matrix[i][len(tag_list)+2*dependency_map["next"]+1] = 1.0
        else:
            pass

    for tail, head, rel in dependency:
        rel = rel.lower()
        try:
            adjacent_matrix[word_index[tail]][word_index[head]] = 1.0
            adjacent_matrix[word_index[head]][word_index[tail]] = 1.0
            syntactic_feature_matrix[word_index[tail]][len(tag_list)+2*dependency_map[rel]] = 1.0
            syntactic_feature_matrix[word_index[head]][len(tag_list)+2*dependency_map[rel]+1] = 1.0
        except:
            global global_input_error
            global_input_error += 1
            print(tokens, word_index)
            print(ntokens)
    adjacent_matrix = np.mat(adjacent_matrix)
    adjacent_matrix = adjacent_matrix[np.newaxis]
    bias_matrix = adj_to_bias(adjacent_matrix, [max_seq_length], nhood=1)
    bias_matrix = np.reshape(bias_matrix, [max_seq_length, max_seq_length])
    # adjacent_matrix = np.mat(adjacent_matrix)
    # syntactic_feature_matrix = np.mat(syntactic_feature_matrix)
    # adjacent_matrix = adjacent_matrix[np.newaxis]
    # syntactic_feature_matrix = syntactic_feature_matrix[np.newaxis]
    return bias_matrix, syntactic_feature_matrix

def matrix_generation_duplicate_pos_low_dim(sentence, ntokens, max_seq_length):
    global global_input_error
    syntactic_low_dim = len(pos_list_low_dim) + 2*len(dependency_list_low_dim)
    adjacent_matrix = np.zeros((max_seq_length, max_seq_length), dtype = np.float32)
    syntactic_feature_matrix = np.zeros((max_seq_length, max_seq_length), dtype = np.float32)
    sentence_dependency_parse = spacy_nlp(sentence)
    pos = []
    tag = []
    dependency = []
    tokens = []
    for token in sentence_dependency_parse:
        pos.append(token.pos_)
        tag.append(token.tag_)
        dependency.append((token.i, token.head.i, token.dep_))
        tokens.append(token.text)

    bool_list = [0] * max_seq_length
    for i,item in enumerate(ntokens):
        if item[0:2] != "##" and item != "[CLS]" and item != "[SEP]" and item != "[PAD]":
            bool_list[i] = 1
    word_index = [i for i,x in enumerate(bool_list) if x==1]
    if len(tokens) == len(word_index):
        tem = 0
        for i, item in enumerate(ntokens):
            if item == "[CLS]":
                syntactic_feature_matrix[i][pos_map_low_dim["CLS"]] = 1.0
            elif item == "[SEP]":
                syntactic_feature_matrix[i][pos_map_low_dim["SEP"]] = 1.0
            elif item[0:2] == "##":
                syntactic_feature_matrix[i][tem_pos] = 1.0
            elif item == "[PAD]":
                pass
            else:
                if item in pos_list_low_dim:
                    tem_pos = pos_map_low_dim[pos[tem]]
                else:
                    tem_pos = pos_map_low_dim["OTHERS"]
                syntactic_feature_matrix[i][tem_pos] = 1.0
                tem += 1
    
        for i, item in enumerate(ntokens):
            if item[0:2] == "##":
                adjacent_matrix[i][i-1] = 1.0
                adjacent_matrix[i-1][i] = 1.0
                syntactic_feature_matrix[i-1][len(pos_map_low_dim)+2*dependency_map_low_dim["next"]] = 1.0
                syntactic_feature_matrix[i][len(pos_map_low_dim)+2*dependency_map_low_dim["next"]+1] = 1.0
            else:
                pass

        for tail, head, rel in dependency:
            rel = rel.lower()
            try:
                adjacent_matrix[word_index[tail]][word_index[head]] = 1.0
                adjacent_matrix[word_index[head]][word_index[tail]] = 1.0
                if rel in dependency_list_low_dim:
                    syntactic_feature_matrix[word_index[tail]][len(pos_map_low_dim)+2*dependency_map_low_dim[rel]] = 1.0
                    syntactic_feature_matrix[word_index[head]][len(pos_map_low_dim)+2*dependency_map_low_dim[rel]+1] = 1.0
                else:
                    syntactic_feature_matrix[word_index[tail]][len(pos_map_low_dim)+2*dependency_map_low_dim["others"]] = 1.0
                    syntactic_feature_matrix[word_index[head]][len(pos_map_low_dim)+2*dependency_map_low_dim["others"]+1] = 1.0
            except:
                global_input_error += 1
                print(tokens, word_index)
                print(ntokens)
    else:
        global_input_error += 1
        print(global_input_error)
        print(len(tokens), len(word_index))
        print(tokens, word_index)
        print(ntokens)
    adjacent_matrix = np.mat(adjacent_matrix)
    adjacent_matrix = adjacent_matrix[np.newaxis]
    bias_matrix = adj_to_bias(adjacent_matrix, [max_seq_length], nhood=1)
    bias_matrix = np.reshape(bias_matrix, [max_seq_length, max_seq_length])
    # adjacent_matrix = np.mat(adjacent_matrix)
    # syntactic_feature_matrix = np.mat(syntactic_feature_matrix)
    # adjacent_matrix = adjacent_matrix[np.newaxis]
    # syntactic_feature_matrix = syntactic_feature_matrix[np.newaxis]
    return bias_matrix, syntactic_feature_matrix

def write_tokens(tokens,mode):
    if mode=="test":
        path = os.path.join(FLAGS.output_dir, "token_"+mode+".txt")
        wf = open(path,'a')
        for token in tokens:
            if token!="[PAD]":
                wf.write(token+'\n')
        wf.close()

def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode):
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i
    with open(os.path.join(FLAGS.output_dir,'label2id.pkl'),'wb') as w:
        pickle.dump(label_map,w)

    textlist = example.text.split()
    labellist = example.label.split()
    tokens = []
    labels = []
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = labellist[i]
        for m,  tok in enumerate(token):
            if m == 0:
                labels.append(label_1)
            else:
                labels.append("X")

    # drop if token is longer than max_seq_length
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    ntokens.append("[SEP]")
    segment_ids.append(0)
    label_ids.append(label_map["[SEP]"])

    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    
    # The mask has 1 for real tokens and 0 for padding tokens.
    input_mask = [1] * len(input_ids)
    
    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)
        ntokens.append("[PAD]")
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length

    sentence = example.text
    global Flag_wordpiece
    if Flag_wordpiece == 0:
        bias_matrix, syntactic_feature_matrix = matrix_generation_tag(sentence, ntokens, max_seq_length)
    elif Flag_wordpiece == 1:
        bias_matrix, syntactic_feature_matrix = matrix_generation_split_tag(sentence, ntokens, max_seq_length)
    elif Flag_wordpiece == 2:
        bias_matrix, syntactic_feature_matrix = matrix_generation_duplicate_tag(sentence, ntokens, max_seq_length)
    elif Flag_wordpiece == 3:
        bias_matrix, syntactic_feature_matrix = matrix_generation_duplicate_pos_low_dim(sentence, ntokens, max_seq_length)
    else:
        pass

    if ex_index < 4 : # Examples before model run
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        # tf.logging.info("tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        tf.logging.info("adjacent_matrix: %s" % " ".join([str(x) for x in bias_matrix]))
        tf.logging.info("syntactic_feature_matrix: %s" % " ".join([str(x) for x in syntactic_feature_matrix]))
        #tf.logging.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        bias_matrix = bias_matrix,
        syntactic_feature_matrix = syntactic_feature_matrix
        #label_mask = label_mask
    )
    write_tokens(ntokens,mode)
    return feature

def filed_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file, mode=None
):
    writer = tf.python_io.TFRecordWriter(output_file)

    # examples = examples[0:100]

    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer,mode)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f
        def create_float_feature(values):
            f = tf.train.Feature(float_list=tf.train.FloatList(value=values.flatten()))
            return f
        def create_byte_feature(values):
            f = tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode('utf-8') if type(value)==str else value for value in values]))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        features["bias_matrix"] = create_float_feature(feature.bias_matrix)
        features["syntactic_feature_matrix"] = create_float_feature(feature.syntactic_feature_matrix)
        #features["label_mask"] = create_int_feature(feature.label_mask)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())

def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "bias_matrix": tf.VarLenFeature(tf.float32),
        "syntactic_feature_matrix": tf.VarLenFeature(tf.float32),
        # "label_ids":tf.VarLenFeature(tf.int64),
        #"label_mask": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            if t.dtype == tf.float32:
                t = tf.sparse_tensor_to_dense(t)
                t = tf.reshape(t, [seq_length, seq_length])
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d
    return input_fn

def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings,
                 bias_matrix, syntactic_feature_matrix,
                 dropout_rate=1.0, lstm_size=512, cell='lstm', num_layers=1):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    output_layer = model.get_sequence_output()
    max_seq_length = output_layer.shape[1].value
    used = tf.sign(tf.abs(input_ids))
    lengths = tf.reduce_sum(used, reduction_indices=1)  # [batch_size] 大小的向量，包含了当前batch中的序列长度

    gat_bilstm_crf = GAT_BILSTM_CRF(embedded_chars=output_layer, hidden_unit=lstm_size, cell_type=cell, num_layers=num_layers,
                          dropout_rate=dropout_rate, initializers=initializers, num_labels=num_labels,
                          seq_length=max_seq_length, labels=labels, lengths=lengths, is_training=is_training,
                          bias_matrix=bias_matrix, syntactic_feature_matrix=syntactic_feature_matrix)

    (loss, logits, trans, predict) = gat_bilstm_crf.add_gat_blstm_crf_layer()
    return loss, logits, trans, predict

def create_model_init(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings,
                 bias_matrix, syntactic_feature_matrix,
                 dropout_rate=1.0, lstm_size=512, cell='lstm', num_layers=1):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    output_layer = model.get_sequence_output()
    max_seq_length = output_layer.shape[1].value
    used = tf.sign(tf.abs(input_ids))
    lengths = tf.reduce_sum(used, reduction_indices=1)  # [batch_size] 大小的向量，包含了当前batch中的序列长度

    gat_bilstm_crf = GAT_BILSTM_CRF(embedded_chars=output_layer, hidden_unit=lstm_size, cell_type=cell, num_layers=num_layers,
                          dropout_rate=dropout_rate, initializers=initializers, num_labels=num_labels,
                          seq_length=max_seq_length, labels=labels, lengths=lengths, is_training=is_training,
                          bias_matrix=bias_matrix, syntactic_feature_matrix=syntactic_feature_matrix)
    output_layer = gat_bilstm_crf.get_gat_embeddings()

    hidden_size = output_layer.shape[-1].value

    output_weight = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer()
    )
    with tf.variable_scope("loss"):
        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        output_layer = tf.reshape(output_layer, [-1, hidden_size])
        logits = tf.matmul(output_layer, output_weight, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, num_labels])
        # mask = tf.cast(input_mask,tf.float32)
        # loss = tf.contrib.seq2seq.sequence_loss(logits,labels,mask)
        # return (loss, logits, predict)
        ##########################################################################
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_sum(per_example_loss)
        probabilities = tf.nn.softmax(logits, axis=-1)
        predict = {"predict": tf.argmax(probabilities,axis=-1), "log_probs": log_probs}
        return (loss, per_example_loss, logits, predict)
        ##########################################################################

def create_model_concat(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings,
                 bias_matrix, syntactic_feature_matrix,
                 dropout_rate=1.0, lstm_size=512, cell='lstm', num_layers=1):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    output_layer_bert = model.get_sequence_output()
    
    output_layer = tf.concat([output_layer_bert, syntactic_feature_matrix], axis = -1)

    hidden_size = output_layer.shape[-1].value

    output_weight = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer()
    )
    with tf.variable_scope("loss"):
        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        output_layer = tf.reshape(output_layer, [-1, hidden_size])
        logits = tf.matmul(output_layer, output_weight, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, num_labels])
        # mask = tf.cast(input_mask,tf.float32)
        # loss = tf.contrib.seq2seq.sequence_loss(logits,labels,mask)
        # return (loss, logits, predict)
        ##########################################################################
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_sum(per_example_loss)
        probabilities = tf.nn.softmax(logits, axis=-1)
        predict = {"predict": tf.argmax(probabilities,axis=-1), "log_probs": log_probs}
        return (loss, per_example_loss, logits, predict)
        ##########################################################################

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        bias_matrix = features["bias_matrix"]
        syntactic_feature_matrix = features["syntactic_feature_matrix"]
        #label_mask = features["label_mask"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        if Flag_init == 0:
            total_loss, logits, trans, predictsDict = create_model(
                bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
                num_labels, False, bias_matrix, syntactic_feature_matrix)
        elif Flag_init == 1:
            total_loss, per_example_loss, logits, predictsDict = create_model_init(
                bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
                num_labels, False, bias_matrix, syntactic_feature_matrix)
        elif Flag_init == 2:
            total_loss, per_example_loss, logits, predictsDict = create_model_concat(
                bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
                num_labels, False, bias_matrix, syntactic_feature_matrix)      
        
        predictsDict["input_mask"] = input_mask
        tvars = tf.trainable_variables()
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()
                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            # 原模型构建方法
            pred_ids = predictsDict["predict"]
            def metric_fn(label_ids, pred_ids):
                return {
                    "eval_loss": tf.metrics.mean_squared_error(labels=label_ids, predictions=pred_ids),
                }
            def metric_fn_bert(per_example_loss, label_ids, logits, num_labels):
                # def metric_fn(label_ids, logits):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                precision = tf_metrics.precision(label_ids,predictions,num_labels,[1,2],average="macro")
                recall = tf_metrics.recall(label_ids,predictions,num_labels,[1,2],average="macro")
                f = tf_metrics.f1(label_ids,predictions,num_labels,[1,2],average="macro")
                #
                return {
                    "eval_precision":precision,
                    "eval_recall":recall,
                    "eval_f": f,
                    #"eval_loss": loss,
                }
            if Flag_init == 0:
                eval_metrics = (metric_fn, [label_ids, pred_ids])
            else:
                eval_metrics = (metric_fn_bert, [per_example_loss, label_ids, logits, num_labels])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            # 模仿biobert构建
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode = mode, predictions = predictsDict, scaffold_fn = scaffold_fn
            )
            # 原模型
            # output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            #     mode = mode, predictions = pred_ids, scaffold_fn = scaffold_fn
            # )
        return output_spec
    return model_fn

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    processors = {
        "ner": NerProcessor
    }
    #if not FLAGS.do_train and not FLAGS.do_eval:
    #    raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))
    
    tf.gfile.MakeDirs(FLAGS.output_dir)
    
    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        filed_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        filed_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", len(eval_examples))
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        eval_steps = None
        if FLAGS.use_tpu:
            eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)
        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)
        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
    if FLAGS.do_predict:
        label2idPath = os.path.join(FLAGS.output_dir, 'label2id.pkl')
        if os.path.exists(label2idPath):
            with open(label2idPath,'rb') as rf:
                label2id = pickle.load(rf)
                id2label = {value:key for key,value in label2id.items()}
        else:
            tf.logging.info("***** Warning! label2id.pkl not exist *****")
            tf.logging.info("***** Creating label2id.pkl during predict (not recommended) *****")
            label2id = {}
            for i, label in enumerate(label_list):
                label2id[label] = i
                id2label = {value:key for key,value in label2id.items()}
            with open(label2idPath,'wb') as w:
                pickle.dump(label2id,w)
                
        token_path = os.path.join(FLAGS.output_dir, "token_test.txt")
        if os.path.exists(token_path):
            os.remove(token_path)
        token_modi_path = os.path.join(FLAGS.output_dir, "token_modi_test.txt")
        if os.path.exists(token_modi_path):
            os.remove(token_modi_path)
        
        predict_examples = processor.get_test_examples(FLAGS.data_dir)

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        filed_based_convert_examples_to_features(predict_examples, label_list,
                                                FLAGS.max_seq_length, tokenizer,
                                                predict_file,mode="test")
                            
        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", len(predict_examples))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
        tf.logging.info("  Example of predict_examples = %s", predict_examples[0].text)
        if FLAGS.use_tpu:
            # Warning: According to tpu_estimator.py Prediction on TPU is an
            # experimental feature and hence not supported here
            raise ValueError("Prediction in TPU not supported")
        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)
        prf = estimator.evaluate(input_fn=predict_input_fn, steps=None)
        
        tf.logging.info("***** token-level evaluation results *****")
        for key in sorted(prf.keys()):
            tf.logging.info("  %s = %s", key, str(prf[key]))
        
        output_predict_file = os.path.join(FLAGS.output_dir, "label_test.txt")
        # 模仿biobert构建
        with open(output_predict_file,'w') as writer:
            for resultIdx, prediction in enumerate(result):
                # Fix for "padding occurrence amid sentence" error 
                # (which occasionally cause mismatch between the number of predicted tokens and labels.)
                assert len(prediction["predict"]) == len(prediction["input_mask"]), "len(prediction['predict']) != len(prediction['input_mask']) Please report us!"
                predLabelSent = []
                for predLabel, inputMask in zip(prediction["predict"], prediction["input_mask"]):
                    # predLabel : Numerical Value
                    if inputMask != 0:
                        if predLabel == label2id['[PAD]']:
                            predLabelSent.append('O')
                        else:
                            predLabelSent.append(id2label[predLabel])
                output_line = "\n".join(predLabelSent) + "\n"
                writer.write(output_line)

if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
    print(global_input_error)


