# BioByGANS
This repository provides the code for BioByGANS, a BioBERT&SpaCy - Graph Attention Network - Softmax based method for biomedical entity recognition. This project is done by Information Center, Academy of Miliary Medical Sciences.

## Download
The [BioBERT](https://github.com/dmis-lab/biobert) pre-trained weights are provided by Lee et al. from DMIS-Lab, of which the paper is [BioBERT: a pre-trained biomedical language representation model for biomedical text mining](http://doi.org/10.1093/bioinformatics/btz682).  

The pre-trained weights we used in this project is [BioBERT-Base v1.1 (+ PubMed 1M)](https://drive.google.com/file/d/1R84voFKHfWV9xjzeLzWBbmY1uOMYpnyD/view?usp=sharing) - based on BERT-base-Cased (same vocabulary).  

For more pre-trained weights, please visit [BioBERT](https://github.com/dmis-lab/biobert).  

## Datasets
We use 8 open-source biomedical corpora covering genes, proteins, diseases, and chemicals, including BC2GM, JNLPBA, Species-800, LINNAEUS, BC5CDR, NCBI-Disease, BC4CHEMD.  

[Datasets](https://drive.google.com/open?id=1OletxmPYNkz2ltOr9pyT0b0iBtUWxslh) used in this project are pre-processed and provided by [Lee et al.](http://doi.org/10.1093/bioinformatics/btz682), which has been tagged in BIO formats.  You can first download the corpora via the above link to further run our program.  

The NLP tool used in this project is [SpaCy](https://spacy.io/). SpaCy is a powerful and lightweight NLP tool for various languages, of which the functions include tokenizer, tagger, parser, etc. Specifically, we use the version 3.2.1, with the language package [en_core_web_trf](https://spacy.io/models/en#en_core_web_trf) to achieve our results. SpaCy with en_core_web_trf achieves the precision of 100% on tokenization, 98% on part of speech tagging, 95% on sentence segmentation, 95% on unlabeled dependencies, and 90% on general named entity recognition.

## Installation
Our project is based on Tensorflow 1.15 (python version = 3.7.11). The model can be installed by following instructions:  
```bash
$ git clone https://github.com/zxw1995shawn/BioByGANS.git
$ cd BioByGANS
$ pip install -r requirements.txt
```
The model can also be implemented in CPU environment, just through changing the tensorflow as a CPU version.

## Inplementing BioNER
We take the chemical NER on BC5CDR-disease as an example in this section. Setting several environment variables as:
```bash
$ export BIOBERT_DIR=./biobert_v1.1_pubmed
$ export NER_DIR=./corpus/BC5CDR-chem
$ export OUTPUT_DIR=./ner_outputs
$ mkdir -p OUTPUT_DIR
```
Following command runs NER code:
```bash
$ python run_BioByGANS.py --do_train=true --do_eval=true --do_predict=true --vocab_file=$BIOBERT_DIR/vocab.txt --bert_config_file=$BIOBERT_DIR/bert_config.json --init_checkpoint=$BIOBERT_DIR/model.ckpt-1000000 --num_train_epochs=50.0 --max_seq_length=256 --train_batch_size=32 --learning_rate=3e-5 num_gat_heads=12 num_gat_units=64 --data_dir=$NER_DIR --output_dir=$OUTPUT_DIR
```
If you use CPUs to run this project, corresponding command is as follows:
```bash
$ python run_BioByGANS_cpu.py --do_train=true --do_eval=true --do_predict=true --vocab_file=$BIOBERT_DIR/vocab.txt --bert_config_file=$BIOBERT_DIR/bert_config.json --init_checkpoint=$BIOBERT_DIR/model.ckpt-1000000 --num_train_epochs=50.0 --max_seq_length=256 --train_batch_size=32 --learning_rate=3e-5 num_gat_heads=12 num_gat_units=64 --data_dir=$NER_DIR --output_dir=$OUTPUT_DIR
```
After training, developing and testing, Use `./biocodes/ner_detokenize.py` to obtain word-level prediction file, and use `./biocodes/conlleval.pl` to get results.
```bash
$ python biocodes/ner_detokenize.py --token_test_path=$OUTPUT_DIR/token_test.txt --label_test_path=$OUTPUT_DIR/label_test.txt --answer_path=$NER_DIR/test.tsv --output_dir=$OUTPUT_DIR
$ perl biocodes/conlleval.pl < $OUTPUT_DIR/NER_result_conll.txt
```
The results for BC5CDR-chem corpus is like:
```
processed 124750 tokens with 5385 phrases; found: 5407 phrases; correct: 5083.
accuracy:  99.59%; precision:  94.00%; recall:  94.41%; FB1:  94.20
             MISC: precision:  94.00%; recall:  94.41%; FB1:  94.20  5407
``` 
## Contact Information
For help or suggestions for BioByGANS, please concact Xiangwen Zheng(`xwzheng60602@163.com`).
