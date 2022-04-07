![img](imgs/ARES_black.jpg)

<p align="center">

<a href="www.thuir.cn">
<img alt="THUIR" src="https://img.shields.io/badge/THUIR-ver%201.0-blueviolet">
</a>
<a href="./LICENSE">
<img alt="License" src="https://img.shields.io/badge/License-Apache2.0-blue.svg">
</a>
<a>
<img alt="made-with-python" src="https://img.shields.io/badge/Made%20with-Python-red.svg">
</a>
<a>
<img alt="code-size" src="https://img.shields.io/github/languages/code-size/xuanyuan14/ARES-master?color=green">
</a>

</p>

## Introduction
This codebase contains source-code of the Python-based implementation (ARES) of our SIGIR 2022 paper.
  - [Chen, Jia, et al. "Axiomatically Regularized Pre-training for Ad hoc Search." To Appear in the Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2022.](https://xuanyuan14.github.io/files/SIGIR22Chen.pdf)

## Requirements
* python 3.7
* torch==1.9.0
* transformers==4.9.2
* tqdm, nltk, numpy, boto3
* [trec_eval](https://github.com/usnistgov/trec_eval) for evaluation on TREC DL 2019
* [anserini](https://github.com/castorini/anserini) for generating "RANK" axiom scores

## Why this repo?
In this repo, you can pre-train ARES<sub>simple</sub> and Transformer<sub>ICT</sub> models, and fine-tune all pre-trained models with the same architecture as BERT. The papers are listed as follows:
* BERT ([Bert: Pre-training of deep bidirectional transformers for language understanding](https://arxiv.org/pdf/1810.04805.pdf&usg=ALkJrhhzxlCL6yTht2BRmH9atgvKFxHsxQ))
* Transformer<sub>ICT</sub> ([Latent Retrieval for Weakly Supervised Open Domain Question Answering.](https://arxiv.org/pdf/1906.00300))
* PROP ([PROP: Pre-training with representative words prediction for ad-hoc retrieval.](https://dl.acm.org/doi/pdf/10.1145/3437963.3441777))
* ARES ([Axiomatically Regularized Pre-training for Ad hoc Search.](https://xuanyuan14.github.io/files/SIGIR22Chen.pdf))

You can download the pre-trained ARES checkpoint [ARES<sub>simple</sub>](https://drive.google.com/file/d/1QvJ-hs6VtK4nlrlFkzPZAXfTtY-QjTiU/view?usp=sharing) from Google drive and extract it.

## Pre-training Data

### Download data
Download the **MS MARCO** corpus from the official [website](https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz).  
Download the **ADORE+STAR Top100 Candidates** files from this [repo](https://github.com/jingtaozhan/DRhard).

### Pre-process data
To save memory, we store most files using the numpy `memmap` or `jsonl` format in the `./preprocess` directory.

Document files:
* `doc_token_ids.memmap`: each line is the token ids for a document
* `docid2idx.json`: `{docid: memmap_line_id}`

Query files:
* `queries.doctrain.jsonl`: MS MARCO training queries `{"id" qid, "ids": token_ids}` for each line
* `queries.docdev.jsonl`: MS MARCO validating queries `{"id" qid, "ids": token_ids}` for each line 
* `queries.dl2019.jsonl`: TREC DL 2019 queries `{"id" qid, "ids": token_ids}` for each line

Human label files:
* `msmarco-doctrain-qrels.tsv`: `qid 0 docid 1` for training set
* `dev-qrels.txt`: `qid relevant_docid` for validating set
* `2019qrels-docs.txt`: `qid relevant_docid` for TREC DL 2019 set

Top 100 candidate files:
* `train.rank.tsv`, `dev.rank.tsv`, `test.rank.tsv`: `qid docid rank` for each line

Pseudo queries and axiomatic features:
* `doc2qs.jsonl`: `{"docid": docid, "queries": [qids]}` for each line
* `sample_qs_token_ids.memmap`: each line is the token ids for a pseudo query
* `sample_qid2id.json`: `{qid: memmap_line_id}`
* `axiom.memmap`: axiom can be one of the `['rank', 'prox-1', 'prox-2', 'rep-ql', 'rep-tfidf', 'reg', 'stm-1', 'stm-2', 'stm-3']`, each line is an axiomatic score for a query


## Quick Start
Note that to accelerate the training process, we adopt the parallel training technique. The scripts for pre-training and fine-tuning are as follow: 

### Pre-training

```shell
export BERT_DIR=/path/to/bert-base/
export XGB_DIR=/path/to/xgboost.model

cd pretrain

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 NCCL_BLOCKING_WAIT=1 \
python  -m torch.distributed.launch --nproc_per_node=6 --nnodes=1 train.py \
        --model_type ARES \
        --PRE_TRAINED_MODEL_NAME BERT_DIR \
        --gpu_num 6 --world_size 6 \
        --MLM --axiom REP RANK REG PROX STM \
        --clf_model XGB_DIR
```
Here model type can be `ARES` or `ICT`.

### Zero-shot evaluation (based on AS top100)
```shell
export MODEL_DIR=/path/to/ares-simple/
export CKPT_NAME=ares.ckpt

cd finetune

CUDA_VISIBLE_DEVICES=0 python train.py \
        --test \
        --PRE_TRAINED_MODEL_NAME MODEL_DIR \
        --model_type ARES \
        --model_name ARES_simple \
        --load_ckpt \
        --model_path CKPT_NAME
```
You can get:
```bash
#####################
<----- MS Dev ----->
MRR @10: 0.2991
MRR @100: 0.3130
QueriesRanked: 5193
#####################
```
on MS MARCO dev set and:
```bash
#############################
<--------- DL 2019 --------->
QueriesRanked: 43
nDCG @10: 0.5955
nDCG @100: 0.4863
#############################
```
on DL 2019 set.

### Fine-tuning
```shell
export MODEL_DIR=/path/to/ares-simple/

cd finetune

CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_BLOCKING_WAIT=1 \
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 train.py \
        --model_type ARES \
        --distributed_train \
        --PRE_TRAINED_MODEL_NAME MODEL_DIR \
        --gpu_num 4 --world_size 4 \
        --model_name ARES_simple
```

### Visualization
```shell
export MODEL_DIR=/path/to/ares-simple/
export SAVE_DIR=/path/to/output/
export CKPT_NAME=ares.ckpt

cd visualization

CUDA_VISIBLE_DEVICES=0 python visual.py \
    --PRE_TRAINED_MODEL_NAME MODEL_DIR \
    --model_name ARES_simple \
    --visual_q_num 1 \
    --visual_d_num 5 \
    --save_path SAVE_DIR \
    --model_path CKPT_NAME
```

##Results
Zero-shot performance:

| Model Name |   MS MARCO MRR@10 |  MS MARCO MRR@100  | DL NDCG@10 |  DL NDCG@100 |  COVID  |  EQ |  
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |   
| BM25     |  0.2962 | 0.3107  |  0.5776 | 0.4795 | 0.4857 | 0.6690 |  
| BERT     |  0.1820 | 0.2012  |  0.4059 | 0.4198 | 0.4314 | 0.6055 |   
| PROP<sub>wiki</sub> |  0.2429 | 0.2596  |  0.5088 | 0.4525 | 0.4857 | 0.5991 |  
| PROP<sub>marco</sub>   |  0.2763 |  0.2914  | 0.5317 | 0.4623 | 0.4829 | 0.6454 |  
| ARES<sub>strict</sub>  |  0.2630 |  0.2785  | 0.4942 | 0.4504 | 0.4786 | 0.6923 |  
| ARES<sub>hard</sub>  |  0.2627 |  0.2780  | 0.5189 | 0.4613 | 0.4943 | 0.6822 |  
| ARES<sub>simple</sub>  |  0.2991 |  0.3130 | 0.5955 | 0.4863 | 0.4957 | 0.6916 |   


Few-shot performance:
![img](imgs/few-shot-metric.png)

Visualization (attribution values have been normalized within a document):
![img](imgs/ARES_simple_ag.png)

## Citation
If you find our work useful, please do not save your star and cite our work:
```
@inproceedings{chen2022axiomatically,
  title={Axiomatically Regularized Pre-training for Ad hoc Search},
  author={Chen, Jia and Liu, Yiqun and Fang, Yan and Mao, Jiaxin and Fang, Hui and Yang, Shenghao and Xie, Xiaohui and Zhang, Min and Ma, Shaoping},
  booktitle={Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2022}
}
```


## Notice
* Please make sure that all the pre-trained model parameters have been loaded correctly, or the zero-shot and the fine-tuning performance will be greatly impacted.
* We welcome anyone who would like to contribute to this repo. 🤗
* If you have any other questions, please feel free to contact me via [chenjia0831@gmail.com]() or open an issue.
* Code for data preprocessing will come soon. Please stay tuned~ 
