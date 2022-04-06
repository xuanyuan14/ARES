## Data Preprocess

Since different datasets require different pre-processing, we only provide some helper functions and scripts here.

### Anserini Scripts

We use BM25 implemented by `anserini` to perform first-stage retrieval.

Please make sure you have correctly installed `anserini` and `pyserini`.

### Tokenize

You can pre-tokenize your dataset offline for faster training.
```bash
python convert_tokenize.py \
    --vocab_dir {path_to_vocab} \
    --type {'query', 'doc', 'triples'} \
    --input {path_to_input} \
    --output {path_to_output}
```
File format:

* query: `qid \t query` for each line
* doc: `{"id": docid, "contents": doc}` for each line
* triples: `{"query": query_text, "doc_pos": positive_doc, "doc_neg": negative_doc}` for each line

### Small Datasets

#### TREC-COVID

We follow the same data preprocess as `OpenMatch`, please refer to [experiments-treccovid](https://github.com/thunlp/OpenMatch/blob/master/docs/experiments-treccovid.md)

#### Robust04

We use BM25 to generate Top-200 candidates for each query, and the fine-tuning procedure is similar to MS-MARCO

#### MQ2007

We use BM25 to generate Top-200 candidates for each query, and the fine-tuning procedure is similar to MS-MARCO

Note that `trec_eval` cannot be used to compute metrics for MQ2007 directly. You should first convert the `trec` output file and use `Eval4.0.pl` for evaluation. `Eval4.0.pl` is from [LETOR4.0](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/letor-4-0/)
```bash
python convert_to_pred.py \
    --input_trec {path_to_trec_output} \
    --qrels {path_to_qrels} \
    --output {path_to_output}

perl Eval4.0.pl {path_to_qrels} {path_to_output} ./eval_result 0
```
