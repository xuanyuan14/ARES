import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def tokenize_file(tokenizer, input_file, output_file, file_type):
    total_size = sum(1 for _ in open(input_file))
    with open(output_file, 'w') as outFile:
        for line in tqdm(open(input_file), total=total_size,
                         desc=f"Tokenize: {os.path.basename(input_file)}"):
            if file_type == "query":
                seq_id, text = line.split("\t")
            else:
                line = json.loads(line.strip())

            tokens = tokenizer.tokenize(text)
            ids = tokenizer.convert_tokens_to_ids(tokens)[: 512]
            outFile.write(json.dumps(
                {"id": seq_id, "ids": ids}
            ))
            outFile.write("\n")


def tokenize_queries(tokenizer, input_file, output_file):
    total_size = sum(1 for _ in open(input_file))
    f = open(output_file, 'w')
    with open(input_file, 'r') as r:
        for line in tqdm(r, total=total_size):
            query_id, text = line.strip().split('\t')
            tokens = tokenizer.tokenize(text)
            ids = tokenizer.convert_tokens_to_ids(tokens)[: 512]
            f.write(json.dumps({
                'query_id': query_id,
                'query': ids
            }) + '\n')
    f.close()


def tokenize_docs(tokenizer, input_file, output_file):
    total_size = sum(1 for _ in open(input_file))
    f = open(output_file, 'w')
    with open(input_file, 'r') as r:
        for line in tqdm(r, total=total_size):
            line = json.loads(line.strip())
            tokens = tokenizer.tokenize(line['doc'])
            ids = tokenizer.convert_tokens_to_ids(tokens)[: 512]
            f.write(json.dumps({
                'id': line['doc_id'],
                'contents': ids
            }) + '\n')
    f.close()


def tokenize_pairwise(tokenizer, input_file, output_file):
    total_size = sum(1 for _ in open(input_file))
    f = open(output_file, 'w')
    with open(input_file, 'r') as r:
        for line in tqdm(r, total=total_size):
            line = json.loads(line.strip())
            tokens = tokenizer.tokenize(line['query'])
            query_ids = tokenizer.convert_tokens_to_ids(tokens)[: 512]

            tokens = tokenizer.tokenize(line['doc_pos'])
            pos_ids = tokenizer.convert_tokens_to_ids(tokens)[: 512]

            tokens = tokenizer.tokenize(line['doc_neg'])
            neg_ids = tokenizer.convert_tokens_to_ids(tokens)[: 512]
            f.write(json.dumps({
                'query': query_ids,
                'doc_pos': pos_ids,
                'doc_neg': neg_ids
            }) + '\n')
    f.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--vocab_dir", default='bert-base-uncased', type=str)
    parser.add_argument("--type", default='query', type=str)
    parser.add_argument("--input", default='', type=str, required=True)
    parser.add_argument("--output", default='', type=str, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)

    if args.type == "query":
        tokenize_queries(tokenizer, args.input, args.output)
    elif args.type == "doc":
        tokenize_docs(tokenizer, args.input, args.output)
    elif args.type == "triples":
        tokenize_pairwise(tokenizer, args.input, args.output)