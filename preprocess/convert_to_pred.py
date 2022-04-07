from tqdm import tqdm
from collections import defaultdict
import argparse

def trec_to_pred(args):
    trec = defaultdict(dict)
    with open(args.input_trec, 'r') as f:
        for line in f:
            qid, _, docid, rank, score, _ = line.strip().split(' ')
            trec[qid][docid] = score

    f = open(args.output, 'w')
    with open(args.qrels, 'r') as r:
        for line in r:
            line = line.strip().split()
            qid = line[1].split(':')[1]
            docid = line[-7]
            if docid in trec[qid]:
                f.write(trec[qid][docid] + '\n')
            else:
                f.write('0.0\n')

    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_trec", default='', type=str, required=True)
    parser.add_argument("--output", default='', type=str, required=True)
    parser.add_argument("--qrels", default='', type=str, required=True)
    args = parser.parse_args()

    trec_to_pred(args)