import os
import sys
sys.path.insert(0, '../')

from tqdm import tqdm
import json
import torch
import numpy as np
import pandas as pd
from datetime import timedelta

from model.modeling import ARESReranker


if __name__ == "__main__":
    model_path = "path/to/model"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ARESReranker.from_pretrained(model_path).to(device)

    query1 = "What is the best way to get to the airport"
    query2 = "what do you like to eat?"
    
    doc1 = "The best way to get to the airport is to take the bus"
    doc2 = "I like to eat apples"


    ### Score a batch of q-d pairs
    qd_pairs = [
        (query1, doc1), (query1, doc2),
        (query2, doc1), (query2, doc2)
    ]

    score = model.score(qd_pairs)
    print("qd scores", score)

    ### Rerank a single query
    score = model.rerank_query(query1, [doc1, doc2])
    print("query1 scores", score)

    ### Rerank a batch of queries
    query1_topk = [ doc1, doc2 ]
    query2_topk = [ doc1, doc2 ]

    score = model.rerank([query1, query2], [query1_topk, query2_topk])
