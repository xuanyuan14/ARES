'''
@ref: Axiomatically Regularized Pre-training for Ad hoc Search
@author: Jia Chen, Yiqun Liu, Yan Fang, Jiaxin Mao, Hui Fang, Shenghao Yang, Xiaohui Xie, Min Zhang, Shaoping Ma.
'''
# encoding: utf-8
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


class TrainQDDatasetPairwise(Dataset):
    def __init__(self, q_ids, d_ids, q_dict, d_dict, did2idx, config, labels, mode='train'):
        self.q_ids = q_ids
        self.d_ids = d_ids
        self.q_dict = q_dict
        self.d_dict = d_dict
        self.did2idx = did2idx
        self.labels = labels
        self.mode = mode
        self.config = config

    def __len__(self):
        return len(self.q_ids)

    def __getitem__(self, item):
        cls_id, sep_id = 101, 102
        q_id = self.q_ids[item]
        d_id = self.d_ids[item]

        q_id = q_id[0]
        pos_did, neg_did = d_id[0], d_id[1]

        query_input_ids, pos_doc_input_ids, neg_doc_input_ids = self.q_dict[str(q_id)], self.d_dict[self.did2idx[pos_did]].tolist(), \
                                                                self.d_dict[self.did2idx[neg_did]].tolist()
        query_input_ids = query_input_ids[: self.config.max_q_len]
        max_passage_length = self.config.max_len - 3 - len(query_input_ids)

        pos_doc_input_ids = pos_doc_input_ids[:max_passage_length]
        neg_doc_input_ids = neg_doc_input_ids[:max_passage_length]

        pos_input_ids = [cls_id] + query_input_ids + [sep_id] + pos_doc_input_ids + [sep_id]
        neg_input_ids = [cls_id] + query_input_ids + [sep_id] + neg_doc_input_ids + [sep_id]

        pos_token_type_ids = [0] * (2 + len(query_input_ids)) + [1] * (1 + len(pos_doc_input_ids))
        neg_token_type_ids = [0] * (2 + len(query_input_ids)) + [1] * (1 + len(neg_doc_input_ids))

        pos_token_ids = np.array(pos_input_ids)
        neg_token_ids = np.array(neg_input_ids)
        token_ids = np.stack((pos_token_ids.flatten(), neg_token_ids.flatten()))

        pos_attention_mask = np.int64(pos_token_ids > 0)
        neg_attention_mask = np.int64(neg_token_ids > 0)
        attention_mask = np.stack((pos_attention_mask, neg_attention_mask))

        pos_token_type_ids = np.array(pos_token_type_ids)
        neg_token_type_ids = np.array(neg_token_type_ids)
        token_type_ids = np.stack((pos_token_type_ids, neg_token_type_ids))

        return {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
        }


class TestQDDataset(Dataset):
    def __init__(self, q_ids, d_ids, token_ids, attention_mask, token_type_ids, mode='test'):
        self.q_ids = q_ids
        self.d_ids = d_ids
        self.token_ids = token_ids
        self.attention_mask = attention_mask
        self.token_type_ids= token_type_ids
        self.mode = mode

    def __len__(self):
        return len(self.q_ids)

    def __getitem__(self, item):
        q_id = self.q_ids[item]
        d_id = self.d_ids[item]
        token_ids = np.array(self.token_ids[item])
        attention_mask = np.array(self.attention_mask[item])
        token_type_ids = np.array(self.token_type_ids[item])

        return {
                "q_id": q_id,
                "d_id": d_id,
                'token_ids': token_ids.flatten(),
                'attention_mask': attention_mask.flatten(),
                'token_type_ids': token_type_ids.flatten(),
        }


# [CLS] q [SEP] d [SEP]
def get_train_qd_loader(df_qds, train_top100, q_dict, d_dict, did2idx, config, mode='train'):
    q_max_len, max_len, batch_size = config.max_q_len, config.max_len, config.batch_size
    q_ids = df_qds[0].values.tolist()
    d_ids = df_qds[2].values.tolist()

    qd_dict = {}
    for q_id, d_id in zip(q_ids, d_ids):
        if q_id not in qd_dict:
            qd_dict[q_id] = []
        qd_dict[q_id].append(d_id)

    top100_dict = {}
    top_qids = train_top100[0].values.tolist()
    top_dids = train_top100[1].values.tolist()
    for qid, did in zip(top_qids, top_dids):
        if qid not in top100_dict:
            top100_dict[qid] = []
        top100_dict[qid].append(did)

    new_q_ids, new_d_ids, labels = [], [], []

    q_num = len(q_ids)
    for idx in tqdm(range(q_num), desc=f"Loading train q-d progress"):  # top 20 hard candidate？需要改回来嘛？
        this_qid = q_ids[idx]
        neg_cands = set(top100_dict[this_qid]) - set(qd_dict[this_qid])
        neg_cands = list(neg_cands)
        neg_dids = random.sample(neg_cands, config.neg_docs_per_q)
        for i in range(config.neg_docs_per_q):
            new_q_ids.append([this_qid])
            new_d_ids.append([d_ids[idx], neg_dids[i]])
            labels.append([1, 0])

    print('Loading tokens...')
    ds = TrainQDDatasetPairwise(
        q_ids=new_q_ids,
        d_ids=new_d_ids,
        q_dict=q_dict,
        d_dict=d_dict,
        did2idx=did2idx,
        config=config,
        labels=labels,
        mode='train'
    )
    batch_size = batch_size // 2

    if config.distributed_train:
        sampler = DistributedSampler(ds, num_replicas=config.world_size, rank=config.local_rank)
        return DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=0,
            sampler=sampler
        )
    else:
        if mode == 'train':
            return DataLoader(
                ds,
                batch_size=batch_size,
                num_workers=0,
                shuffle=True,
            )
        else:
            return DataLoader(
                ds,
                batch_size=batch_size,
                num_workers=0,
                shuffle=False,
            )


def get_test_qd_loader(top100qd, q_dict, d_dict, did2idx, config):
    cls_id, sep_id = 101, 102
    q_ids = top100qd[0].values.tolist()
    d_ids = top100qd[1].values.tolist()

    qd_dict = {}
    for q_id, d_id in zip(q_ids, d_ids):
        if q_id not in qd_dict:
            qd_dict[q_id] = []
        qd_dict[q_id].append(d_id)

    q_num = len(q_dict)
    qids = list(set(q_dict.keys()))
    tokens_np = np.zeros((q_num * 100, config.max_len), dtype='int32')  # (q_num * 100)  x 512
    token_type_np = np.zeros((q_num * 100, config.max_len), dtype='int32')  # (q_num * 100)  x 512

    new_q_ids, new_d_ids = [], []
    for idx in tqdm(range(len(qids)), desc=f"Loading test q-d pair progress"):
        this_qid = qids[idx]

        query_input_ids = q_dict[str(this_qid)]
        query_input_ids = query_input_ids[: config.max_q_len]
        max_passage_length = config.max_len - 3 - len(query_input_ids)

        dids = qd_dict[int(this_qid)]
        assert len(dids) == 100
        for rank in range(len(dids)):
            this_did = dids[rank]
            doc_input_ids = d_dict[did2idx[this_did]].tolist()
            doc_input_ids = doc_input_ids[:max_passage_length]
            input_ids = [cls_id] + query_input_ids + [sep_id] + doc_input_ids + [sep_id]
            token_type_ids = [0] * (2 + len(query_input_ids)) + [1] * (1 + len(doc_input_ids))
            cat_len = min(len(input_ids), config.max_len)

            new_q_ids.append(this_qid)
            new_d_ids.append(this_did)
            tokens_np[idx * 100 + rank, :cat_len] = np.array(input_ids)
            token_type_np[idx * 100 + rank, :cat_len] = np.array(token_type_ids)

    attention_mask = np.int64(tokens_np > 0).tolist()
    tokens = tokens_np.tolist()  # q_num x 512
    token_type = token_type_np.tolist()  # q_num x 512

    ds = TestQDDataset(
        q_ids=new_q_ids,
        d_ids=new_d_ids,
        token_ids=tokens,
        token_type_ids=token_type,
        attention_mask=attention_mask,
        mode='test'
    )

    return DataLoader(
        ds,
        batch_size=100,  # 100 docs per q
        num_workers=0,
        shuffle=False,
    )

