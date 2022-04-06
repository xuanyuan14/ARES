'''
@ref: Axiomatically Regularized Pre-training for Ad hoc Search
@author: Jia Chen, Yiqun Liu, Yan Fang, Jiaxin Mao, Hui Fang, Shenghao Yang, Xiaohui Xie, Min Zhang, Shaoping Ma.
'''
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


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


class VisualTestQDDataset(Dataset):
    def __init__(self, q_ids, d_ids, ranks,token_ids, ref_token_ids, token_type_ids, ref_token_type_ids, attention_mask, mode='test'):
        self.q_ids = q_ids
        self.d_ids = d_ids
        self.ranks = ranks
        self.token_ids = token_ids
        self.ref_token_ids=ref_token_ids
        self.token_type_ids = token_type_ids
        self.ref_token_type_ids = ref_token_type_ids
        self.attention_mask = attention_mask
        self.mode = mode

    def __len__(self):
        return len(self.q_ids)

    def __getitem__(self, item):
        q_id = self.q_ids[item]
        d_id = self.d_ids[item]
        rank = self.ranks[item]
        token_ids = np.array(self.token_ids[item])
        attention_mask = np.array(self.attention_mask[item])
        token_type_ids = np.array(self.token_type_ids[item])
        ref_token_ids=np.array(self.ref_token_ids[item])
        ref_token_type_ids=np.array(self.ref_token_type_ids[item])
        return {
                "q_id": q_id,
                "d_id": d_id,
                "rank": rank,
                'token_ids': token_ids.flatten(),
                'attention_mask': attention_mask.flatten(),
                'token_type_ids': token_type_ids.flatten(),
                'ref_token_ids': ref_token_ids.flatten(),
                'ref_token_type_ids': ref_token_type_ids.flatten()
        }


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


def get_visual_test_qd_loader(top100qd, q_dict, d_dict, did2idx, config):
    cls_id, sep_id, pad_id = 101, 102, 0
    q_ids = top100qd["q_id"].values.tolist()
    d_ids = top100qd["d_id"].values.tolist()
    ranks = top100qd["rank"].values.tolist()
    d_num = config.visual_d_num
    q_num = config.visual_q_num
    qd_dict = {}
    for q_id, d_id, rank in zip(q_ids, d_ids,ranks):
        if q_id not in qd_dict:
            qd_dict[q_id] = []
        qd_dict[q_id].append([d_id,rank])

    qids = list(q_dict.keys())[:q_num]
    tokens_np = np.zeros((q_num * d_num, config.max_len), dtype='int32')  # (q_num * d_num)  x 512
    token_type_np = np.zeros((q_num * d_num, config.max_len), dtype='int32')  # (q_num * d_num)  x 512
    ref_tokens_np = np.zeros((q_num * d_num, config.max_len), dtype='int32')  # (q_num * d_num)  x 512
    ref_token_type_np = np.zeros((q_num * d_num, config.max_len), dtype='int32')  # (q_num * d_num)  x 512

    new_q_ids, new_d_ids, new_ranks = [], [], []
    for idx in tqdm(range(len(qids)), desc=f"Loading test q-d pair progress"):
        this_qid = qids[idx]
        query_input_ids = q_dict[str(this_qid)]
        query_input_ids = query_input_ids[: config.max_q_len]
        max_passage_length = config.max_len - 3 - len(query_input_ids)

        did_ranks = qd_dict[str(this_qid)][:d_num]
        assert len(did_ranks) == d_num
        for rank in range(len(did_ranks)):
            this_did,this_rank = did_ranks[rank]
            doc_input_ids = d_dict[did2idx[this_did]].tolist()
            doc_input_ids = doc_input_ids[:max_passage_length]
            input_ids = [cls_id] + query_input_ids + [sep_id] + doc_input_ids + [sep_id]
            ref_input_ids = [cls_id] + [pad_id] * len(query_input_ids) + [sep_id] + [pad_id] * len(doc_input_ids) + [sep_id]
            token_type_ids = [0] * (2 + len(query_input_ids)) + [1] * (1 + len(doc_input_ids))
            ref_token_type_ids = [0] * len(token_type_ids)
            cat_len = min(len(input_ids), config.max_len)

            new_q_ids.append(this_qid)
            new_d_ids.append(this_did)
            new_ranks.append(this_rank)
            tokens_np[idx * d_num + rank, :cat_len] = np.array(input_ids)
            token_type_np[idx * d_num + rank, :cat_len] = np.array(token_type_ids)
            ref_tokens_np[idx * d_num + rank, :cat_len] = np.array(ref_input_ids)
            ref_token_type_np[idx * d_num + rank, :cat_len] = np.array(ref_token_type_ids)
    attention_mask = np.int64(tokens_np > 0).tolist()
    tokens = tokens_np.tolist()  # q_num x 512
    token_type = token_type_np.tolist()  # q_num x 512
    ref_tokens = ref_tokens_np.tolist()  # q_num x 512
    ref_token_type = ref_token_type_np.tolist()  # q_num x 512

    ds = VisualTestQDDataset(
        q_ids=new_q_ids,
        d_ids=new_d_ids,
        ranks=new_ranks,
        token_ids=tokens,
        ref_token_ids=ref_tokens,
        token_type_ids=token_type,
        ref_token_type_ids=ref_token_type,
        attention_mask=attention_mask,
        mode='test'
    )

    return DataLoader(
        ds,
        batch_size=config.batch_size,  # 100 docs per q
        num_workers=0,
        shuffle=False,
    )
