'''
@ref: Axiomatically Regularized Pre-training for Ad hoc Search
@author: Jia Chen, Yiqun Liu, Yan Fang, Jiaxin Mao, Hui Fang, Shenghao Yang, Xiaohui Xie, Min Zhang, Shaoping Ma.
'''
# encoding: utf-8
import random
import numpy as np
import collections
import pandas as pd
from tqdm import tqdm
import xgboost as xgb
from scipy.special import *
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import BertTokenizer


MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])
# masked_lm_prob=0.15, max_predictions_per_seq=60, True, bert_vocab_list (id)
def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list, id2token):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    cand_indices = []

    START_DOC = False
    for (i, token) in enumerate(tokens):  # token_ids
        if token == 102:  # SEP
            START_DOC = True
            continue
        if token == 101:  # CLS
            continue
        if not START_DOC:
            continue

        if (whole_word_mask and len(cand_indices) >= 1 and id2token[token].startswith("##")):
            cand_indices[-1].append(i)
        else:
            cand_indices.append([i])

    num_to_mask = min(max_predictions_per_seq, max(1, int(round(len(cand_indices) * masked_lm_prob))))
    random.shuffle(cand_indices)
    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indices:
        if len(masked_lms) >= num_to_mask:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_mask:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)
            # 80% of the time, replace with [MASK]
            if random.random() < 0.8:
                masked_token = 103
            else:
                # 10% of the time, keep original
                if random.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = random.choice(vocab_list)
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
            tokens[index] = masked_token

    assert len(masked_lms) <= num_to_mask
    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    mask_indices = [p.index for p in masked_lms]
    masked_token_labels = [p.label for p in masked_lms]

    return tokens, masked_token_labels, mask_indices


def create_masked_lm_predictions_ict(tokens, masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list, id2token):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    cand_indices = []
    for (i, token) in enumerate(tokens):  # token_ids
        if (whole_word_mask and len(cand_indices) >= 1 and id2token[token].startswith("##")):  # startswith ##
            cand_indices[-1].append(i)
        else:
            cand_indices.append([i])

    num_to_mask = min(max_predictions_per_seq, max(1, int(round(len(cand_indices) * masked_lm_prob))))
    random.shuffle(cand_indices)
    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indices:
        if len(masked_lms) >= num_to_mask:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_mask:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)
            # 80% of the time, replace with [MASK]
            if random.random() < 0.8:
                masked_token = 103
            else:
                # 10% of the time, keep original
                if random.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = random.choice(vocab_list)
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
            tokens[index] = masked_token

    assert len(masked_lms) <= num_to_mask
    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    mask_indices = [p.index for p in masked_lms]
    masked_token_labels = [p.label for p in masked_lms]

    return tokens, masked_token_labels, mask_indices


class TrainICTPairwise(Dataset):
    def __init__(self, dids, d_dict, did2idx, config):
        self.dids = dids
        self.d_dict = d_dict
        self.did2idx = did2idx
        self.config = config

        self.tokenizer = BertTokenizer.from_pretrained(config.PRE_TRAINED_MODEL_NAME)
        self.vocab_list = list(self.tokenizer.vocab[key] for key in self.tokenizer.vocab)
        self.id2token = {self.tokenizer.vocab[key]: key for key in self.tokenizer.vocab}
        self.sep_token_id = self.tokenizer.vocab["."]
        self.cls_id = 101

    def __len__(self):
        return len(self.dids)

    def __getitem__(self, item):
        this_did = self.dids[item]

        doc_ids = self.d_dict[self.did2idx[this_did]].tolist()
        sep_pos = [-1] + [i for i, id in enumerate(doc_ids) if id == self.sep_token_id] + [len(doc_ids) - 1]
        sentences = [doc_ids[sep_pos[i] + 1: sep_pos[i + 1] + 1] for i in range(len(sep_pos) - 1)]
        removes = [random.random() < 0.9 for _ in range(len(sentences))]

        s_ids, c_ids = [], []
        b_token_ids, b_attention_mask, b_masked_lm_ids = np.array([[]]), np.array([[]]), np.array([[]])

        for idx, remove in enumerate(removes):
            if remove == 1:
                sentence = [self.cls_id] + sentences[idx]
                context = sentences[: idx] + sentences[idx + 1:]
                context = [self.cls_id] + [w for s in context for w in s]

                sentence = sentence[: self.config.max_len]
                context = context[: self.config.max_len]
                s_ids.append(sentence)
                c_ids.append(context)

                s_input_ids = np.zeros(self.config.max_len, dtype=np.int)
                c_input_ids = np.zeros(self.config.max_len, dtype=np.int)
                s_input_ids[: len(sentence)] = sentence
                c_input_ids[: len(context)] = context

                s_attention_mask = np.int64(s_input_ids > 0)
                c_attention_mask = np.int64(c_input_ids > 0)
                attention_mask = np.stack((s_attention_mask, c_attention_mask))

                s_input_ids, s_masked_lm_ids, s_masked_lm_positions = create_masked_lm_predictions_ict(
                    s_input_ids,
                    masked_lm_prob=self.config.masked_lm_prob,
                    max_predictions_per_seq=self.config.max_predictions_per_seq,
                    whole_word_mask=True,
                    vocab_list=self.vocab_list,
                    id2token=self.id2token)
                c_input_ids, c_masked_lm_ids, c_masked_lm_positions = create_masked_lm_predictions_ict(
                    c_input_ids,
                    masked_lm_prob=self.config.masked_lm_prob,
                    max_predictions_per_seq=self.config.max_predictions_per_seq,
                    whole_word_mask=True,
                    vocab_list=self.vocab_list,
                    id2token=self.id2token)
                s_lm_label_array = np.full(self.config.max_len, dtype=np.int, fill_value=-1)
                c_lm_label_array = np.full(self.config.max_len, dtype=np.int, fill_value=-1)
                s_lm_label_array[s_masked_lm_positions] = s_masked_lm_ids
                c_lm_label_array[c_masked_lm_positions] = c_masked_lm_ids
                masked_lm_ids = np.stack((s_lm_label_array, c_lm_label_array))

                token_ids = np.stack((s_input_ids.flatten(), c_input_ids.flatten()))
                b_token_ids = token_ids if len(b_token_ids) == 1 else np.concatenate((b_token_ids, token_ids), axis=0)
                b_attention_mask = attention_mask if len(b_attention_mask) == 1 else np.concatenate((b_attention_mask, attention_mask), axis=0)
                b_masked_lm_ids = masked_lm_ids if len(b_masked_lm_ids) == 1 else np.concatenate((b_masked_lm_ids, masked_lm_ids), axis=0)

        # clip
        b_token_ids = b_token_ids[: self.config.batch_size, :]
        b_attention_mask = b_attention_mask[: self.config.batch_size, :]
        b_masked_lm_ids = b_masked_lm_ids[: self.config.batch_size, :]  # no greater than max batch size

        return {
            'token_ids': b_token_ids,  # b x 2
            'attention_mask': b_attention_mask,
            'masked_lm_ids': b_masked_lm_ids,
        }


def get_ict_loader(d_dict, did2idx, config):

    dids = list(did2idx.keys())
    print('Loading tokens...')
    ds = TrainICTPairwise(
        dids=dids,
        d_dict=d_dict,
        did2idx=did2idx,
        config=config
    )
    batch_size = 1
    if config.distributed_train:
        sampler = DistributedSampler(ds, num_replicas=config.world_size, rank=config.local_rank)
        return DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=0,
            sampler=sampler
        )
    else:
        return DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=0,
            shuffle=True,
        )


class TrainQDDatasetPairwise(Dataset):
    def __init__(self, q_ids, d_ids, d_dict, did2idx, config, gen_qs, gen_qid2id):
        self.q_ids = q_ids
        self.d_ids = d_ids
        self.d_dict = d_dict
        self.did2idx = did2idx
        self.gen_qs = gen_qs
        self.gen_qid2id = gen_qid2id
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(self.config.PRE_TRAINED_MODEL_NAME)
        self.vocab_list = list(self.tokenizer.vocab[key] for key in self.tokenizer.vocab)
        self.id2token = {self.tokenizer.vocab[key]: key for key in self.tokenizer.vocab}

    def __len__(self):
        return len(self.q_ids)

    def __getitem__(self, item):
        cls_id, sep_id = 101, 102
        q_id = self.q_ids[item]
        d_id = self.d_ids[item]

        pos_q_id, neg_q_id = q_id[0], q_id[1]
        did = d_id[0]

        pos_query_input_ids = self.gen_qs[self.gen_qid2id[pos_q_id]].tolist()
        neg_query_input_ids = self.gen_qs[self.gen_qid2id[neg_q_id]].tolist()

        doc_input_ids = self.d_dict[self.did2idx[did]].tolist()
        pos_query_input_ids = pos_query_input_ids[: self.config.max_q_len]
        neg_query_input_ids = neg_query_input_ids[: self.config.max_q_len]

        pos_max_passage_length = self.config.max_len - 3 - len(pos_query_input_ids)
        neg_max_passage_length = self.config.max_len - 3 - len(neg_query_input_ids)

        pos_doc_input_ids = doc_input_ids[:pos_max_passage_length]
        neg_doc_input_ids = doc_input_ids[:neg_max_passage_length]

        pos_input_ids = [cls_id] + pos_query_input_ids + [sep_id] + pos_doc_input_ids + [sep_id]
        neg_input_ids = [cls_id] + neg_query_input_ids + [sep_id] + neg_doc_input_ids + [sep_id]

        pos_token_type_ids = [0] * (2 + len(pos_query_input_ids)) + [1] * (1 + len(pos_doc_input_ids))
        neg_token_type_ids = [0] * (2 + len(neg_query_input_ids)) + [1] * (1 + len(neg_doc_input_ids))

        pos_token_ids = np.array(pos_input_ids)
        neg_token_ids = np.array(neg_input_ids)

        pos_attention_mask = np.int64(pos_token_ids > 0)
        neg_attention_mask = np.int64(neg_token_ids > 0)
        attention_mask = np.stack((pos_attention_mask, neg_attention_mask))

        pos_token_type_ids = np.array(pos_token_type_ids)
        neg_token_type_ids = np.array(neg_token_type_ids)
        token_type_ids = np.stack((pos_token_type_ids, neg_token_type_ids))

        pos_token_ids, pos_masked_lm_ids, pos_masked_lm_positions = create_masked_lm_predictions(
            pos_token_ids,
            masked_lm_prob=self.config.masked_lm_prob,
            max_predictions_per_seq=self.config.max_predictions_per_seq,
            whole_word_mask=True,
            vocab_list=self.vocab_list,
            id2token=self.id2token)
        neg_token_ids, neg_masked_lm_ids, neg_masked_lm_positions = create_masked_lm_predictions(
            neg_token_ids,
            masked_lm_prob=self.config.masked_lm_prob,
            max_predictions_per_seq=self.config.max_predictions_per_seq,
            whole_word_mask=True,
            vocab_list=self.vocab_list,
            id2token=self.id2token)
        token_ids = np.stack((pos_token_ids.flatten(), neg_token_ids.flatten()))

        pos_lm_label_array = np.full(self.config.max_len, dtype=np.int, fill_value=-1)
        neg_lm_label_array = np.full(self.config.max_len, dtype=np.int, fill_value=-1)
        pos_lm_label_array[pos_masked_lm_positions] = pos_masked_lm_ids
        neg_lm_label_array[neg_masked_lm_positions] = neg_masked_lm_ids

        masked_lm_ids = np.stack((pos_lm_label_array, neg_lm_label_array))

        return {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'masked_lm_ids': masked_lm_ids,
        }


# [CLS] q [SEP] d [SEP]
def get_train_qd_loader(d_dict, did2idx, config, doc2query=None, gen_qs=None, gen_qid2id=None, axiom_feature=None):
    q_max_len, max_len, batch_size = config.max_q_len, config.max_len, config.batch_size

    new_q_ids, new_d_ids = [], []
    doc_num = len(did2idx)
    dids = list(did2idx.keys())

    # loading xgboost model
    model = xgb.XGBRFClassifier()
    model.load_model(config.clf_model)

    all_case = []
    for idx in tqdm(range(doc_num), desc=f"Sampling Pre-train Query Pairs progress"):
        this_did = dids[idx]
        if this_did not in doc2query:
            continue

        qids = [[qid] for qid in doc2query[this_did]]
        q_num = len(qids)
        for i in range(q_num):
            q_id = qids[i][0]
            idx = gen_qid2id[q_id]
            for k in range(len(axiom_feature)):
                this_feature_name, this_feature = axiom_feature[k][0], axiom_feature[k][1]
                if this_feature_name == 'RANK':
                    score = this_feature[idx][0] if this_feature[idx][0] != 0 else 1e12
                else:
                    score = this_feature[idx][0]
                score = this_feature[idx][0] if this_feature_name not in ['PROX-1', 'PROX-2', 'RANK'] else (1 / (score + 1e-12))
                qids[i].append(score)

        all_pairs = []
        for i in range(q_num):
            for j in range(i+1, q_num):
                q1, q2 = qids[i], qids[j]
                all_pairs.append([q1, q2])

        k = min(2, len(all_pairs))
        sampled_pairs = random.sample(all_pairs, k=k)

        for pair in sampled_pairs:
            qid1, qid2 = pair[0][0], pair[1][0]
            case = []
            for i in range(len(axiom_feature)):
                axiom_1 = pair[0][i + 1]
                axiom_2 = pair[1][i + 1]
                if axiom_1 > axiom_2:
                    case.append(1)
                elif axiom_1 == axiom_2:
                    case.append(0)
                else:
                    case.append(-1)
            all_case.append(case)
            new_q_ids.append([qid1, qid2])
            new_d_ids.append([this_did])

    all_case = pd.DataFrame(np.array(all_case))
    all_case.columns = ['PROX-1', 'PROX-2', 'REP-QL', 'REP-TFIDF', 'REG', 'STM-1', 'STM-2', 'STM-3', 'RANK']
    pred_prob = model.predict(all_case)
    for idx, pred in enumerate(pred_prob):
        result = 1 if pred > 0.5 else 0
        if result == 0:  # swap
            qid1 = new_q_ids[idx][0]
            qid2 = new_q_ids[idx][1]
            new_q_ids[idx][0] = qid2
            new_q_ids[idx][1] = qid1

    print('Loading tokens...')
    ds = TrainQDDatasetPairwise(
        q_ids=new_q_ids,
        d_ids=new_d_ids,
        d_dict=d_dict,
        did2idx=did2idx,
        config=config,
        gen_qs=gen_qs,
        gen_qid2id=gen_qid2id,
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
        return DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=0,
            shuffle=True,
        )


