'''
@ref: Axiomatically Regularized Pre-training for Ad hoc Search
@author: Jia Chen, Yiqun Liu, Yan Fang, Jiaxin Mao, Hui Fang, Shenghao Yang, Xiaohui Xie, Min Zhang, Shaoping Ma.
'''
# encoding: utf-8
import os
import sys
sys.path.insert(0, '../')

from tqdm import tqdm
import json
import torch
import numpy as np
import pandas as pd
from datetime import timedelta


from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import PretrainedConfig, BertConfig
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
from model.modeling import ARES, ICT

from dataloader import get_train_qd_loader, get_test_qd_loader
from config import get_config
from ms_marco_eval import compute_metrics_from_files
import warnings

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True


def train_epoch(model, scaler, qd_loader, optimizer, scheduler, device, config):
    model.train()
    losses = []

    num_instances = len(qd_loader)
    model_name = config.model_name
    for step, batch_data in enumerate(tqdm(qd_loader, desc=f"Fine-tuning {model_name} progress", total=num_instances)):
        input_ids, attention_mask, token_type_ids = batch_data["token_ids"], batch_data["attention_mask"], batch_data["token_type_ids"]
        this_batch_size = input_ids.size()[0]

        # b/2 x 2 x 512 ==> b x 512
        input_ids = input_ids.reshape(this_batch_size * 2, -1)
        attention_mask = attention_mask.reshape(this_batch_size * 2, -1)
        token_type_ids = token_type_ids.reshape(this_batch_size * 2, -1)

        input_ids = input_ids.to(device)  # bs x 512
        attention_mask = attention_mask.to(device)  # bs x 512
        token_type_ids = token_type_ids.to(device)

        with autocast():
            output = model(
                input_ids=input_ids,
                config=config,
                input_mask=attention_mask,
                token_type_ids=token_type_ids,
            )  # bs x 1

            softmax = nn.Softmax(dim=1)
            marginloss = nn.MarginRankingLoss(margin=1.0, reduction='mean')
            batch_size = output.size(0)
            logits = output.reshape(batch_size // 2, 2)
            logits = softmax(logits)
            pos_logits = logits[:, 0]
            neg_logits = logits[:, 1]
            rop_label = torch.ones_like(pos_logits)
            loss = marginloss(pos_logits, neg_logits, rop_label)

        loss = loss / config.gradient_accumulation_steps
        losses.append(loss.item())
        scaler.scale(loss).backward()

        # gradient accumulation
        if (step + 1) % config.gradient_accumulation_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.clip)
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            optimizer.zero_grad()

        if step % int(config.print_every) == 0:
            print(f"\n[Train] Loss at step {step} = {loss.item()}, lr = {optimizer.state_dict()['param_groups'][0]['lr']}")
    return np.mean(losses)


def eval_model(model, qd_loader, device, config):
    model.eval()
    df_rank = pd.DataFrame(columns=['q_id', 'd_id', 'rank', 'score'])
    q_id_list, d_id_list, rank, score = [], [], [], []

    num_instances = len(qd_loader)
    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(qd_loader, desc=f"Evaluating progress", total=num_instances)):
            input_ids, attention_mask, token_type_ids = batch_data["token_ids"], batch_data["attention_mask"], \
                                                               batch_data["token_type_ids"]

            input_ids = input_ids.to(device)  # bs x 512
            attention_mask = attention_mask.to(device)  # bs x 512
            token_type_ids = token_type_ids.to(device)

            output = model(
                input_ids=input_ids,
                config=config,
                input_mask=attention_mask,
                token_type_ids=token_type_ids,
            )  # 100 x 1

            output = output.squeeze()
            q_ids = batch_data["q_id"]
            d_ids = batch_data["d_id"]
            scores = output.cpu().tolist()
            tuples = list(zip(q_ids, d_ids, scores))
            sorted_tuples = sorted(tuples, key=lambda x: x[2], reverse=True)  # 看一下top100的分数分布
            for idx, this_tuple in enumerate(sorted_tuples):
                q_id_list.append(this_tuple[0])
                d_id_list.append(this_tuple[1])
                rank.append(idx + 1)
                score.append(this_tuple[2])

        df_rank['q_id'] = q_id_list
        df_rank['d_id'] = d_id_list
        df_rank['rank'] = rank
        df_rank['score'] = score
    return df_rank


if __name__ == '__main__':
    config = get_config()

    # automatically create save dirs
    save_dir = f"{config.PRE_TRAINED_MODEL_NAME}/ckpt"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_model_path = f"{config.PRE_TRAINED_MODEL_NAME}/ckpt/model_state"

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.set_device(config.local_rank)
        print('GPU is ON!')
        device = torch.device(f'cuda:{config.local_rank}')
    else:
        device = torch.device("cpu")

    # distributed training
    if config.distributed_train and not config.test:
        torch.distributed.init_process_group(backend="nccl", timeout=timedelta(180000000))
        local_rank = config.local_rank
        if local_rank != -1:
            print("Using Distributed")

    # Train Data Loader
    df_train_qds = pd.read_csv(config.train_qd_dir, sep=' ', header=None)
    if config.local_rank == 0:
        df_test_qds = pd.read_csv(config.test_qd_dir, sep=' ', header=None)
        df_dl2019_qds = pd.read_csv(config.dl2019_qd_dir, sep=' ', header=None)

    best_nDCG_dl2019, best_MRR_test = 0., 0.
    train_top100 = pd.read_csv(config.train100_dir, sep='\t', header=None)
    if config.local_rank == 0:
        test_top100 = pd.read_csv(config.test100_dir, sep='\t', header=None)
        dl2019_top100 = pd.read_csv(config.dl100_dir, sep='\t', header=None)

    # json files
    train_qs, test_qs, dl2019_qs, doc2query = {}, {}, {}, {}
    with open(config.train_qs_dir) as f_train_qs:
        for line in f_train_qs:
            es = json.loads(line)
            qid, ids = es["id"], es["ids"]
            if qid not in train_qs:
                train_qs[qid] = ids

    if config.local_rank == 0:
        with open(config.test_qs_dir) as f_test_qs:
            for line in f_test_qs:
                es = json.loads(line)
                qid, ids = es["id"], es["ids"]
                if qid not in test_qs:
                    test_qs[qid] = ids
        with open(config.dl2019_qs_dir) as f_dl2019_qs:
            for line in f_dl2019_qs:
                es = json.loads(line)
                qid, ids = es["id"], es["ids"]
                if qid not in dl2019_qs:
                    dl2019_qs[qid] = ids

    with open(config.docid2id_dir) as f_docid2id:
        docid2id = json.load(f_docid2id)
    print("Load dicts done!")

    collection_size = len(docid2id)
    doc_tokens = np.memmap(config.memmap_doc_dir, dtype='int32', shape=(collection_size, 512))

    cfg = PretrainedConfig.get_config_dict(config.PRE_TRAINED_MODEL_NAME)[0]
    if not config.gradient_checkpointing:
        del cfg["gradient_checkpointing"]
    cfg = BertConfig.from_dict(cfg)

    if not config.load_ckpt:  # train
        if config.model_type == 'ICT':
            model = ICT.from_pretrained(config.PRE_TRAINED_MODEL_NAME, config=cfg)
        else:
            model = ARES.from_pretrained(config.PRE_TRAINED_MODEL_NAME, config=cfg)
    else:  # test
        if config.model_type == 'ARES':
            model = ARES(config=cfg)
        elif config.model_type == 'PROP':
            model = PROP(config=cfg)
        else:
            model = ICT(config=cfg)
        model.load_state_dict({k.replace("module.", ""): v for k, v in torch.load(f"{config.PRE_TRAINED_MODEL_NAME}/ckpt/{config.model_path}",
                                                                                  map_location={'cuda:0': f'cuda:{config.local_rank}'}).items()})

    model = model.to(device)
    print("Loading model...")
    model = model.cuda()

    scaler = GradScaler(enabled=True)

    if config.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    elif config.optim == 'amsgrad':
        optimizer = torch.optim.Amsgrad(model.parameters(), lr=config.lr)
    elif config.optim == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=config.lr)
    else:  # adamw, weight decay not depend on the lr
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr, eps=config.adam_epsilon)

    if not config.test:   # train
        if config.distributed_train:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True, broadcast_buffers=False)
            config.warm_up = config.warm_up / config.gpu_num

        if config.local_rank == 0:
            print("\n========== Loading dev data ==========")
            test_qd_loader = get_test_qd_loader(test_top100, test_qs, doc_tokens, docid2id, config)
            print(f"test_q: {len(test_qs)}, test_q_batchs:{len(test_qd_loader)}")

            print("\n========== Loading DL 2019 data ==========")
            dl2019_qd_loader = get_test_qd_loader(dl2019_top100, dl2019_qs, doc_tokens, docid2id, config)
            print(f"dl2019_q: {len(dl2019_qs)}, dl2019_q_batchs:{len(dl2019_qd_loader)}")

        for epoch in range(config.epochs):
            print(f'Epoch {epoch + 1}/{config.epochs}')
            print('-' * 10)

            print("========== Loading training data ==========")
            train_qd_loader = get_train_qd_loader(df_train_qds, train_top100, train_qs, doc_tokens, docid2id, config, mode='train')  # b_sz * data samples
            print(f"train_qd_pairs: {len(df_train_qds)}, train_batchs:{len(train_qd_loader)}, batch_size: {config.batch_size}")

            total_steps = len(train_qd_loader)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(total_steps * config.warm_up),
                num_training_steps=total_steps
            )

            train_loss = train_epoch(
                model,
                scaler,
                train_qd_loader,
                optimizer,
                scheduler,
                device,
                config,
            )
            scheduler.step()
            print(f'Train loss {train_loss}')

            if config.local_rank == 0:
                qd_rank = eval_model(
                    model,
                    dl2019_qd_loader,
                    device,
                    config,
                )
                df_rank = pd.DataFrame(columns=['q_id', 'Q0', 'd_id', 'rank', 'score', 'standard'])
                df_rank['q_id'] = qd_rank['q_id']
                df_rank['Q0'] = ['Q0'] * len(qd_rank['q_id'])
                df_rank['d_id'] = qd_rank['d_id']
                df_rank['rank'] = qd_rank['rank']
                df_rank['score'] = qd_rank['score']
                df_rank['standard'] = ['STANDARD'] * len(qd_rank['q_id'])
                df_rank.to_csv(f"{save_dir}/dl2019_qd_rank.tsv", sep=' ', index=False, header=False)  # !
                result_lines = os.popen(f'trec_eval -m ndcg_cut.10,100 {config.dl2019_qd_dir} {save_dir}/dl2019_qd_rank.tsv').read().strip().split("\n")
                ndcg_10, ndcg_100 = float(result_lines[0].strip().split()[-1]), float(
                    result_lines[1].strip().split()[-1])
                metrics = {'nDCG @10': ndcg_10, 'nDCG @100': ndcg_100, 'QueriesRanked': len(set(qd_rank['q_id']))}

                print('\n#############################')
                print('<--------- DL 2019 --------->')
                for metric in sorted(metrics):
                    print('{}: {}'.format(metric, metrics[metric]))
                print('#############################\n')
                nDCG_dl2019 = round(metrics['nDCG @10'], 4)
                nDCG_dl2019_100 = round(metrics['nDCG @100'], 4)
                if nDCG_dl2019 > best_nDCG_dl2019:
                    best_nDCG_dl2019 = nDCG_dl2019
                    qd_rank.to_csv(f"{save_dir}/best_{config.model_type}_dl2019_qd_rank.tsv", sep='\t', index=False,
                                   header=False)

                # test msmarco dev
                qd_rank = eval_model(
                    model,
                    test_qd_loader,
                    device,
                    config,
                )
                qd_rank.to_csv(f"{save_dir}/test_qd_rank.tsv", sep='\t', index=False, header=False)
                metrics = compute_metrics_from_files(config.test_qd_dir, f"{save_dir}/test_qd_rank.tsv")
                print('\n#####################')
                print('<----- MS Dev ----->')
                for metric in sorted(metrics):
                    print('{}: {}'.format(metric, metrics[metric]))
                print('#####################\n')
                MRR_test = round(metrics['MRR @10'], 4)
                MRR_test_100 = round(metrics['MRR @100'], 4)
                if MRR_test > best_MRR_test:
                    best_MRR_test = MRR_test
                    qd_rank.to_csv(f"{save_dir}/best_{config.model_type}_test_qd_rank.tsv", sep='\t', index=False, header=False)

                print('[SAVE] Saving model ... ')
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                torch.save(model_to_save.state_dict(),f"{save_dir}/{config.model_type}_{MRR_test}_{MRR_test_100}_e{epoch + 1}")

    else:  # test
        print("\n========== Loading dev data ==========")
        test_qd_loader = get_test_qd_loader(test_top100, test_qs, doc_tokens, docid2id, config)
        print(f"test_q: {len(test_qs)}, test_q_batchs:{len(test_qd_loader)}")

        print("\n========== Loading DL 2019 data ==========")
        dl2019_qd_loader = get_test_qd_loader(dl2019_top100, dl2019_qs, doc_tokens, docid2id, config)
        print(f"dl2019_q: {len(dl2019_qs)}, dl2019_q_batchs:{len(dl2019_qd_loader)}")

        qd_rank = eval_model(
            model,
            dl2019_qd_loader,
            device,
            config,
        )
        df_rank = pd.DataFrame(columns=['q_id', 'Q0', 'd_id', 'rank', 'score', 'standard'])
        df_rank['q_id'] = qd_rank['q_id']
        df_rank['Q0'] = ['Q0'] * len(qd_rank['q_id'])
        df_rank['d_id'] = qd_rank['d_id']
        df_rank['rank'] = qd_rank['rank']
        df_rank['score'] = qd_rank['score']
        df_rank['standard'] = ['STANDARD'] * len(qd_rank['q_id'])
        df_rank.to_csv(f"{save_dir}/dl2019_qd_rank_as100.tsv", sep=' ', index=False, header=False)
        result_lines = os.popen(f'trec_eval -m ndcg_cut.10,100 {config.dl2019_qd_dir} {save_dir}/dl2019_qd_rank_as100.tsv').read().strip().split("\n")
        ndcg_10, ndcg_100 = float(result_lines[0].strip().split()[-1]), float(result_lines[1].strip().split()[-1])
        metrics = {'nDCG @10': ndcg_10, 'nDCG @100': ndcg_100, 'QueriesRanked': len(set(qd_rank['q_id']))}
        print('\n#############################')
        print('<--------- DL 2019 --------->')
        for metric in sorted(metrics):
            print('{}: {}'.format(metric, metrics[metric]))
        print('#############################\n')

        # test msmarco dev
        qd_rank = eval_model(
            model,
            test_qd_loader,
            device,
            config,
        )
        qd_rank.to_csv(f"{save_dir}/test_qd_rank_as100.tsv", sep='\t', index=False, header=False)
        metrics = compute_metrics_from_files(config.test_qd_dir, f"{save_dir}/test_qd_rank_as100.tsv")
        print('\n#####################')
        print('<----- MS Dev ----->')
        for metric in sorted(metrics):
            print('{}: {}'.format(metric, metrics[metric]))
        print('#####################\n')
