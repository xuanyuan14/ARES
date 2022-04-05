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
from datetime import timedelta, datetime
from model.modeling import ARES, ICT

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import PretrainedConfig, BertConfig
from torch import nn
from torch.cuda.amp import autocast, GradScaler

from dataloader import get_train_qd_loader, get_ict_loader
from config import get_config
import warnings

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True


def train_epoch(model, scaler, qd_loader, optimizer, scheduler, device, config):
    model.train()
    losses = []

    num_instances = len(qd_loader)
    for step, batch_data in enumerate(tqdm(qd_loader, desc=f"Pretraining {config.model_type} progress", total=num_instances)):
        input_ids, attention_mask, masked_lm_ids = batch_data["token_ids"], batch_data["attention_mask"], batch_data["masked_lm_ids"]
        if config.model_type == 'ICT':
            token_type_ids = None
            input_ids, attention_mask, masked_lm_ids = input_ids.squeeze(), attention_mask.squeeze(), masked_lm_ids.squeeze()
            this_batch_size = input_ids.size()[0]
            if this_batch_size < 2:
                continue
        else:
            this_batch_size = input_ids.size()[0]
            token_type_ids = batch_data["token_type_ids"]

            input_ids = input_ids.reshape(this_batch_size * 2, -1)
            attention_mask = attention_mask.reshape(this_batch_size * 2, -1)
            masked_lm_ids = masked_lm_ids.reshape(this_batch_size * 2, -1)
            token_type_ids = token_type_ids.reshape(this_batch_size * 2, -1) if token_type_ids is not None else token_type_ids

        input_ids = input_ids.to(device)  # bs x 512
        attention_mask = attention_mask.to(device)  # bs x 512
        masked_lm_ids = masked_lm_ids.to(device)

        token_type_ids = token_type_ids.to(device) if token_type_ids is not None else token_type_ids

        with autocast():
            loss = model(
                input_ids=input_ids,
                config=config,
                input_mask=attention_mask,
                token_type_ids=token_type_ids,
                masked_lm_labels=masked_lm_ids,
                device=device
            )

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

        if step % 5000 == 0 and config.local_rank == 0:
            print('[SAVE] Saving model ... ')
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            this_loss = round(float(np.mean(losses)), 4)
            torch.save(model_to_save.state_dict(), f"{save_dir}/{config.model_name}_{this_loss}_step{step}")
    return np.mean(losses)


if __name__ == '__main__':

    # get configs
    config = get_config()

    # set save dir
    today = datetime.today().strftime('%Y-%m-%d')
    save_dir = f"{config.PRE_TRAINED_MODEL_NAME}/ckpt/{today}"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    config.local_rank = config.local_rank
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.set_device(config.local_rank)
        print('GPU is ON!')
        device = torch.device(f'cuda:{config.local_rank}')
    else:
        device = torch.device("cpu")

    # set max timeout=50 hours
    if config.distributed_train:
        torch.distributed.init_process_group(backend="nccl", timeout=timedelta(180000000), rank=config.local_rank, world_size=config.world_size)
        local_rank = config.local_rank
        if local_rank != -1:
            print("Using Distributed")

    # json files
    doc2query = {}
    with open(config.doc2query_dir) as f_doc2query:
        for line in f_doc2query:
            es = json.loads(line)
            docid = es["docid"]
            queries = es["queries"]
            if docid not in doc2query:
                doc2query[docid] = queries

    with open(config.gen_qid2id_dir) as f_gen_qid2id:
        gen_qid2id = json.load(f_gen_qid2id)

    # save memory
    if config.model_type == 'ARES':
        q_num = len(gen_qid2id)

        axiom_rank = np.memmap(f"{config.axiom_feature_dir}/memmap/rank.memmap", dtype='float', shape=(q_num, 1))
        axiom_list = []
        print(config.axiom)
        if 'PROX' in config.axiom:
            prox_1 = np.memmap(f"{config.axiom_feature_dir}/memmap/prox-1.memmap", dtype='float', shape=(q_num, 1))
            prox_2 = np.memmap(f"{config.axiom_feature_dir}/memmap/prox-2.memmap", dtype='float', shape=(q_num, 1))
            axiom_list.append(['PROX-1', prox_1])
            axiom_list.append(['PROX-2', prox_2])

        if 'REP' in config.axiom:
            rep_ql = np.memmap(f"{config.axiom_feature_dir}/memmap/rep-ql.memmap", dtype='float', shape=(q_num, 1))
            rep_tfidf = np.memmap(f"{config.axiom_feature_dir}/memmap/rep-tfidf.memmap", dtype='float', shape=(q_num, 1))
            axiom_list.append(['REP-QL', rep_ql])
            axiom_list.append(['REP-TFIDF', rep_tfidf])

        if 'REG' in config.axiom:
            reg = np.memmap(f"{config.axiom_feature_dir}/memmap/reg.memmap", dtype='float', shape=(q_num, 1))
            axiom_list.append(['REG', reg])

        if 'STM' in config.axiom:
            stm_1 = np.memmap(f"{config.axiom_feature_dir}/memmap/stm-1.memmap", dtype='float', shape=(q_num, 1))
            stm_2 = np.memmap(f"{config.axiom_feature_dir}/memmap/stm-2.memmap", dtype='float', shape=(q_num, 1))
            stm_3 = np.memmap(f"{config.axiom_feature_dir}/memmap/stm-3.memmap", dtype='float', shape=(q_num, 1))

            axiom_list.append(['STM-1', stm_1])
            axiom_list.append(['STM-2', stm_2])
            axiom_list.append(['STM-3', stm_3])

        axiom_list.append(['RANK', axiom_rank])
        gen_qs_size = len(gen_qid2id)
        gen_qs_tokens = np.memmap(config.gen_qs_memmap_dir, dtype='int32', shape=(gen_qs_size, 15))

    with open(config.docid2id_dir) as f_docid2id:
        docid2id = json.load(f_docid2id)
    collection_size = len(docid2id)
    doc_tokens = np.memmap(config.memmap_doc_dir, dtype='int32', shape=(collection_size, 512))

    print("Load data done!")

    cfg = PretrainedConfig.get_config_dict(config.PRE_TRAINED_MODEL_NAME)[0]
    if not config.gradient_checkpointing:
        del cfg["gradient_checkpointing"]  # gradient checkpointing conflicts with parallel training
        del cfg["parameter_sharing"]
    cfg = BertConfig.from_dict(cfg)

    # train
    if not config.load_ckpt:
        if config.model_type == 'ICT':
            model = ICT.from_pretrained(config.PRE_TRAINED_MODEL_NAME, config=cfg)
        else:
            model = ARES.from_pretrained(config.PRE_TRAINED_MODEL_NAME, config=cfg)
    else:
        if config.model_type == 'ICT':
            model = ICT(config=cfg)
        else:
            model = ARES(config=cfg)
        model.load_state_dict({k.replace("module.", ""): v for k, v in torch.load(f"{config.PRE_TRAINED_MODEL_NAME}/ckpt/{config.model_path}",
                                                                                  map_location={'cuda:0': f'cuda:{config.local_rank}'}).items()})
    model = model.to(device)
    print("Loading model...")
    model = model.cuda()

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

    # train
    if config.distributed_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=True)
        config.warm_up = config.warm_up / config.gpu_num

    for epoch in range(config.epochs):
        print(f'Epoch {epoch + 1}/{config.epochs}')
        print('-' * 10)

        print("========== Loading training data ==========")
        if config.model_type == 'ARES':
            train_qd_loader = get_train_qd_loader(doc_tokens, docid2id, config,
                                                  doc2query=doc2query,
                                                  gen_qs=gen_qs_tokens,
                                                  gen_qid2id=gen_qid2id,
                                                  axiom_feature=axiom_list)  # b_sz * data samples
        else:
            train_qd_loader = get_ict_loader(doc_tokens, docid2id, config)
        print(f"train_batchs:{len(train_qd_loader)}, batch_size: {config.batch_size}")

        scaler = GradScaler(enabled=True)
        total_steps = len(train_qd_loader) * config.epochs

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
            print('[SAVE] Saving model ... ')
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            this_loss = round(float(train_loss), 4)
            torch.save(model_to_save.state_dict(), f"{save_dir}/{config.model_name}_{this_loss}")



