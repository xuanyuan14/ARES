'''
@ref: Axiomatically Regularized Pre-training for Ad hoc Search
@author: Jia Chen, Yiqun Liu, Yan Fang, Jiaxin Mao, Hui Fang, Shenghao Yang, Xiaohui Xie, Min Zhang, Shaoping Ma.
'''
import os
import random
from tqdm import tqdm
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from model.modeling import ARES
from transformers import PretrainedConfig, BertConfig,BertTokenizer
from dataloader import get_visual_test_qd_loader, get_test_qd_loader
from config import get_config
import warnings
from captum.attr import LayerIntegratedGradients
# from captum.attr import visualization as viz
import visualization as viz
from gensim.models import KeyedVectors
warnings.filterwarnings("ignore")


def eval_model(model,test_qd_loader,device, config):
    model.eval()
    qd_rank = pd.DataFrame(columns=['q_id', 'd_id', 'rank', 'score'])
    q_id_list, d_id_list, rank, score = [], [], [], []
    top5_q_id_list, top5_d_id_list, top5_rank_list,top5_score_list = [],[],[],[]
    num_instances = len(test_qd_loader)
    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(test_qd_loader, desc=f"Evaluating progress", total=num_instances)):
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
            top5_q_id_list.extend(q_ids[:5])
            top5_d_id_list.extend(d_ids[:5])
            top5_score_list.extend(scores[:5])
            tuples = list(zip(q_ids, d_ids, scores))
            sorted_tuples = sorted(tuples, key=lambda x: x[2], reverse=True)
            for idx, this_tuple in enumerate(sorted_tuples):
                q_id_list.append(this_tuple[0])
                d_id_list.append(this_tuple[1])
                rank.append(idx + 1)
                score.append(this_tuple[2])
        qd_rank['q_id'] = q_id_list
        qd_rank['d_id'] = d_id_list
        qd_rank['rank'] = rank
        qd_rank['score'] = score
    df_rank = pd.DataFrame(columns=['q_id', 'Q0', 'd_id', 'rank', 'score', 'standard'])
    df_rank['q_id'] = qd_rank['q_id']
    df_rank['Q0'] = ['Q0'] * len(qd_rank['q_id'])
    df_rank['d_id'] = qd_rank['d_id']
    df_rank['rank'] = qd_rank['rank']
    df_rank['score'] = qd_rank['score']
    df_rank['standard'] = ['STANDARD'] * len(qd_rank['q_id'])
    df_rank.to_csv(f"{config.save_path}/dl2019_qd_rank_{config.model_name}.tsv", sep=' ', index=False, header=False)
    result_lines = os.popen(f'trec_eval -m ndcg_cut.10,100 {config.dl2019_qd_dir} {config.save_path}/dl2019_qd_rank_{config.model_name}.tsv').read().strip().split("\n")
    ndcg_10, ndcg_100 = float(result_lines[0].strip().split()[-1]), float(result_lines[1].strip().split()[-1])
    metrics = {'nDCG @10': ndcg_10, 'nDCG @100': ndcg_100, 'QueriesRanked': len(set(qd_rank['q_id']))}
    print('\n#############################')
    print(config.model_name)
    print('<--------- DL 2019 --------->')
    for metric in sorted(metrics):
        print('{}: {}'.format(metric, metrics[metric]))
    print('#############################\n')
    return df_rank


def visual_model(lig, tokenizer, qd_loader, df_dl2019_qds,device, config):
    score_viz_list = []
    index = 0
    for i, batch_data in enumerate(tqdm(qd_loader, desc=f"IG progress", total=len(qd_loader))):
        q_ids,d_ids,ranks,input_ids,ref_input_ids, attention_mask, token_type_ids,ref_token_type_ids = \
            batch_data["q_id"],batch_data["d_id"],batch_data["rank"],\
            batch_data["token_ids"],batch_data["ref_token_ids"], batch_data["attention_mask"],batch_data["token_type_ids"],batch_data["ref_token_type_ids"]
        # print(q_ids,d_ids)
        input_ids = input_ids.to(device)  # bs x 512
        ref_input_ids = ref_input_ids.to(device)  # bs x 512
        attention_mask = attention_mask.to(device)  # bs x 512
        token_type_ids = token_type_ids.to(device)
        ref_token_type_ids = ref_token_type_ids.to(device)  
        attributions, deltas = lig.attribute(
            inputs=(input_ids, token_type_ids),
            baselines=(ref_input_ids,ref_token_type_ids),
            return_convergence_delta=True,
            additional_forward_args=(attention_mask),
            internal_batch_size=5
        )
        for j,(attribution,delta) in enumerate(zip(attributions, deltas)):  # for 512*768 in bs*512*768
            attribution_sum = attribution.sum(dim=-1).squeeze(0)  # 512
            tokens = [token.replace("Ġ", "") for token in tokenizer.convert_ids_to_tokens(input_ids[j])]
            sep_index = tokens.index('[SEP]')
            query_tokens = tokens[:sep_index]
            doc_tokens = tokens[sep_index:sep_index+250]
            tokens = query_tokens+doc_tokens
            query_attribution_sum = attribution_sum[:sep_index] / torch.norm(attribution_sum)
            doc_attribution_sum = attribution_sum[sep_index:sep_index+250] / torch.norm(attribution_sum[sep_index:sep_index+250])
            attribution_sum = torch.cat((query_attribution_sum,doc_attribution_sum),axis=-1)
            v_q_id,v_d_id,rank = q_ids[j],d_ids[j],ranks[j]
            try:
                level = df_dl2019_qds.set_index(0).loc[int(v_q_id)].set_index(2).loc[str(v_d_id)][3]
            except:
                level = -1
            score_viz = viz.VisualizationDataRecord(
                attribution_sum,
                level,
                rank,
                v_q_id,
                v_d_id,
                tokens,
                delta,
            )
            score_viz_list.append(score_viz)
            # index += 1
    html = viz.visualize_text(score_viz_list)
    html_filepath = f"output_{config.model_name}.html"
    with open(html_filepath, "w") as html_file:
        html_file.write(html.data)


if __name__ == '__main__':
    config = get_config()
    random.seed(config.seed)    
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    config.local_rank = config.local_rank
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.set_device(config.local_rank)
        print('GPU is ON!')
        device = torch.device('cuda')
    else:
        device = torch.device("cpu")
    df_dl2019_qds = pd.read_csv(config.dl2019_qd_dir, sep=' ', header=None)
    dl2019_top100 = pd.read_csv(config.dl100_dir, sep='\t', header=None)
    dl2019_qs = {}
    with open(config.dl2019_qs_dir) as f_qs:
        for line in f_qs:
            es = json.loads(line)
            qid, ids = es["id"], es["ids"]
            if qid not in dl2019_qs:
                dl2019_qs[qid] = ids
    with open(config.docid2id_dir) as f_docid2id:
        docid2id = json.load(f_docid2id)
    collection_size = len(docid2id)
    doc_tokens = np.memmap(config.memmap_doc_dir, dtype='int32', shape=(collection_size, 512))
    print("\n========== Loading DL 2019 data ==========")
    dl2019_qd_loader = get_test_qd_loader(dl2019_top100, dl2019_qs, doc_tokens, docid2id, config)
    print(f"dl2019_q: {len(dl2019_qs)}, dl2019_q_batchs:{len(dl2019_qd_loader)}")

    print("Loading model...")
    if 'BERT' in config.model_name:
        model = ARES.from_pretrained(config.PRE_TRAINED_MODEL_NAME)
        tokenizer = BertTokenizer.from_pretrained(config.PRE_TRAINED_MODEL_NAME)
    elif 'ARES' in config.model_name or 'PROP' in config.model_name:
        cfg = PretrainedConfig.get_config_dict(config.PRE_TRAINED_MODEL_NAME)[0]
        if not config.gradient_checkpointing:
            del cfg["gradient_checkpointing"]
            del cfg["parameter_sharing"]
        cfg = BertConfig.from_dict(cfg)
        model = ARES(config=cfg)
        model.load_state_dict({k.replace("module.", ""): v for k, v in torch.load(f"{config.model_path}/{config.model_name}", map_location={'cuda:0':f'cuda:{config.local_rank}'}).items()},strict=False)
        tokenizer = BertTokenizer.from_pretrained(config.PRE_TRAINED_MODEL_NAME)
    model = model.to(device)
    print("Loading model finish")
    model_prefix = model.base_model_prefix
    model_base = getattr(model, model_prefix)
    if hasattr(model_base, "embeddings"):
        model_embeddings = getattr(model_base, "embeddings")
    lig = LayerIntegratedGradients(model, model_embeddings)
    qd_rank = eval_model(
        model,
        dl2019_qd_loader,
        device,
        config
    )
    print("\n========== Loading visual DL 2019 data ==========")
    visual_dl2019_qd_loader = get_visual_test_qd_loader(qd_rank, dl2019_qs, doc_tokens, docid2id,config)
    visual_model(
        lig,
        tokenizer,
        visual_dl2019_qd_loader,
        df_dl2019_qds,
        device,
        config
    )