'''
@ref: Axiomatically Regularized Pre-training for Ad hoc Search
@author: Jia Chen, Yiqun Liu, Yan Fang, Jiaxin Mao, Hui Fang, Shenghao Yang, Xiaohui Xie, Min Zhang, Shaoping Ma.
'''
# encoding: utf-8
import sys
import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MarginRankingLoss
from torch.nn import Softmax
from torch.cuda.amp import autocast
from transformers import BertModel, BertPreTrainedModel


sys.path.insert(0, '../')
PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'spanbert-base-cased': "https://dl.fbaipublicfiles.com/fairseq/models/spanbert_hf_base.tar.gz",
    'spanbert-large-cased': "https://dl.fbaipublicfiles.com/fairseq/models/spanbert_hf.tar.gz"
}


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


# TransformerICT
class ICT(BertPreTrainedModel):
    def __init__(self, config):
        super(ICT, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls = nn.Linear(config.hidden_size, 1)
        self.cls.predictions = BertLMPredictionHead(config, self.bert.embeddings.word_embeddings.weight)
        self.config = config

        self.init_weights()

    @autocast()
    def forward(self, input_ids, config, input_mask, token_type_ids=None, masked_lm_labels=None, device=None):

        batch_size = input_ids.size(0)
        outputs = self.bert(input_ids,
                            attention_mask=input_mask,
                            return_dict=False
                            )

        sequence_output, pooled_output = outputs[0], outputs[1]

        if masked_lm_labels is not None:
            # MLM loss
            lm_prediction_scores = self.cls.predictions(sequence_output)
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            mlm_loss = loss_fct(lm_prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1)) if config.MLM else 0.

            # ICT loss
            logits = pooled_output.reshape(batch_size//2, 2, self.config.hidden_size)
            s_encode = logits[:, 0, :]  # bs/2, 1, h
            c_encode = logits[:, 1, :]  # bs/2, 1, h

            logit = torch.matmul(s_encode, c_encode.transpose(-2, -1))
            target = torch.from_numpy(np.array([i for i in range(batch_size // 2)])).long().to(device)
            loss = nn.CrossEntropyLoss()
            ict_loss = loss(logit, target).mean()

            loss = mlm_loss + ict_loss
            return loss

        else:
            prediction_scores = self.cls(self.dropout(pooled_output))
            return prediction_scores


class ARES(BertPreTrainedModel):
    def __init__(self, config):
        super(ARES, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls = nn.Linear(config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.cls.predictions = BertLMPredictionHead(config, self.bert.embeddings.word_embeddings.weight)
        self.config = config

        self.init_weights()

    @autocast()
    def forward(self, input_ids, config, input_mask, token_type_ids, masked_lm_labels=None, device=None):

        batch_size = input_ids.size(0)
        outputs = self.bert(input_ids,
                            attention_mask=input_mask,
                            token_type_ids=token_type_ids,
                            return_dict=False
                            )

        sequence_output, pooled_output = outputs[0], outputs[1]
        prediction_scores = self.cls(self.dropout(pooled_output))

        if masked_lm_labels is not None:
            # MLM loss
            lm_prediction_scores = self.cls.predictions(sequence_output)
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            mlm_loss = loss_fct(lm_prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1)) if config.MLM else 0.

            # Pairwise loss
            logits = prediction_scores.reshape(batch_size // 2, 2)
            softmax = Softmax(dim=1)
            logits = softmax(logits)
            pos_logits = logits[:, 0]
            neg_logits = logits[:, 1]
            marginloss = MarginRankingLoss(margin=1.0, reduction='mean')

            rep_label = torch.ones_like(pos_logits)
            rep_loss = marginloss(pos_logits, neg_logits, rep_label)

            loss = mlm_loss + rep_loss
            return loss
        else:
            return prediction_scores
