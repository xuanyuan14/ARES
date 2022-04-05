'''
@ref: Axiomatically Regularized Pre-training for Ad hoc Search
@author: Jia Chen, Yiqun Liu, Yan Fang, Jiaxin Mao, Hui Fang, Shenghao Yang, Xiaohui Xie, Min Zhang, Shaoping Ma.
'''
# encoding: utf-8
import argparse
import pprint
import yaml


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def config_str(self):
        return pprint.pformat(self.__dict__)

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += self.config_str
        return config_str

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            kwargs = yaml.load(f)

        return Config(**kwargs)


def read_config(path):
    return Config.load(path)


def get_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initilized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser()
    # Training
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--epochs', type=int, default=20,
                        help='num_epochs')
    parser.add_argument('--batch_size', type=int, default=25,
                        help='batch size')
    parser.add_argument('--neg_docs_per_q', type=int, default=4,
                        help='number of sampled docs per q-d pair')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument('--lr', type=float, default=3e-5,
                        help='learning rate')
    parser.add_argument('--clip', type=float, default=1.0,
                        help='gradient clip norm')
    parser.add_argument('--warm_up', type=float, default=0.1,
                        help='warm up proportion')
    parser.add_argument('--gradient_checkpointing', action="store_true")
    parser.add_argument('--max_len', type=int, default=512,
                        help='max length')
    parser.add_argument('--max_q_len', type=int, default=15,
                        help='max query length')
    parser.add_argument('--model_name', type=str, default='ARES_simple',
                        help='the model name')
    parser.add_argument('--model_type', type=str, default='ARES',
                        choices=['ARES', 'PROP', 'BERT', 'ICT'],
                        help='the model type')
    parser.add_argument('--optim', type=str, default='adamw',
                        choices=['adam', 'amsgrad', 'adagrad', 'adamw'],
                        help='optimizer')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--embed_dim', type=int, default=100)
    parser.add_argument('--freeze', type=bool, default=False)
    parser.add_argument('--world_size', type=int, default=4)
    parser.add_argument('--distributed_train', action="store_true")
    parser.add_argument('--gpu_num', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--PRE_TRAINED_MODEL_NAME', default='/path/to/ares-simple/',
                        help='huggingface model name')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--load_ckpt', action="store_true", help='whether to load a trained checkpoint')
    parser.add_argument('--model_path', default='model_state_ARES', help='name of checkpoint to load')
    parser.add_argument('--print_every', default=200)
    parser.add_argument('--local_rank', type=int, default=0, help='node rank for distributed training')

    # human labels
    parser.add_argument('--train_qd_dir', default='../preprocess/msmarco-doctrain-qrels.tsv')
    parser.add_argument('--test_qd_dir', default='../preprocess/dev-qrels.txt')
    parser.add_argument('--dl2019_qd_dir', default='../preprocess/2019qrels-docs.txt')

    # queries
    parser.add_argument('--train_qs_dir', default='../preprocess/queries.doctrain.json')
    parser.add_argument('--test_qs_dir', default='../preprocess/queries.docdev.json')
    parser.add_argument('--dl2019_qs_dir', default='../preprocess/queries.dl2019.json')

    # docs
    parser.add_argument('--memmap_doc_dir', default='../preprocess/doc_token_ids.memmap')
    parser.add_argument('--docid2id_dir', default='../preprocess/docid2idx.json')

    # STAR+ADORE Top100
    parser.add_argument('--train100_dir', default='../preprocess/train.rank.tsv')
    parser.add_argument('--test100_dir', default='../preprocess/dev.rank.tsv')
    parser.add_argument('--dl100_dir', default='../preprocess/test.rank.tsv')

    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)