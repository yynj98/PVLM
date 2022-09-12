import argparse
import torch
import numpy as np
from os.path import join
from pprint import pprint
import random

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=34, help="random seed for initialization")
parser.add_argument("--cuda", type=str, default='0')
parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
parser.add_argument("--no_img", action="store_true", default=False)

parser.add_argument("--dataset", type=str, default='t2017')
parser.add_argument("--few_shot_file", type=str, default='few-shot.tsv')
parser.add_argument("--template", type=int, default=1)

parser.add_argument("--model_name", type=str, default='bert-base-uncased')
VISUAL_MODELS = ['nf_resnet50', 'resnet50', 'resnetv2_50x1_bitm', 'vit_base_patch16_224']
parser.add_argument("--visual_model_name", type=str, choices=VISUAL_MODELS, default='nf_resnet50')

# learnable template settings
parser.add_argument("--prompt_token", type=str, default='[unused1]')
parser.add_argument("--prompt_shape", type=str, default='333-0')  # 333-0 for aspect-level PT, 333-3 for aspect-level PVLM
parser.add_argument("--lstm_dropout", type=float, default=0.0)

parser.add_argument("--img_token", type=str, default='[unused2]')
parser.add_argument("--img_token_len", type=int, default=3)

parser.add_argument("--data_dir", type=str, default='datasets')
parser.add_argument("--img_dir", type=str)
parser.add_argument("--out_dir", type=str, default='out')

parser.add_argument("--ckpt_name", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr_lm_model", type=float)
parser.add_argument("--lr_visual_encoder", type=float)
parser.add_argument("--decay_rate", type=float, default=0.98)
parser.add_argument("--weight_decay", type=float, default=0.0005)
parser.add_argument("--early_stop", type=int, default=20)


args = parser.parse_args()
args.data_dir = join(args.data_dir, args.dataset)
args.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() and not args.no_cuda else "cpu")
args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()  # only for setting seeds


def set_seed(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


set_seed(args)
pprint(f'[#] args: \n{args}')
