# coding: utf-8
import os
import argparse
import yaml

from functools import partial
import itertools
import numpy as np
import torch

import data
from config import Qwen2Config
from model import Qwen2ForCausalLM
from mup import set_base_shapes
from mup_lr import get_lr_data
from utils import load_config, setprec, batchify, batchloader

def lr_check(mup,
             optimizer,
             batch_size,
             nsteps,
             data_dir,
             ffn_ratio,
             nhead,
             nkvhead,
             nlayers,
             dropout,
             tied,
             init_var,
             precision,
             device,
             base_shapes_path,
             output_dir='',
             legend=False):

    corpus = data.Corpus(data_dir)
    ntokens = len(corpus.dictionary)

    def gen(w, standparam=False, readout_zero_init=True, query_zero_init=True):
        def f():
            config = Qwen2Config(
                vocab_size=ntokens,
                hidden_act="silu",
                attn_implementation="eager",
                hidden_size=int(w),
                intermediate_size=int(w * ffn_ratio),
                num_attention_heads=nhead,
                num_key_value_heads=nkvhead,
                num_hidden_layers=nlayers,
                attn_mult=8 if mup else None,
                embd_pdrop=dropout,
                attn_pdrop=dropout,
                resid_pdrop=dropout,
                tie_word_embeddings=tied,
                initializer_range=init_var
            )
            model = Qwen2ForCausalLM(config).to(device)

            model = setprec(model, precision)
            if standparam:
                set_base_shapes(model, None)
            else:
                set_base_shapes(model, base_shapes_path)
            model.apply(partial(model._init_weights, readout_zero_init=readout_zero_init, query_zero_init=query_zero_init))
            return model.to(device)
        return f

    filter_trainable_by_name = None

    def get_trainable(model):
        params = model.parameters()
        if filter_trainable_by_name is not None:
            params = []
            for name, p in model.named_parameters():
                if filter_trainable_by_name(name):
                    params.append(p)
        return params

    def gen_opt(lr):
        #import model as _model
        def f(model):
            if mup:
                from mup.optim import MuAdam as Adam
                from mup.optim import MuAdamW as AdamW
                from mup.optim import MuSGD as SGD
            else:
                from torch.optim import SGD, Adam, AdamW
            if optimizer == 'sgd':
                op = SGD(get_trainable(model), lr=lr)
            elif optimizer == 'adam':
                op = Adam(get_trainable(model), lr=lr)
            elif optimizer == 'adamw':
                op = AdamW(get_trainable(model), lr=lr)
            return op
        return f

    def gen_data(seq_len):
        def f():
            return batchify(corpus.train, seq_len, device=args.device)
        return f

    def gen_loader(batch_size):
        def f(train_data):
            return batchloader(train_data, batch_size)
        return f

    optimizer = optimizer.replace('mu', '')

    widths = [80, 160, 320, 640]
    #seq_lens = [64, 32, 16, 8]
    #batch_sizes = [16, 8, 4, 2]
    seq_lens = [8]
    batch_sizes = [8]
    log2lrs = np.linspace(-16, 0, 50)

    models = {f"{w=}_{s=}_{b=}": {
        "model": gen(w, standparam=not mup),
        "data": gen_data(s),
        "loader": gen_loader(b)
    } for w, s, b in itertools.product(widths, seq_lens, batch_sizes)}

    optimizers = {log2lr: gen_opt(2**log2lr) for log2lr in log2lrs}
    
    df = get_lr_data(models, optimizers, mup=mup,
                     dict_in_out=True, nsteps=nsteps, lossfn='nll')

    prm = 'muP' if mup else 'SP'
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f'{prm.lower()}_{optimizer}_lr.csv')
    df.to_csv(filepath, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='setting.yml')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    config = load_config(args.input)

    data_path = config['data']['path']
    shape_path = config['shape']['path']
    output_path = config['exp']['output']['path']
    seed = config['exp']['seed']

    d_model = config['model']['width']
    ffn_ratio = config['model']['ffn_ratio']
    nhead = config['model']['nhead']
    nkvhead = config['model']['nkvhead']
    nlayers = config['model']['nlayers']
    dropout = config['model']['dropout']
    tied = config['model']['tied']
    init_var = config['model']['init_var']

    lr = config['train']['lr']
    optimizer  = config['train']['optimizer']
    batch_size = config['train']['batch_size']
    coord_check_nsteps = config['train']['coord_check_nsteps']
    coord_check_nseeds = config['train']['coord_check_nseeds']
    lr_check_nsteps = config['train']['lr_check_nsteps']
    precision = config['train']['precision']
    seq_len = config['train']['seq_len']

    # Set the random seed manually for reproducibility.
    torch.manual_seed(seed)
    device = args.device

    coord_args = dict(
        optimizer=optimizer,
        batch_size=batch_size,
        nsteps=lr_check_nsteps,
        data_dir=data_path,
        ffn_ratio=ffn_ratio,
        nhead=nhead,
        nkvhead=nkvhead,
        nlayers=nlayers,
        dropout=dropout,
        tied=tied,
        init_var=init_var,
        precision=precision,
        device=device,
        base_shapes_path=shape_path,
        output_dir=output_path,
        legend=False
    )
    
    lr_check(mup=True, **coord_args)
    lr_check(mup=False, **coord_args)
