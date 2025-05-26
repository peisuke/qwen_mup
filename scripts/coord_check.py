# coding: utf-8
import os
import argparse
import yaml

from functools import partial
import numpy as np
import torch

import data
from config import Qwen2Config
from model import Qwen2ForCausalLM
from mup import set_base_shapes
from mup_coord import get_coord_data
from utils import load_config, setprec, batchify, batchloader

def coord_check(mup,
                lr,
                optimizer,
                batch_size,
                nsteps,
                nseeds,
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
                seq_len,
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
                initializer_range=init_var  # init_varをinitializer_rangeに対応付け（近似）
            )
            model = Qwen2ForCausalLM(config).to(device)

            model = setprec(model, precision)
            if standparam:
                set_base_shapes(model, None)
            else:
                set_base_shapes(model, base_shapes_path)

            model.apply(partial(model._init_weights, readout_zero_init=readout_zero_init, query_zero_init=query_zero_init))
            return model
        return f

    optimizer = optimizer.replace('mu', '')
    widths = 2**np.arange(0, 5) * 40
    models = {w: gen(w, standparam=not mup) for w in widths}

    train_data = batchify(corpus.train, batch_size, device=device)
    loader = batchloader(train_data, seq_len)
    df = get_coord_data(models, loader, mup=mup, lr=lr, optimizer=optimizer,
                        nseeds=nseeds, dict_in_out=True, nsteps=nsteps, lossfn='nll')

    prm = 'muP' if mup else 'SP'
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f'{prm.lower()}_{optimizer}_coord.csv')
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
    precision = config['train']['precision']
    seq_len = config['train']['seq_len']

    # Set the random seed manually for reproducibility.
    torch.manual_seed(seed)
    device = args.device

    coord_args = dict(
        lr=lr,
        optimizer=optimizer,
        batch_size=batch_size,
        nsteps=coord_check_nsteps,
        nseeds=coord_check_nseeds,
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
        seq_len=seq_len,
        base_shapes_path=shape_path,
        output_dir=output_path,
        legend=False
    )
    
    coord_check(mup=True, **coord_args)
    coord_check(mup=False, **coord_args)
