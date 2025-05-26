# coding: utf-8
import argparse

import data
from config import Qwen2Config
from model import Qwen2ForCausalLM
from mup import get_shapes, make_base_shapes
from utils import load_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='setting.yml')
    args = parser.parse_args()

    config = load_config(args.input)

    data_path = config['data']['path']
    shape_path = config['shape']['path']

    d_model = config['model']['width']
    ffn_ratio = config['model']['ffn_ratio']
    nhead = config['model']['nhead']
    nkvhead = config['model']['nkvhead']
    nlayers = config['model']['nlayers']
    dropout = config['model']['dropout']
    tied = config['model']['tied']
    init_var = config['model']['init_var']

    corpus = data.Corpus(data_path)
    ntokens = len(corpus.dictionary)

    common_config = dict(
        vocab_size=ntokens,
        hidden_act="silu",
        attn_implementation="eager",
        num_attention_heads=nhead,
        num_key_value_heads=nkvhead,
        num_hidden_layers=nlayers,
        embd_pdrop=dropout,
        attn_pdrop=dropout,
        resid_pdrop=dropout,
        tie_word_embeddings=tied,
        initializer_range=init_var,
    )
    
    print(f'saving base shapes at {shape_path}')
    base_config = Qwen2Config(
        hidden_size=d_model,
        intermediate_size=int(d_model * ffn_ratio),
        **common_config
    )
    base_model = Qwen2ForCausalLM(base_config)
    base_shapes = get_shapes(base_model)

    print("creating delta model...")
    delta_config = Qwen2Config(
        hidden_size=d_model // 2,
        intermediate_size=int(d_model * ffn_ratio // 2),
        **common_config
    )
    delta_model = Qwen2ForCausalLM(delta_config)
    delta_shapes = get_shapes(delta_model)

    make_base_shapes(base_shapes, delta_shapes, savefile=shape_path)
    print('done and exit')
