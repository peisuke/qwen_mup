import yaml
import torch

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setprec(t, precision):
    if precision == 'half':
        # do nothing since this is handled by AMP
        return t
    elif precision == 'float':
        return t.float()
    elif precision == 'double':
        return t.double()
    else:
        raise ValueError(f'invalid precision string {args.precision}')

def get_batch(source, i, seq_len):
    seq_len = min(seq_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len]
    batch = {
        "input_ids": data,
        "attention_mask": torch.ones_like(data),
        "labels": target,
    }
    return batch

def batchloader(train_data, seq_len):
    for batch, i in enumerate(range(0, train_data.size(0) - 1, seq_len)):
        yield get_batch(train_data, i, seq_len)

def batchify(data, bsz, device):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)
