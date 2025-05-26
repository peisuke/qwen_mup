import pandas as pd
import torch
import torch.nn.functional as F


#def get_batch(source, i, seq_len):
#    seq_len = min(seq_len, len(source) - 1 - i)
#    data = source[i:i+seq_len]
#    target = source[i+1:i+1+seq_len]
#    batch = {
#        "input_ids": data,
#        "attention_mask": torch.ones_like(data),
#        "labels": target,
#    }
#    return batch
#
#def batchloader(train_data, seq_len):
#    for batch, i in enumerate(range(0, train_data.size(0) - 1, seq_len)):
#        yield get_batch(train_data, i, seq_len)

def train(model, opt, dataloader, nsteps=3, dict_in_out=False, flatten_input=False, flatten_output=False, 
          output_name='loss', lossfn='xent', cuda=True):
    model.train()
    train_loss = 0
    count = 0
    #dataloader = batchloader(dataset, batch_size)
    for batch_idx, batch in enumerate(dataloader, 1):
        data_len = None
        if dict_in_out:
            data_len = list(batch.values())[0].shape[0]
            if cuda:                        
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.cuda()
            outputs = model(**batch)
            loss = outputs[output_name] if isinstance(outputs, dict) else outputs[0]
        else:
            (x, target) = batch
            data_len = x.shape[0]
            if cuda:
                x, target = x.cuda(), target.cuda()
            if flatten_input:
                x = x.view(x.size(0), -1)
            output = model(x)
            if flatten_output:
                output = output.view(-1, output.shape[-1])
            if lossfn == 'xent':
                loss = F.cross_entropy(output, target)
            elif lossfn == 'mse':
                loss = F.mse_loss(output, F.one_hot(target, num_classes=output.size(-1)).float())
            elif lossfn == 'nll':
                loss = F.nll_loss(output, target)
            else:
                raise NotImplementedError()
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        train_loss += loss.item() * data_len  # sum up batch loss
        count += data_len
        #if batch_idx == nsteps: break
        if count > nsteps:
            break

    train_loss /= count

    return train_loss

def get_lr_data(models, optimizers, mup=True, nsteps=3,
                dict_in_out=False, flatten_input=False, flatten_output=False, 
                output_name='loss', lossfn='xent', filter_module_by_name=None,
                fix_data=True, cuda=True, 
                output_fdict=None, input_fdict=None, param_fdict=None,
                show_progress=True):
    df = []

    for key, gen in models.items():
        gen_model = gen["model"]
        gen_data = gen["data"]
        gen_loader = gen["loader"]
        print(key)

        if show_progress:
            from tqdm import tqdm
            pbar = tqdm(total=len(optimizers))

        for log2lr, gen_optimizer in optimizers.items():
            model = gen_model()
            dataset = gen_data()
            dataloader = gen_loader(dataset)
            opt = gen_optimizer(model)
            train_loss = train(model, opt, dataloader, nsteps=nsteps, dict_in_out=dict_in_out, output_name=output_name, cuda=cuda)

            df.append({
                "key": key,
                "log2lr": log2lr,
                "loss": train_loss,
            })

            if show_progress:
                pbar.update(1)
        if show_progress:
            pbar.close()
    return pd.DataFrame(df)
