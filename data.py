import torch
import numpy as np

def get_batch(dataset, batch_size, context_length, device):
    max_start = len(dataset) - context_length
    starts = torch.randint(0, max_start,(batch_size,))

    xs = []
    ys = []
    for start in starts:
        start = start.item()
        x_i = dataset[start : start + context_length]
        y_i = dataset[start + 1 : start + context_length + 1]
        xs.append(x_i)
        ys.append(y_i)

    x = torch.tensor(np.array(xs), dtype = torch.long, device = device)
    y = torch.tensor(np.array(ys), dtype = torch.long, device = device)

    return x, y
