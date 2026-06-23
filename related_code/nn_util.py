import torch

def softmax(in_features, dim):
    
    max_values = torch.max(in_features, dim = dim, keepdim= True).values
    shifted = in_features - max_values
    exp_values = torch.exp(shifted)
    denominator = torch.sum(exp_values, dim = dim, keepdim= True)

    return exp_values / denominator

def cross_entropy(inputs, target):
    batch_size = inputs.shape[0]
    max_values = torch.max(inputs, dim = -1, keepdim= True).values
    shifted = inputs - max_values
    
    exp_shifted = torch.exp(shifted)
    log_sum_exp = torch.log(torch.sum(exp_shifted, dim = -1, keepdim= True)) + max_values
    log_sum_exp = log_sum_exp.squeeze(-1)

    correct_logits = inputs[torch.arange(batch_size, device= inputs.device), target]

    losses = log_sum_exp - correct_logits
    return torch.mean(losses)

def gradient_clipping(parameters, max_l2_norm):
    with torch.no_grad():
        sum_squared_norm = 0
        for param in parameters:
            if param.grad is None:
                continue
            sum_squared_norm += torch.sum(param.grad ** 2)
            total_norm = torch.sqrt(sum_squared_norm)

        if total_norm > max_l2_norm:
            scale = max_l2_norm / (total_norm + 1e-6)
            for param in parameters:
                if param.grad is None:
                    continue
                param.grad *= scale
    
