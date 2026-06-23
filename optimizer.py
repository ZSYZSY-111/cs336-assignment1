import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr = 1e-3, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 0.0):
        defaults = {
            "lr": lr, #学习率
            "betas": betas, #一阶动量和二阶动量里的beta
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure = None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        with torch.no_grad():
            for group in self.param_groups:
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]

                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    grad = p.grad
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    state["step"] += 1
                    t = state["step"]

                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    bias_correction1 = 1 - beta1 ** t
                    bias_correction2 = 1 - beta2 ** t
                    
                    step_size = lr / bias_correction1
                    denom = exp_avg_sq.sqrt().div_(bias_correction2 ** 0.5).add_(eps)

                    p.addcdiv_(exp_avg, denom, value=-step_size)

                    if weight_decay != 0:
                        p.add_(p, alpha=-lr * weight_decay)

        return loss
        

def get_lr_cosine_schedule(
    it,
    max_learning_rate,
    min_learning_rate,
    warmup_iters,
    cosine_cycle_iters,
):
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters

    if it > cosine_cycle_iters:
        return min_learning_rate

    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))

    return min_learning_rate + cosine_factor * (max_learning_rate - min_learning_rate)