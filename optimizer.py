import math
from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer
from sophia.sophia import SophiaG


class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1])
            )
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias
        )
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                # State should be stored in this dictionary
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]

                # Complete the implementation of AdamW here, reading and saving
                # your state in the `state` dictionary above.
                # The hyperparameters can be read from the `group` dictionary
                # (they are lr, betas, eps, weight_decay, and correct_bias, as saved in
                # the constructor).
                #
                # 1- Update first and second moments of the gradients.
                # 2- Apply bias correction.
                #    (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                #     also given as the pseudo-code in the project description).
                # 3- Update parameters (p.data).
                # 4- After that main gradient-based update, update again using weight decay
                #    (incorporating the learning rate again).

                ### TODO
                # raise NotImplementedError

                # Initialize state
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                # Hyperparameters fetching
                alpha = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                correct_bias = group["correct_bias"]

                # Update the step count
                state["step"] += 1
                t = state["step"]

                # 1- Update first and second moments of the gradients.
                state["m"] = beta1 * state["m"] + (1 - beta1) * grad
                state["v"] = beta2 * state["v"] + (1 - beta2) * (grad ** 2)

                # 2- Apply bias correction. 
                if correct_bias:
                    bias_correction1 = 1 - beta1 ** t
                    bias_correction2 = 1 - beta2 ** t
                    alpha *= math.sqrt(bias_correction2) / bias_correction1
                
                # m_hat = state["m"] / bias_correction1
                # v_hat = state["v"] / bias_correction2
                # p.data = p.data - alpha * (m_hat / (torch.sqrt(v_hat) + eps))


                # 3- Update parameters (p.data).
                p.data -= alpha * state["m"] / (torch.sqrt(state["v"]) + eps)

                # 4- After that main gradient-based update, update again using weight decay
                if weight_decay > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * weight_decay)
                    # p.data -= alpha * weight_decay * p.data


        return loss


# Adding Sophia to the optimizer options
def get_optimizer(optimizer_name, params, **kwargs):
    if optimizer_name == 'adam':
        return AdamW(params, **kwargs)
    elif optimizer_name == 'sophia':
        # Use SophiaG with the best-suggested default parameters
        lr = kwargs.get('lr', 2e-4)
        betas = kwargs.get('betas', (0.965, 0.99))
        rho = kwargs.get('rho', 0.01)
        weight_decay = kwargs.get('weight_decay', 1e-4)
        return SophiaG(params, lr=lr, betas=betas, rho=rho, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not recognized")