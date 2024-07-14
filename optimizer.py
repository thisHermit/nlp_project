import math
from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


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
                
                # initilizations

                if len(state) ==0:
                    state["moment_m"] = torch.zeros_like(p.data)
                    state["moment_v"] = torch.zeros_like(p.data)
                    state["step_t"] = 0
                    
                state["step_t"] += 1
                mt, vt =  state["moment_m"], state["moment_v"]
                beta1, beta2 = group['betas']

                # update first and second moments
                mt.mul_(beta1).add_(grad, alpha=1 - beta1)
                vt.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # corr_mt = mt / (beta1 ** state["step_t"])
                # corr_vt = vt / (beta2 ** state["step_t"])

                # efficient method update parameters
                step_size = group["lr"] * math.sqrt(1 - beta2 ** state["step_t"]) / (1 - beta1 ** state["step_t"])
                p.data.addcdiv_(mt , vt.sqrt().add_(group["eps"]) , value = - step_size )

                # Apply weight decay
                if group["weight_decay"] != 0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])



                
                # raise NotImplementedError

        return loss
