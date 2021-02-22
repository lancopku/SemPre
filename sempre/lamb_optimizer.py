import math

import torch
import torch.optim
import torch.distributed as dist

from fairseq.optim import FairseqOptimizer, register_optimizer


@register_optimizer("lamb")
class FairseqLAMB(FairseqOptimizer):
    """LAMB optimizer for fairseq."""

    def __init__(self, args, params):
        super().__init__(args)

        grad_averaging = not getattr(self.args, "no_grad_averaging", False)
        bias_correction = not getattr(self.args, "no_bias_correction", False)
        set_grad_none = not getattr(self.args, "no_set_grad_none", False)

        if torch.cuda.is_available():
            try:
                from apex.optimizers import FusedLAMB as _FusedLAMB  # noqa

                self._optimizer = _FusedLAMB(
                    params,
                    adam_w_mode=True,
                    bias_correction=bias_correction,
                    grad_averaging=grad_averaging,
                    set_grad_none=set_grad_none,
                    max_grad_norm=self.args.clip_norm,
                    **self.optimizer_config,
                )
            except ImportError:
                self._optimizer = LAMB(
                    params,
                    bias_correction=bias_correction,
                    grad_averaging=grad_averaging,
                    set_grad_none=set_grad_none,
                    **self.optimizer_config,
                )
        else:
            self._optimizer = LAMB(
                params,
                bias_correction=bias_correction,
                grad_averaging=grad_averaging,
                set_grad_none=set_grad_none,
                **self.optimizer_config,
            )

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--lamb-betas', default='(0.9, 0.999)', metavar='B',
                            help='betas for Adam optimizer')
        parser.add_argument('--lamb-eps', type=float, default=1e-6, metavar='D',
                            help='epsilon for Adam optimizer')
        parser.add_argument('--weight-decay', '--wd', default=0.01, type=float, metavar='WD',
                            help='weight decay')
        parser.add_argument('--no-grad-averaging', action='store_true')
        parser.add_argument('--no-bias-correction', action='store_true')
        parser.add_argument('--no-set-grad-none', action='store_true')
        # fmt: on

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            "lr": self.args.lr[0],
            "betas": eval(self.args.lamb_betas),
            "eps": self.args.lamb_eps,
            "weight_decay": self.args.weight_decay,
        }

    def average_params(self):
        """Reduce Params is only used during BMUF distributed training."""
        state_dict = self.optimizer.state_dict()
        total_gpus = float(dist.get_world_size())

        for _, value in state_dict["state"].items():
            value["exp_avg"] /= total_gpus
            value["exp_avg_sq"] /= total_gpus
            dist.all_reduce(value["exp_avg"], op=dist.ReduceOp.SUM)
            dist.all_reduce(value["exp_avg_sq"], op=dist.ReduceOp.SUM)


class LAMB(torch.optim.Optimizer):
    """Implements LAMB algorithm.
    LAMB was proposed in `Large Batch Optimization for Deep Learning:
      Training BERT in 76 minutes` (https://arxiv.org/abs/1904.00962).
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        bias_correction (bool, optional): whehter apply bias correction. (default: True)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its norm. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        grad_averaging (bool, optional): whether apply (1-beta1) to grad when
            calculating running averages of gradient. (default: True)
        set_grad_none (bool, optional): whether set grad to None when zero_grad()
            method is called. (default: True)

    """

    def __init__(
        self,
        params,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0.01,
        grad_averaging=True,
        set_grad_none=True,
    ):
        defaults = dict(
            lr=lr,
            bias_correction=bias_correction,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            grad_averaging=grad_averaging,
        )
        super(LAMB, self).__init__(params, defaults)

        self.set_grad_none = set_grad_none

    @property
    def supports_memory_efficient_fp16(self):
        return True

    def zero_grad(self):
        if self.set_grad_none:
            for group in self.param_groups:
                for p in group["params"]:
                    p.grad = None
        else:
            super(LAMB, self).zero_grad()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError(
                        "LAMB does not support sparse gradients, please consider SparseAdam instead"
                    )

                p_data_fp32 = p.data.float()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)

                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                beta1, beta2 = group["betas"]
                beta3 = 1 if not group["grad_averaging"] else (1 - beta1)

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(beta3, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                if group["bias_correction"]:
                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    bias_correction = math.sqrt(bias_correction2) / bias_correction1
                else:
                    bias_correction1 = 1
                    bias_correction2 = 1
                    bias_correction = 1

                # reuse grad
                update = grad
                update.zero_()

                # decoupled weight decay
                if group["weight_decay"] != 0:
                    update.add_(group["weight_decay"], p_data_fp32)
                # adam update
                update.addcdiv_(bias_correction, exp_avg, denom)

                # compute trust ratio
                p_norm = p_data_fp32.norm()
                u_norm = update.norm()
                trust_ratio = p_norm / u_norm if p_norm != 0 and u_norm != 0 else 1.0

                # update
                p_data_fp32.add_(-group["lr"] * trust_ratio, update)

                p.data.copy_(p_data_fp32)

        return loss
