"""
Monkey patched are mostly here
"""
from .task import (  # noqa: F401
    masked_definition_full_task,
    masked_definition_partial_task,
    piqa_task,
    wic_task,
)
from .criterion import wic_criterion, masked_lm_prediction_criterion  # noqa: F401
from . import lamb_optimizer  # noqa: F401


from fairseq.models.roberta import RobertaModel


def roberta_model_upgrade_state_dict_named(self, state_dict, name):
    """
    We are deleting unnecessary things from the checkpoints and resizing embeddings
    this ensures the pre-trained roberta checkpoints still able to be loaded
    """
    import torch

    super(RobertaModel, self).upgrade_state_dict_named(state_dict, name)

    prefix = name + "." if name != "" else ""
    current_head_names = (
        []
        if not hasattr(self, "classification_heads")
        else self.classification_heads.keys()
    )

    self_state_dict = self.state_dict()
    # Hacks: handle different-sized embedding
    with torch.no_grad():
        # NOTE: should check if the dictionaries actually match

        embedding_names = [
            "decoder.sentence_encoder.embed_tokens.weight",
            "decoder.sentence_encoder.embed_positions.weight",
            "decoder.lm_head.weight",
            "decoder.lm_head.bias",
        ]

        for embedding_name in embedding_names:
            self_tensor = self_state_dict.get(embedding_name, None)
            if self_tensor is None:
                continue

            load_tensor = state_dict[prefix + embedding_name]
            if self_tensor.size() > load_tensor.size():
                new_tensor = self_tensor.new_zeros(self_tensor.size())
                new_tensor[: load_tensor.size(0)] = load_tensor
                state_dict[embedding_name] = new_tensor
                print(
                    f"WARNING: {embedding_name} size mismatch, "
                    f"extend with zero tensors {load_tensor.size()} -> {self_tensor.size()}"
                )
            elif self_tensor.size() < load_tensor.size():
                state_dict[embedding_name] = load_tensor[: self_tensor.size(0)]
                print(
                    f"WARNING: {embedding_name} size mismatch, "
                    f"truncate {load_tensor.size()} -> {self_tensor.size()}"
                )

    # Handle new classification heads present in the state dict.
    keys_to_delete = []
    for k in state_dict.keys():
        if not k.startswith(prefix + "classification_heads."):
            continue

        head_name = k[len(prefix + "classification_heads.") :].split(".")[0]
        num_classes = state_dict[
            prefix + "classification_heads." + head_name + ".out_proj.weight"
        ].size(0)
        inner_dim = state_dict[
            prefix + "classification_heads." + head_name + ".dense.weight"
        ].size(0)

        if getattr(self.args, "load_checkpoint_heads", False):
            if head_name not in current_head_names:
                self.register_classification_head(head_name, num_classes, inner_dim)
        else:
            if head_name not in current_head_names:
                print(
                    "WARNING: deleting classification head ({}) from checkpoint "
                    "not present in current model: {}".format(head_name, k)
                )
                keys_to_delete.append(k)
            elif (
                num_classes
                != self.classification_heads[head_name].out_proj.out_features
                or inner_dim != self.classification_heads[head_name].dense.out_features
            ):
                print(
                    "WARNING: deleting classification head ({}) from checkpoint "
                    "with different dimensions than current model: {}".format(
                        head_name, k
                    )
                )
                keys_to_delete.append(k)

    # handle removed LM Head
    for k in state_dict.keys():
        if not k.startswith(prefix + "decoder.lm_head."):
            continue
        if k[len(prefix) :] not in self_state_dict:
            keys_to_delete.append(k)
            print(f"WARNING: deleting lm head ({k}) from checkpoint")

    for k in keys_to_delete:
        del state_dict[k]

    # Copy any newly-added classification heads into the state dict
    # with their current weights.
    if hasattr(self, "classification_heads"):
        cur_state = self.classification_heads.state_dict()
        for k, v in cur_state.items():
            if prefix + "classification_heads." + k not in state_dict:
                print("Overwriting", prefix + "classification_heads." + k)
                state_dict[prefix + "classification_heads." + k] = v


RobertaModel.upgrade_state_dict_named = roberta_model_upgrade_state_dict_named

from fairseq.tasks.sentence_prediction import SentencePredictionTask


def sentence_prediction_add_args(parser):
    """
    Add task-specific arguments to the parser. Add truncate_sequence compared to the
    original add_args. Maked init_token and separator_token actually tokens instead of
    indices in the original add_args.
    """
    parser.add_argument("data", metavar="FILE", help="file prefix for data")
    parser.add_argument("--num-classes", type=int, default=-1, help="number of classes")
    parser.add_argument(
        "--init-token",
        type=str,
        default=None,
        help="add token at the beginning of each batch item",
    )
    parser.add_argument(
        "--separator-token",
        type=str,
        default=None,
        help="add separator token between inputs",
    )
    parser.add_argument("--regression-target", action="store_true", default=False)
    parser.add_argument("--no-shuffle", action="store_true", default=False)
    parser.add_argument(
        "--truncate-sequence",
        action="store_true",
        default=False,
        help="truncate sequence to max_positions",
    )
    parser.add_argument(
        "--add-prev-output-tokens",
        action="store_true",
        default=False,
        help="add prev_output_tokens to sample, used for encoder-decoder arch",
    )


SentencePredictionTask.add_args = staticmethod(sentence_prediction_add_args)


def sentence_prediction_init(self, args, data_dictionary, label_dictionary):
    """
    Treat init_token and separator token as tokens instead of indices.
    """
    super(SentencePredictionTask, self).__init__(args)
    self.dictionary = data_dictionary
    self._label_dictionary = label_dictionary
    if not hasattr(args, "max_positions"):
        self._max_positions = (args.max_source_positions, args.max_target_positions)
    else:
        self._max_positions = args.max_positions
    args.tokens_per_sample = self._max_positions

    init_index = data_dictionary.add_symbol(args.init_token)
    separator_index = data_dictionary.add_symbol(args.separator_token)
    print(
        f"| init {args.init_token} ({init_index}) "
        f"| sep {args.separator_token} ({separator_index}) "
    )
    args.init_token = init_index
    args.separator_token = separator_index


SentencePredictionTask.__init__ = sentence_prediction_init


from fairseq.tasks.sentence_ranking import SentenceRankingTask


def sentence_ranking_add_args(parser):
    """Add task-specific arguments to the parser. Add truncate_sequence compared to the
    original add_args. Maked init_token and separator_token actually tokens instead of
    indices in the original add_args."""
    parser.add_argument("data", metavar="FILE", help="file prefix for data")
    parser.add_argument(
        "--num-classes", type=int, help="number of sentences to be ranked"
    )
    parser.add_argument(
        "--init-token", type=str, help="add token at the beginning of each batch item"
    )
    parser.add_argument(
        "--separator-token", type=str, help="add separator token between inputs"
    )
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument(
        "--truncate-sequence",
        action="store_true",
        help="Truncate sequence to max_positions",
    )
    parser.add_argument(
        "--max-option-length", type=int, help="max length for each option"
    )


SentenceRankingTask.add_args = staticmethod(sentence_ranking_add_args)


def sentence_ranking_init(self, args, dictionary):
    """
    Treat init_token and separator token as tokens instead of indices.
    """
    super(SentenceRankingTask, self).__init__(args)
    self.dictionary = dictionary

    init_index = dictionary.add_symbol(args.init_token)
    separator_index = dictionary.add_symbol(args.separator_token)
    print(
        f"| init {args.init_token} ({init_index}) "
        f"| sep {args.separator_token} ({separator_index}) "
    )
    args.init_token = init_index
    args.separator_token = separator_index


SentenceRankingTask.__init__ = sentence_ranking_init


from fairseq.data.encoders.gpt2_bpe import GPT2BPE


def gpt2bpe_decode(self, x: str) -> str:
    """
    look out for special tokens
    """
    return self.bpe.decode(
        [
            int(tok) if tok not in {"<s>", "<unk>", "</s>", "<mask>"} else tok
            for tok in x.split()
        ]
    )


GPT2BPE.decode = gpt2bpe_decode

from fairseq.data.encoders.gpt2_bpe_utils import Encoder


def encoder_decode(self, tokens):
    text = "".join([self.decoder.get(token, token) for token in tokens])
    text = bytearray([self.byte_decoder[c] for c in text]).decode(
        "utf-8", errors=self.errors
    )
    return text


Encoder.decode = encoder_decode


"""
this won't work, the function is already called when sempre is being imported
see the new train file for solution

from fairseq import options


def options_add_checkpoint_args(parser):
    group = options.add_checkpoint_args(parser)
    # fmt: off
    group.add_argument('--patience', type=int, default=-1, metavar='N',
                       help=('early stop training if valid performance doesn\'t '
                             'improve for N consecutive validation runs; note '
                             'that this is influenced by --validate-interval'))
    # fmt: on
    return group


options.add_checkpoint_args = options_add_checkpoint_args
"""

# NOTE: the following is for fairseq.meters
# there is a lot of from ... import ... in the codebase
# we can simply redefine the meter class to backport

import time
from typing import Optional

import torch
import numpy as np


def type_as(a, b):
    if torch.is_tensor(a) and torch.is_tensor(b):
        return a.to(b)
    else:
        return a


def safe_round(number, ndigits):
    if hasattr(number, "__round__"):
        return round(number, ndigits)
    elif torch is not None and torch.is_tensor(number) and number.numel() == 1:
        return safe_round(number.item(), ndigits)
    elif np is not None and np.ndim(number) == 0 and hasattr(number, "item"):
        return safe_round(number.item(), ndigits)
    else:
        return number


from fairseq.meters import AverageMeter


def averagemeter_init(self, round: Optional[int] = None):
    self.round = round
    self.reset()


AverageMeter.__init__ = averagemeter_init


def averagemeter_reset(self):
    self.val = None  # most recent update
    self.sum = 0  # sum from all updates
    self.count = 0  # total n from all updates


AverageMeter.reset = averagemeter_reset


def averagemeter_update(self, val, n=1):
    if val is not None:
        self.val = val
        if n > 0:
            self.sum = type_as(self.sum, val) + (val * n)
            self.count = type_as(self.count, n) + n


AverageMeter.update = averagemeter_update


def averagemeter_state_dict(self):
    return {
        "val": self.val,
        "sum": self.sum,
        "count": self.count,
        "round": self.round,
    }


AverageMeter.state_dict = averagemeter_state_dict


def averagemeter_load_state_dict(self, state_dict):
    self.val = state_dict["val"]
    self.sum = state_dict["sum"]
    self.count = state_dict["count"]
    self.round = state_dict.get("round", None)


AverageMeter.load_state_dict = averagemeter_load_state_dict


def averagemeter_avg(self):
    return self.sum / self.count if self.count > 0 else self.val


AverageMeter.avg = property(averagemeter_avg)


def averagemeter_smoothed_value(self) -> float:
    val = self.avg
    if self.round is not None and val is not None:
        val = safe_round(val, self.round)
    return val


AverageMeter.smoothed_value = property(averagemeter_smoothed_value)


from fairseq.meters import TimeMeter


def timemeter_init(self, init: int = 0, n: int = 0, round: Optional[int] = None):
    self.round = round
    self.reset(init, n)


TimeMeter.__init__ = timemeter_init


def timemeter_reset(self, init=0, n=0):
    self.init = init
    self.start = time.perf_counter()
    self.n = n
    self.i = 0


TimeMeter.reset = timemeter_reset


def timemeter_update(self, val=1):
    self.n = type_as(self.n, val) + val


TimeMeter.update = timemeter_update


def timemeter_state_dict(self):
    return {"init": self.elapsed_time, "n": self.n, "round": self.round}


TimeMeter.state_dict = timemeter_state_dict


def timemeter_load_state_dict(self, state_dict):
    if "start" in state_dict:
        # backwards compatibility for old state_dicts
        self.reset(init=state_dict["init"])
    else:
        self.reset(init=state_dict["init"], n=state_dict["n"])
        self.round = state_dict.get("round", None)


TimeMeter.load_state_dict = timemeter_load_state_dict


def timemeter_elapsed_time(self):
    return self.init + (time.perf_counter() - self.start)


TimeMeter.elapsed_time = property(timemeter_elapsed_time)


def timemeter_smoothed_value(self) -> float:
    val = self.avg
    if self.round is not None and val is not None:
        val = safe_round(val, self.round)
    return val


TimeMeter.smoothed_value = property(timemeter_smoothed_value)


from fairseq.meters import StopwatchMeter


def stopwatchmeter_init(self, round: Optional[int] = None):
    self.round = round
    self.sum = 0
    self.n = 0
    self.start_time = None


StopwatchMeter.__init__ = stopwatchmeter_init


def stopwatchmeter_start(self):
    self.start_time = time.perf_counter()


StopwatchMeter.start = stopwatchmeter_start


def stopwatchmeter_stop(self, n=1):
    if self.start_time is not None:
        delta = time.perf_counter() - self.start_time
        self.sum = self.sum + delta
        self.n = type_as(self.n, n) + n


StopwatchMeter.stop = stopwatchmeter_stop


def stopwatchmeter_reset(self):
    self.sum = 0  # cumulative time during which stopwatch was active
    self.n = 0  # total n across all start/stop
    self.start()


StopwatchMeter.reset = stopwatchmeter_reset


def stopwatchmeter_state_dict(self):
    return {"sum": self.sum, "n": self.n, "round": self.round}


StopwatchMeter.state_dict = stopwatchmeter_state_dict


def stopwatchmeter_load_state_dict(self, state_dict):
    self.sum = state_dict["sum"]
    self.n = state_dict["n"]
    self.start_time = None
    self.round = state_dict.get("round", None)


StopwatchMeter.load_state_dict = stopwatchmeter_load_state_dict


def stopwatchmeter_avg(self):
    return self.sum / self.n if self.n > 0 else self.sum


StopwatchMeter.avg = property(stopwatchmeter_avg)


def stopwatchmeter_elapsed_time(self):
    if self.start_time is None:
        return 0.0
    return time.perf_counter() - self.start_time


StopwatchMeter.elapsed_time = property(stopwatchmeter_elapsed_time)


def stopwatchmeter_smoothed_value(self) -> float:
    val = self.avg if self.sum > 0 else self.elapsed_time
    if self.round is not None and val is not None:
        val = safe_round(val, self.round)
    return val


StopwatchMeter.smoothed_value = property(stopwatchmeter_smoothed_value)
