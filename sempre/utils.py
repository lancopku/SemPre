import os

from fairseq.data import Dictionary, encoders
from fairseq.models.roberta import RobertaModel
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from .data.dictionary import DictionaryDiscreet


def none_or_str(value):
    if value == "None":
        return None
    return value


def load_dictionary(args, ensure_bpe=True, discreet=False, name="dict.txt"):
    bpe = encoders.build_bpe(args)
    if ensure_bpe:
        assert bpe is not None, "Must set --bpe"

    if discreet:
        dictionary = DictionaryDiscreet.load(
            os.path.join(args.data.split(os.pathsep)[0], name)
        )
    else:
        dictionary = Dictionary.load(os.path.join(args.data.split(os.pathsep)[0], name))

    return dictionary, bpe


def delete_lm_head(model):
    if isinstance(model, RobertaModel):
        del model.decoder.lm_head


def binarize(text, vocab, bpe, append_eos=True):
    bpe_str = bpe.encode(text)
    tokens = [vocab.index(token) for token in bpe_str.split()]
    if append_eos:
        tokens = tokens + [vocab.eos()]
    return tokens


def apply_bert_init(module):
    module.apply(init_bert_params)
