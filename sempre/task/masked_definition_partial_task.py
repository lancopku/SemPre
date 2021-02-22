"""
this uses the preprocessed jsonl formatted wordnet splited into train and valid
valid contains 1000 words from oxford3000 that are not seen in training
see scripts/utils/preprocess_data.py to see how to prepare the data
"""
import json
import os

import numpy as np
import torch

from fairseq.data import (
    Dictionary,
    FairseqDataset,
    IdDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    OffsetTokensDataset,
    PadDataset,
    SortDataset,
    TruncateDataset,
    data_utils,
)
from fairseq.data.encoders.gpt2_bpe import GPT2BPE
from fairseq.tasks import FairseqTask, register_task

from ..data.indexed_raw_label_dataset import IndexedRawLabelDataset
from ..data.mask_token_except_dataset import MaskTokenExceptDataset
from ..utils import apply_bert_init, load_dictionary

# def _extract_bpe_encoded(item_dict):
#     return [int(x) for x in item_dict["bpe_encoded"].split()]


def _extract_raw(item_dict):
    return item_dict["raw"]


def binarize(text, dictionary, bpe=None, prepend_space=False):
    if bpe is not None:
        if isinstance(bpe, GPT2BPE):
            # GPT2BPE is tricky w.r.t. leading spaces
            bpe = bpe.bpe
            # assert prepend_space in ["auto_len", "auto_cap", True, False]
            if prepend_space == "auto_len":
                # encode twice
                tokens_with_space = bpe.encode(" " + text)
                tokens_without_space = bpe.encode(text)
                tokens = min(tokens_with_space, tokens_without_space, key=len)
            else:
                if prepend_space == "auto_cap":
                    prepend_space = not text[0].isupper()

                if prepend_space:
                    text = " " + text
                tokens = bpe.encode(text)
            tokens = map(str, tokens)
        else:
            tokens = bpe.encode(text).split()
    else:
        tokens = text.split()

    return [dictionary.index(token) for token in tokens]


class Synset:

    collection = None  # dict of Synset

    def __init__(self, sid, definition, lemmas):
        self._sid = sid  # str
        self.definition = definition  # list of int
        self.lemmas = lemmas  # list of list of int

    @classmethod
    def from_json_line(cls, line, dictionary, bpe=None):
        item = json.loads(line)
        return cls(
            item["id"],
            binarize(_extract_raw(item["definition"]), dictionary, bpe, True),
            [
                binarize(_extract_raw(x), dictionary, bpe, "auto_len")
                for x in item["lemmas"]
            ],
        )

    @classmethod
    def set_collection_from_jsonl(cls, fp, dictionary, bpe):
        if cls.has_collection():
            return
        collection = {}
        with open(fp, "r", encoding="utf8") as f:
            for line in f:
                synset = cls.from_json_line(line, dictionary, bpe)
                collection[synset._sid] = synset
        cls.collection = collection
        print(f"| Load {len(collection)} synsets")

    @classmethod
    def has_collection(cls):
        return cls.collection is not None

    @classmethod
    def get_synset_from_collection(cls, sid):
        return cls.collection[sid]


class Relation:
    def __init__(self, relation, head_sid, head_lemma, tail_sid, tail_lemma):
        self.relation = relation
        self._head_sid = head_sid
        self._tail_sid = tail_sid
        self.head_lemma = head_lemma
        self.tail_lemma = tail_lemma

    @property
    def head_definition(self):
        return Synset.get_synset_from_collection(self._head_sid).definition

    @property
    def tail_definition(self):
        return Synset.get_synset_from_collection(self._tail_sid).definition

    @classmethod
    def from_json_line(cls, line, dictionary, bpe):
        """ read date and process online
        """
        item = json.loads(line)
        relation = item["relation"]
        head_sid = item["head_id"]
        tail_sid = item["tail_id"]
        if "head_lemmas" in item:
            head_lemmas = [
                binarize(_extract_raw(x), dictionary, bpe, "auto_len")
                for x in item["head_lemmas"]
            ]
        else:
            head_lemmas = Synset.get_synset_from_collection(head_sid).lemmas

        if "tail_lemmas" in item:
            tail_lemmas = [
                binarize(_extract_raw(x), dictionary, bpe, "auto_len")
                for x in item["tail_lemmas"]
            ]
        else:
            tail_lemmas = Synset.get_synset_from_collection(tail_sid).lemmas

        for h in head_lemmas:
            for t in tail_lemmas:
                yield cls(relation, head_sid, h, tail_sid, t)

    @staticmethod
    def load_collection_from_jsonl(fp, dictonary, bpe):
        assert Synset.has_collection(), "Must read synset info before reading relations"
        relations = []
        with open(fp, "r", encoding="utf8") as f:
            for line in f:
                for relation in Relation.from_json_line(line, dictonary, bpe):
                    relations.append(relation)
        return relations


class DefinitionPairDataset(FairseqDataset):
    def __init__(
        self,
        relations,
        eos_token_idx=None,
        sur_token_idx=None,
        def_sep_token_idx=None,
        init_token_idx=None,
        sep_token_idx=None,
        return_types=False,
    ):

        super().__init__()
        # format init sur head_lemma sur def_sep head_def eos sep sur tail_lemma sur def_sep tail_def eos

        assert eos_token_idx is not None
        assert sur_token_idx is not None
        assert def_sep_token_idx is not None
        assert init_token_idx is not None
        assert sep_token_idx is not None

        self.eos_token_idx = eos_token_idx
        self.sur_token_idx = sur_token_idx
        self.def_sep_token_idx = def_sep_token_idx
        self.init_token_idx = init_token_idx
        self.sep_token_idx = sep_token_idx

        self.relations = relations
        self.num_extra_tokens = 1 + 4 + 2 + 2 + 1

        sizes = [
            len(relation.head_lemma)
            + len(relation.head_definition)
            + len(relation.tail_lemma)
            + len(relation.tail_definition)
            for relation in relations
        ]
        self.sizes = np.array(sizes) + self.num_extra_tokens
        self.return_types = return_types

    def __getitem__(self, i):
        relation = self.relations[i]

        if self.return_types:
            types = [0] * (
                6 + len(relation.head_lemma) + len(relation.head_definition)
            ) + [1] * (4 + len(relation.tail_lemma) + len(relation.tail_definition))

            assert len(types) == self.sizes[i]
            return torch.tensor(  # pylint: disable=not-callable
                types, dtype=torch.int64
            )

        tokens = (
            [self.init_token_idx, self.sur_token_idx]
            + relation.head_lemma
            + [self.sur_token_idx, self.def_sep_token_idx]
            + relation.head_definition
            + [self.eos_token_idx, self.sep_token_idx, self.sur_token_idx]
            + relation.tail_lemma
            + [self.sur_token_idx, self.def_sep_token_idx]
            + relation.tail_definition
            + [self.eos_token_idx]
        )
        assert len(tokens) == self.sizes[i]
        return torch.tensor(tokens, dtype=torch.int64)  # pylint: disable=not-callable

    def get_string(self, i):
        raise NotImplementedError

    def __del__(self):
        pass

    def __len__(self):
        return len(self.relations)

    def size(self, index):
        return self.sizes[index]


@register_task("masked_lm_prediction")
class MaskedLMPredictionTask(FairseqTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "data",
            help="colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner",
        )
        parser.add_argument("--mask-token", type=str, default="<mask>")
        parser.add_argument("--separator-token", type=str, default="</s>")
        parser.add_argument("--init-token", type=str, default="<s>")
        parser.add_argument("--surround-token", type=str, default="<sur>")
        parser.add_argument("--def-sep-token", type=str, default=":")
        parser.add_argument(
            "--tokens-per-sample",
            default=512,
            type=int,
            help="max number of total tokens over all segments "
            "per sample for BERT dataset",
        )
        parser.add_argument(
            "--truncate-sequence",
            action="store_true",
            default=False,
            help="Truncate sequence to max_sequence_length",
        )
        parser.add_argument(
            "--mask-prob",
            default=0.15,
            type=float,
            help="probability of replacing a token with mask",
        )
        parser.add_argument(
            "--leave-unmasked-prob",
            default=0.1,
            type=float,
            help="probability that a masked token is unmasked",
        )
        parser.add_argument(
            "--random-token-prob",
            default=0.1,
            type=float,
            help="probability of replacing a token with a random token",
        )
        parser.add_argument(
            "--freq-weighted-replacement",
            action="store_true",
            help="sample random replacement words based on word frequencies",
        )
        parser.add_argument(
            "--mask-whole-words",
            default=False,
            action="store_true",
            help="mask whole words; you may also want to set --bpe",
        )

    def __init__(self, args, bpe, dictionary, label_dictionary):

        super().__init__(args)

        self.bpe = bpe
        self.dictionary = dictionary
        self._label_dictionary = label_dictionary

        self.mask_idx = dictionary.add_symbol(args.mask_token, verbose=True)
        self.init_idx = dictionary.add_symbol(args.init_token, verbose=True)
        self.separator_idx = dictionary.add_symbol(args.separator_token, verbose=True)
        self.surround_idx = dictionary.add_symbol(args.surround_token, verbose=True)
        self.def_sep_idx = dictionary.add_symbol(args.def_sep_token, verbose=True)

    @classmethod
    def setup_task(cls, args, **kwargs):
        assert (
            args.criterion == "masked_lm_prediction"
        ), "Must set criterion=masked_lm_prediction"

        # max_positions is used to determine the number of positional embeddings
        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        assert args.max_positions >= args.tokens_per_sample

        dictionary, bpe = load_dictionary(args, True, True, "dict.txt")
        print("| [input] dictionary: {} types".format(len(dictionary)))

        paths = args.data.split(os.pathsep)
        label_dictionary = Dictionary.load(os.path.join(paths[0], "label_dict.txt"))
        print("| [label] dictionary: {} types".format(len(label_dictionary)))

        return cls(args, bpe, dictionary, label_dictionary)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):

        Synset.set_collection_from_jsonl(
            os.path.join(self.args.data, "info.jsonl"), self.source_dictionary, self.bpe
        )

        data_path = os.path.join(self.args.data, split + ".jsonl")

        relations = Relation.load_collection_from_jsonl(
            data_path, self.source_dictionary, self.bpe
        )

        print(f"| Load {len(relations)} definition pairs")

        labels = [relation.relation for relation in relations]

        dataset = DefinitionPairDataset(
            relations,
            self.dictionary.eos(),
            self.surround_idx,
            self.def_sep_idx,
            self.init_idx,
            self.separator_idx,
            return_types=False,
        )

        type_dataset = DefinitionPairDataset(
            relations,
            self.dictionary.eos(),
            self.surround_idx,
            self.def_sep_idx,
            self.init_idx,
            self.separator_idx,
            return_types=True,
        )

        # create masked input and targets
        if self.args.mask_whole_words:
            bpe = self.bpe
            if bpe is not None:

                def is_beginning_of_word(i):
                    if i < self.source_dictionary.nspecial:
                        # special elements are always considered beginnings
                        return True
                    tok = self.source_dictionary[i]
                    if tok.startswith("madeupword"):
                        return True
                    try:
                        return bpe.is_beginning_of_word(tok)
                    except ValueError:
                        return True

                mask_whole_words = torch.ByteTensor(
                    list(map(is_beginning_of_word, range(len(self.source_dictionary))))
                )
        else:
            mask_whole_words = None

        src_dataset, tgt_dataset = MaskTokenExceptDataset.apply_mask(
            dataset,
            self.source_dictionary,
            pad_idx=self.source_dictionary.pad(),
            mask_idx=self.mask_idx,
            seed=self.args.seed,
            mask_prob=self.args.mask_prob,
            leave_unmasked_prob=self.args.leave_unmasked_prob,
            random_token_prob=self.args.random_token_prob,
            freq_weighted_replacement=self.args.freq_weighted_replacement,
            mask_whole_words=mask_whole_words,
            keep_words=[
                self.surround_idx,
                self.init_idx,
                self.separator_idx,
                self.def_sep_idx,
                self.dictionary.eos(),
            ],
        )

        if self.args.truncate_sequence:
            src_dataset = TruncateDataset(src_dataset, self.args.tokens_per_sample)
            tgt_dataset = TruncateDataset(tgt_dataset, self.args.tokens_per_sample)
            type_dataset = TruncateDataset(type_dataset, self.args.tokens_per_sample)

        label_dataset = OffsetTokensDataset(
            IndexedRawLabelDataset(labels, self.label_dictionary),
            offset=-self.label_dictionary.nspecial,
        )

        with data_utils.numpy_seed(self.args.seed + epoch):
            shuffle = np.random.permutation(len(src_dataset))

        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": PadDataset(
                    src_dataset, pad_idx=self.source_dictionary.pad(), left_pad=False
                ),
                "src_types": PadDataset(type_dataset, pad_idx=0, left_pad=False),
                "src_lengths": NumelDataset(src_dataset, reduce=False),
            },
            "target": PadDataset(
                tgt_dataset, pad_idx=self.target_dictionary.pad(), left_pad=False
            ),
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_dataset, reduce=True),
            "target_label": label_dataset,
        }

        nested_dataset = NestedDictionaryDataset(dataset, sizes=[src_dataset.sizes])

        dataset = SortDataset(nested_dataset, sort_order=[shuffle, src_dataset.sizes])

        print(f"| Loaded {split} with {len(dataset)} samples")
        self.datasets[split] = dataset

    def build_model(self, args):
        from fairseq import models

        model = models.build_model(args, self)

        model.register_classification_head(
            "sentence_classification_head",
            num_classes=len(self.label_dictionary) - self.label_dictionary.nspecial,
        )

        apply_bert_init(model.classification_heads["sentence_classification_head"])

        return model

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    @property
    def label_dictionary(self):
        return self._label_dictionary
