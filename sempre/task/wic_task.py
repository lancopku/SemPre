import json
import os

import numpy as np
import torch
from fairseq.data import (
    IdDataset,
    ListDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PadDataset,
    RawLabelDataset,
    TruncateDataset,
)
from fairseq.models.roberta import RobertaClassificationHead
from fairseq.tasks import FairseqTask, register_task

from ..data.dynamic_dataset import DynamicShuffleDataset, DynamicShuffleMixin
from ..utils import apply_bert_init, delete_lm_head, load_dictionary, none_or_str


@register_task("wic")
class WiCTask(DynamicShuffleMixin, FairseqTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "data", metavar="FILE", help="path to data directory; we load <split>.jsonl"
        )

        parser.add_argument(
            "--init-token",
            type=none_or_str,
            default=None,
            help="add token at the beginning of each batch item",
        )
        parser.add_argument(
            "--separator-token",
            type=none_or_str,
            default=None,
            help="add separator token between inputs",
        )
        parser.add_argument("--no-shuffle", action="store_true", default=False)
        parser.add_argument(
            "--truncate-sequence",
            action="store_true",
            default=False,
            help="truncate sequence to max_positions",
        )
        parser.add_argument(
            "--inspect-data",
            action="store_true",
            help="interactively inspect processed sentences",
        )

    def __init__(self, args, vocab, bpe):
        super().__init__(args)

        self.vocab = vocab
        self.bpe = bpe

        self.init_index = self.vocab.add_symbol(args.init_token)
        self.separator_index = self.vocab.add_symbol(args.separator_token)

        print(
            f"| init {args.init_token} ({self.init_index}) "
            f"| sep {args.separator_token} ({self.separator_index}) "
        )

    @classmethod
    def setup_task(cls, args, **kwargs):
        assert args.criterion == "wic", "Must set --criterion=wic"

        args.num_classes = 2
        args.regression_target = False
        args.tokens_per_sample = args.max_positions

        vocab, bpe = load_dictionary(args, True, True, "dict.txt")
        print("| dictionary: {} types".format(len(vocab)))

        return cls(args, vocab, bpe)

    def load_dataset(
        self, split, epoch=0, combine=False, data_path=None, return_only=False, **kwargs
    ):
        if data_path is None:
            data_path = os.path.join(self.args.data, split + ".jsonl")
        if not os.path.exists(data_path):
            raise FileNotFoundError("Cannot find data: {}".format(data_path))

        src_tokens = []
        # src_types = []
        src_indices = [[], []]
        labels = []

        def binarize(sent, start, end, append_eos=True):

            # special treatment for capital fisrt letter
            # if len(self.bpe.encode(" " + sent[0].lower() + sent[1:]).split()) <= len(
            #     self.bpe.encode(sent).split()
            # ):
            #     sent = " " + sent[0].lower() + sent[1:]
            #     start += 1
            #     end += 1
            # gpt2bpe is fully convertible, so space is also encoded
            # however, the trained models are a bit too sensative to spaces
            # lets reindex the start position of the target word
            if start > 0 and sent[start - 1].isspace():
                start = start - 1
            prefix = self.bpe.encode(sent[:start]).split()
            # the prefix space is included in the target
            target = self.bpe.encode(sent[start:end]).split()
            suffix = self.bpe.encode(sent[end:]).split()
            sent_bpe = prefix + target + suffix
            prefix_len = len(prefix)
            target_len = len(target)

            tokens = [self.vocab.index(token) for token in sent_bpe]
            if append_eos:
                tokens = tokens + [self.source_dictionary.eos()]
            return tokens, prefix_len, target_len

        init_seq = [self.init_index] if self.init_index is not None else []
        sep_seq = [self.separator_index] if self.separator_index is not None else []

        with open(data_path, "r", encoding="utf8") as f:
            for line in f:
                # json dict contains:
                # word, sentence1, sentence2, idx, start1, start2, end1, end2, label (true, false)
                data = json.loads(line)

                tokens1, offset1, len1 = binarize(
                    data["sentence1"], data["start1"], data["end1"]
                )
                tokens2, offset2, len2 = binarize(
                    data["sentence2"], data["start2"], data["end2"]
                )

                # [cls] sent1 [eos] [eos] sent2 [eos]
                tokens = init_seq + tokens1 + sep_seq + tokens2

                # types = [0] * (len(tokens) - len(tokens2)) + [1] * (len(tokens2))

                word1_offset = len(init_seq) + offset1
                word2_offset = len(init_seq) + len(tokens1) + len(sep_seq) + offset2

                tokens = torch.tensor(tokens, dtype=torch.int64)
                # types = torch.tensor(types, dtype=torch.int64)

                src_tokens.append(tokens)
                # src_types.append(types)

                for i, (offset, l) in enumerate(
                    [[word1_offset, len1], [word2_offset, len2]]
                ):

                    indice = torch.tensor(
                        [j for j in range(offset, offset + l)], dtype=torch.int64
                    )

                    src_indices[i].append(indice)

                if "label" in data:
                    labels.append(1 if data["label"] else 0)

        if getattr(self.args, "inspect_data", False):

            def decode(token_t, indices_t):
                token_l = token_t.tolist()
                indices_l = [ind_t.tolist() for ind_t in indices_t]
                token_bpe = [self.source_dictionary[t] for t in token_l]
                string = " ".join(token_bpe)
                print("src:", string)
                for idx, indices in enumerate(indices_l):
                    print(f"tgt{idx+1}", [token_bpe[i] for i in indices])
                print("dec:", self.bpe.decode(string))
                print()

            while True:
                ind = int(input("index> ").strip())
                decode(src_tokens[ind], [src_indices[0][ind], src_indices[1][ind]])

        src_lengths = np.array([len(t) for t in src_tokens])
        src_tokens = ListDataset(src_tokens, src_lengths)
        # src_types = ListDataset(src_types, src_lengths)

        if self.args.truncate_sequence:
            src_tokens = TruncateDataset(src_tokens, self.args.tokens_per_sample)
            # src_types = TruncateDataset(src_types, self.args.tokens_per_sample)

            assert all(
                t.max().item() < self.args.tokens_per_sample for t in src_indices[1]
            ), "Error: will truncate target word"

        for i, indice in enumerate(src_indices):
            assert all(
                i > 0 for j in indice for i in j
            ), "word token index should not be 0"
            lengths = np.array([len(t) for t in indice])
            src_indices[i] = ListDataset(indice, lengths)

        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": PadDataset(
                    src_tokens, pad_idx=self.source_dictionary.pad(), left_pad=False
                ),
                "src_lengths": NumelDataset(src_tokens, reduce=False),
                # "src_types": PadDataset(src_types, pad_idx=0, left_pad=False),
                "src_ranges": {
                    "range1": PadDataset(src_indices[0], pad_idx=0, left_pad=False),
                    "range2": PadDataset(src_indices[1], pad_idx=0, left_pad=False),
                },
            },
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_tokens, reduce=True),
        }

        if labels:
            dataset.update(target_labels=RawLabelDataset(labels))

        nested_dataset = NestedDictionaryDataset(dataset, sizes=[src_tokens.sizes])

        if self.args.no_shuffle:
            dataset = nested_dataset
        else:
            # see scripts/train.py
            # load_dataset is only called once in training
            dataset = DynamicShuffleDataset(nested_dataset, self.args.seed)

        print(f"| Loaded {split} with {len(dataset)} samples")
        if not return_only:
            self.datasets[split] = dataset
        return dataset

    def build_model(self, args):
        from fairseq import models

        model = models.build_model(args, self)

        # just to save gpu memory
        delete_lm_head(model)

        # we input 3 token embeddings, which is handled
        # in wic_criterion
        model.classification_heads[
            getattr(args, "classification_head_name", "sentence_classification_head")
        ] = RobertaClassificationHead(
            args.encoder_embed_dim * 3,
            args.encoder_embed_dim,
            args.num_classes,
            args.pooler_activation_fn,
            args.pooler_dropout,
        )

        # the input dim is way larger, don't apply bert init
        # apply_bert_init(
        #     model.classification_heads[
        #         getattr(
        #             args, "classification_head_name", "sentence_classification_head"
        #         )
        #     ]
        # )

        return model

    @property
    def source_dictionary(self):
        return self.vocab

    @property
    def target_dictionary(self):
        # stub to fool the fairseq criterion, which uses vocab's padding index
        return self.vocab
