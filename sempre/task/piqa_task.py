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
    SortDataset,
    TruncateDataset,
    data_utils,
)
from fairseq.tasks import FairseqTask, register_task

from ..utils import apply_bert_init, delete_lm_head, load_dictionary, none_or_str


@register_task("piqa")
class PIQATask(FairseqTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "data", metavar="DIR", help="path to data directory; we load <split>.jsonl"
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

        self.init_index = vocab.add_symbol(args.init_token, True)
        self.separator_index = vocab.add_symbol(args.separator_token, True)

        print(
            f"| init {args.init_token} ({self.init_index}) "
            f"| sep {args.separator_token} ({self.separator_index}) "
        )
        # just follow fariseq implementation on CQA
        print("| sep is not used in this task")

    @classmethod
    def setup_task(cls, args, **kwargs):
        assert (
            args.criterion == "sentence_ranking"
        ), "Must set --criterion=sentence_ranking"
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

        # need_types = "hf_bert" in self.args.arch

        src_tokens = [[], []]
        # if need_types:
        #     src_types = [[], []]
        src_lengths = [[], []]
        labels = []

        def binarize(sentence, append_eos=True):
            bpe = self.bpe.encode(sentence)
            tokens = [self.vocab.index(token) for token in bpe.split()]
            if append_eos:
                tokens = tokens + [self.vocab.eos()]
            return tokens

        init_seq = [self.init_index] if self.init_index is not None else []
        sep_seq = []

        with open(data_path, "r", encoding="utf8") as f:
            for line in f:
                data = json.loads(line)

                goal = binarize(data["goal"], True)

                for i, key in enumerate(["sol1", "sol2"]):
                    solution = data[key]
                    solution = binarize(solution, True)

                    # [cls] goal [sep] sol [sep]
                    tokens = init_seq + goal + sep_seq + solution
                    # if need_types:
                    #     types = [0] * (len(tokens) - len(solution)) + [1] * len(
                    #         solution
                    #     )

                    tokens = torch.tensor(tokens, dtype=torch.int64)
                    # if need_types:
                    #     types = torch.tensor(types, dtype=torch.int64)

                    src_tokens[i].append(tokens)
                    # if need_types:
                    #     src_types[i].append(types)

                if "label" in data:
                    labels.append(data["label"])

        for i in range(self.args.num_classes):
            src_lengths[i] = np.array([len(t) for t in src_tokens[i]])
            src_tokens[i] = ListDataset(src_tokens[i], src_lengths[i])
            # if need_types:
            #     src_types[i] = ListDataset(src_types[i], src_lengths[i])
            src_lengths[i] = ListDataset(src_lengths[i])

            if self.args.truncate_sequence:
                src_tokens[i] = TruncateDataset(
                    src_tokens[i], self.args.tokens_per_sample
                )
                # if need_types:
                #     src_types[i] = TruncateDataset(
                #         src_types[i], self.args.tokens_per_sample
                #     )

        if getattr(self.args, "inspect_data", True):

            def decode(token_t):
                token_l = token_t.tolist()
                token_bpe = [self.source_dictionary[t] for t in token_l]
                string = " ".join(token_bpe)
                print(string)

            while True:
                ind = int(input("index> ").strip())
                print("src1: ", end="")
                decode(src_tokens[0][ind])
                print("src2: ", end="")
                decode(src_tokens[1][ind])
                print()

        with data_utils.numpy_seed(self.args.seed + epoch):
            shuffle = np.random.permutation(len(src_tokens[0]))

        dataset = {
            "id": IdDataset(),
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_tokens[0], reduce=True),
        }

        for i in range(self.args.num_classes):
            dataset.update(
                {
                    f"net_input{i+1}": {
                        "src_tokens": PadDataset(
                            src_tokens[i],
                            pad_idx=self.source_dictionary.pad(),
                            left_pad=False,
                        ),
                        "src_lengths": src_lengths[i],
                    }
                }
            )
            # if need_types:
            #     dataset[f"net_input{i+1}"].update(
            #         src_types=PadDataset(src_types[i], pad_idx=0, left_pad=False)
            #     )

        if labels:
            dataset.update(target=RawLabelDataset(labels))

        nested_dataset = NestedDictionaryDataset(
            dataset,
            sizes=[np.maximum.reduce([src_token.sizes for src_token in src_tokens])],
        )

        if self.args.no_shuffle:
            dataset = nested_dataset
        else:
            dataset = SortDataset(nested_dataset, sort_order=[shuffle])

        print(f"| Loaded {split} with {len(dataset)} samples")
        self.datasets[split] = dataset
        return dataset

    def build_model(self, args):
        from fairseq import models

        model = models.build_model(args, self)

        # just to save gpu memory
        delete_lm_head(model)

        model.register_classification_head(
            getattr(args, "ranking_head_name", "sentence_classification_head"),
            num_classes=1,
        )
        apply_bert_init(
            model.classification_heads[
                getattr(args, "ranking_head_name", "sentence_classification_head")
            ]
        )

        return model

    @property
    def source_dictionary(self):
        return self.vocab

    @property
    def target_dictionary(self):
        return self.vocab  # stub to fool the fairseq_criterions
