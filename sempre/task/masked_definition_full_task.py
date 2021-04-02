"""
this extracts word-definitions from wordnet and use all of them in training
the validation is using bookwiki with masked_lm but it can be disabled

dict.txt is the one provided with roberta checkpoints
content of label_dict.txt:
hyponyms 329086
hypernyms 329079
synonyms 315984
member_holonyms 60458
member_meronyms 60458
derivationally_related_forms 55296
similar_tos 50736
part_meronyms 37843
part_holonyms 37843
instance_hyponyms 34848
instance_hypernyms 34848
topic_domains 22645
region_domains 8582
verb_groups 8156
antonyms 7977
pertainyms 7746
also_sees 4585
usage_domains 4504
attributes 3402
substance_meronyms 2673
substance_holonyms 2673
entailments 2336
causes 1038
related_tos 0
not_resovled 0
madeupword0000 0
madeupword0001 0
madeupword0002 0
"""
import os
from argparse import Namespace
from functools import lru_cache

import numpy as np
import torch

from fairseq.data import (
    BaseWrapperDataset,
    Dictionary,
    FairseqDataset,
    IdDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    OffsetTokensDataset,
    PadDataset,
    PrependTokenDataset,
    SortDataset,
    TokenBlockDataset,
    TruncateDataset,
    data_utils,
    encoders,
)
from fairseq.meters import AverageMeter, TimeMeter
from fairseq.progress_bar import build_progress_bar
from fairseq.tasks import FairseqTask, register_task


from ..data.indexed_raw_label_dataset import IndexedRawLabelDataset
from ..data.mask_token_except_dataset import MaskTokenExceptDataset
from ..utils import apply_bert_init, load_dictionary, none_or_str


try:
    from nltk.corpus import wordnet as wn
except ImportError as e:
    print("Needs nltk package to build WordNet data")
    raise e

# pylint: disable=line-too-long
# fmt: off
RELATIONS = {
    "synonyms": lambda x: x.synset().lemmas(),                                                                                                   # same as,       shut out        > shut
    "antonyms": lambda x: x.antonyms(),                                                                                                          # opposite of,   happy           > unhappy
    "hypernyms": lambda x: x.hypernyms() + [lemma for synset in x.synset().hypernyms() for lemma in synset.lemmas()],                            # more general,  project         > show
    "instance_hypernyms": lambda x: x.instance_hypernyms() + [lemma for synset in x.synset().instance_hypernyms() for lemma in synset.lemmas()], # is a,          Albert Einstein > physicist       # noqa
    "hyponyms": lambda x: x.hyponyms() + [lemma for synset in x.synset().hyponyms() for lemma in synset.lemmas()],                               # more specific, show            > project
    "instance_hyponyms": lambda x: x.instance_hyponyms() + [lemma for synset in x.synset().instance_hyponyms() for lemma in synset.lemmas()],    # examples,      physicist       > Albert Einstein
    "member_holonyms": lambda x: x.member_holonyms() + [lemma for synset in x.synset().member_holonyms() for lemma in synset.lemmas()],          # member of,     faculty         > professor
    "substance_holonyms": lambda x: x.substance_holonyms() + [lemma for synset in x.synset().substance_holonyms() for lemma in synset.lemmas()], # used in,       oxygen          > water           # noqa
    "part_holonyms": lambda x: x.part_holonyms() + [lemma for synset in x.synset().part_holonyms() for lemma in synset.lemmas()],                # part of,       feather         > bird
    "member_meronyms": lambda x: x.member_meronyms() + [lemma for synset in x.synset().member_meronyms() for lemma in synset.lemmas()],          # has member,    professor       > faculty
    "substance_meronyms": lambda x: x.substance_meronyms() + [lemma for synset in x.synset().substance_meronyms() for lemma in synset.lemmas()], # contain,       water           > oxygen          # noqa
    "part_meronyms": lambda x: x.part_meronyms() + [lemma for synset in x.synset().part_meronyms() for lemma in synset.lemmas()],                # has part,      bird            > feather
    "topic_domains": lambda x: x.topic_domains() + [lemma for synset in x.synset().topic_domains() for lemma in synset.lemmas()],
    "region_domains": lambda x: x.region_domains() + [lemma for synset in x.synset().region_domains() for lemma in synset.lemmas()],
    "usage_domains": lambda x: x.usage_domains() + [lemma for synset in x.synset().usage_domains() for lemma in synset.lemmas()],
    "attributes": lambda x: x.attributes() + [lemma for synset in x.synset().attributes() for lemma in synset.lemmas()],                         # express,       heavy          <> weight
    "derivationally_related_forms": lambda x: x.derivationally_related_forms(),                                                                  # form,          snore           > snorer
    "entailments": lambda x: x.entailments() + [lemma for synset in x.synset().entailments() for lemma in synset.lemmas()],                      # entail         snore           > sleep
    "causes": lambda x: x.causes() + [lemma for synset in x.synset().causes() for lemma in synset.lemmas()],                                     # cause,         project         > appear
#    "also_sees": lambda x: x.also_sees() + [lemma for synset in x.synset().also_sees() for lemma in synset.lemmas()],  # noqa
    "verb_groups": lambda x: x.verb_groups() + [lemma for synset in x.synset().verb_groups() for lemma in synset.lemmas()],
    "similar_tos": lambda x: x.similar_tos() + [lemma for synset in x.synset().similar_tos() for lemma in synset.lemmas()],
    "pertainyms": lambda x: x.pertainyms(),                                                                                                      # pertaining to,
}
# fmt: on
# pylint: enable=line-too-long


class Lemma(object):

    _definition_by_synset_id = {}
    _examples_by_synset_id = {}
    _cached_lemmas = None

    def __init__(self, lemma_id, lemma_name, synset_id, relations):
        self.lemma_id = lemma_id
        self._lemma_name = lemma_name
        self._synset_id = synset_id
        self.relations = relations  # list of (str, Lemma) tuple

    @property
    def definition(self):
        return self._definition_by_synset_id[self._synset_id]

    @property
    def examples(self):
        return self._examples_by_synset_id[self._synset_id]

    @property
    def name(self):
        return self._lemma_name.replace("_", " ")

    def __repr__(self):
        return f"{self.name}: {self.definition}"

    def same_sense(self, other_lemma):
        return self._synset_id == other_lemma._synset_id

    @classmethod
    def build_lexicon(cls, pos=None):
        # pos can be n v a
        assert pos is None or pos in [
            "n",
            "v",
            "a",
        ], f"pos should be n v a, given {pos}"
        if cls._cached_lemmas:
            return cls._cached_lemmas

        if pos is None:
            pos_set = ["n", "v", "a", "r", "s"]
        elif pos == "a":
            pos_set = ["a", "r", "s"]
        else:
            pos_set = [pos]

        print(f"| loading synsets for {pos_set}")

        lemmas = {}
        cls._definition_by_synset_id = {}
        cls._examples_by_synset_id = {}

        def get_lemma_id(lemma):
            return f"{lemma.synset().name()}.{lemma.name()}"

        synsets = list(wn.all_synsets())
        for synset in synsets:
            if synset.pos() not in pos_set:
                continue
            synset_id = synset.name()
            cls._definition_by_synset_id[synset_id] = synset.definition()
            cls._examples_by_synset_id[synset_id] = synset.examples()

            for lemma in synset.lemmas():
                lemma_id = get_lemma_id(lemma)
                lemma_name = lemma.name()
                relations = [
                    (key, get_lemma_id(l))
                    for key, value in RELATIONS.items()
                    for l in value(lemma)
                    if l != lemma and l.synset().pos() in pos_set
                ]
                lemmas[lemma_id] = cls(lemma_id, lemma_name, synset_id, relations)

        for lemma in lemmas.values():
            lemma.relations = [
                (relation, lemmas[lemma_id]) for relation, lemma_id in lemma.relations
            ]
        lemmas = list(lemmas.values())
        cls._cached_lemmas = lemmas
        return lemmas


def generate_word_pairs(lemmas, epoch=0, seed=1):

    starts = []
    ends = []
    labels = []

    lemma_to_index = {lemma: idx for idx, lemma in enumerate(lemmas)}

    with data_utils.numpy_seed(seed, epoch):
        for lemma in lemmas:
            idx = lemma_to_index[lemma]
            for rel, other_lemma in lemma.relations:
                if other_lemma not in lemma_to_index:
                    continue
                starts.append(idx)
                ends.append(lemma_to_index[other_lemma])
                labels.append(rel)

    return [
        {"start": start, "end": end, "relation": label}
        for start, end, label in zip(starts, ends, labels)
    ]


class WordDefinitionDataset(FairseqDataset):
    """
    A dataset reads wordnet, concatenates word and definition, and binarizes it in memory at instantiation.
    """

    def __init__(
        self,
        lemmas,
        dictionary,
        bpe=None,
        surround_token="<sur>",
        separator_token=":",
        no_lemma=False,
        no_definition=False,
    ):
        super().__init__()
        self.tokens_list = []
        self.lines = []
        self.sizes = []
        # these two tokens are in different space from gpt2bpe tokens
        # no conflicts
        self.surround_token = surround_token
        self.separator_token = separator_token
        assert not (
            no_lemma and no_definition
        ), "no_lemma and no_definition cannot be both True"
        if no_lemma:
            self.build_data_no_lemma(lemmas, dictionary, bpe=bpe)
        elif no_definition:
            self.build_data_no_definition(lemmas, dictionary, bpe=bpe)
        else:
            self.build_data(lemmas, dictionary, bpe=bpe)
        self.no_lemma = no_lemma
        self._len = len(self.tokens_list)

    def build_data_no_lemma(self, lemmas, dictionary, bpe=None):
        # only definition
        # dep <sep>
        print("| processing data")
        pbar = build_progress_bar(
            Namespace(log_format="tqdm", log_interval=50, tensorboard_logdir=None),
            lemmas,
        )
        token_per_second = TimeMeter()
        token_per_example = AverageMeter()
        for lemma in pbar:
            line = lemma.definition
            self.lines.append(line)
            if bpe is not None:
                line = bpe.encode(line)
            tokens = dictionary.encode_line(
                line, add_if_not_exist=False, append_eos=True
            ).long()
            self.tokens_list.append(tokens)
            self.sizes.append(len(tokens))
            token_per_second.update(self.sizes[-1])
            token_per_example.update(self.sizes[-1])
            pbar.log(dict(wps=token_per_second, length=token_per_example))
        pbar.print(dict(wps=token_per_second, length=token_per_example))
        self.sizes = np.array(self.sizes)

    def build_data_no_definition(self, lemmas, dictionary, bpe=None):
        # only word
        # <sur> word <sur> <sep>
        template = f"{self.surround_token} {{}} {self.surround_token}"
        print("| processing data")
        pbar = build_progress_bar(
            Namespace(log_format="tqdm", log_interval=50, tensorboard_logdir=None),
            lemmas,
        )
        token_per_second = TimeMeter()
        token_per_example = AverageMeter()
        # bpe encode output space separated numbers
        for lemma in pbar:
            line = template.format(lemma.name)
            self.lines.append(line)
            if bpe is None:
                tokens = dictionary.encode_line(
                    line, add_if_not_exist=False, append_eos=True
                ).long()
            else:
                tokens = dictionary.encode_line(
                    template.format(bpe.encode(" " + lemma.name)),
                    add_if_not_exist=False,
                    append_eos=True,
                ).long()

            self.tokens_list.append(tokens)
            self.sizes.append(len(tokens))
            token_per_second.update(self.sizes[-1])
            token_per_example.update(self.sizes[-1])
            pbar.log(dict(wps=token_per_second, length=token_per_example))
        pbar.print(dict(wps=token_per_second, length=token_per_example))
        self.sizes = np.array(self.sizes)

    def build_data(self, lemmas, dictionary, bpe=None):
        # word with definition
        # <sur> word <sur> <:> def <sep>
        template = f"{self.surround_token} {{}} {self.surround_token} {self.separator_token} {{}}"
        print("| processing data")
        pbar = build_progress_bar(
            Namespace(log_format="tqdm", log_interval=50, tensorboard_logdir=None),
            lemmas,
        )
        token_per_second = TimeMeter()
        token_per_example = AverageMeter()
        # bpe encode output space separated numbers
        for lemma in pbar:
            line = template.format(lemma.name, lemma.definition)
            self.lines.append(template.format(lemma.name, lemma.definition))
            if bpe is None:
                tokens = dictionary.encode_line(
                    line, add_if_not_exist=False, append_eos=True
                ).long()
            else:
                tokens = dictionary.encode_line(
                    template.format(
                        bpe.encode(" " + lemma.name), bpe.encode(" " + lemma.definition)
                    ),
                    add_if_not_exist=False,
                    append_eos=True,
                ).long()

            self.tokens_list.append(tokens)
            self.sizes.append(len(tokens))
            token_per_second.update(self.sizes[-1])
            token_per_example.update(self.sizes[-1])
            pbar.log(dict(wps=token_per_second, length=token_per_example))
        pbar.print(dict(wps=token_per_second, length=token_per_example))
        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self._len:
            raise IndexError("index out of range")

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def get_string(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self._len

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]


class WordDefinitionPairDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        word_index_pairs,
        init_token_idx=None,
        sep_token_idx=None,
        return_types=False,
    ):
        super().__init__(dataset)
        self.word_index_pairs = word_index_pairs
        self.init_token_idx = init_token_idx
        self.sep_token_idx = sep_token_idx

        additional_token = len(
            [idx for idx in [init_token_idx, sep_token_idx] if idx is not None]
        )

        self._sizes = np.array(
            [
                dataset.size(a) + dataset.size(b) + additional_token
                for (a, b) in word_index_pairs
            ]
        )
        self._size = len(word_index_pairs)
        self.return_types = return_types

    def __getitem__(self, idx):
        # <cls> ... <sep> ...
        # -> <cls> <sur> w1 <sur> : d1 <sep> <sep> <sur> w2 <sur> : d2 <sep>
        pair = self.word_index_pairs[idx]
        item0 = self.dataset[pair[0]]
        item1 = self.dataset[pair[1]]

        if self.return_types:
            part0_length = len(item0) + (1 if self.init_token_idx is not None else 0)
            part1_length = len(item1) + (1 if self.sep_token_idx is not None else 0)
            return item0.new([0] * part0_length + [1] * part1_length)

        parts = []
        if self.init_token_idx is not None:
            parts.append(item0.new([self.init_token_idx]))
        parts.append(item0)
        if self.sep_token_idx is not None:
            parts.append(item0.new([self.sep_token_idx]))
        parts.append(item1)

        return torch.cat(parts)

    def __len__(self):
        return self._size

    @property
    def sizes(self):
        return self._sizes

    def num_tokens(self, index):
        return self._sizes[index]

    def size(self, index):
        return self._sizes[index]


@register_task("masked_def_lm")
class MaskedDefLMTask(FairseqTask):
    """Task for fine-tuning masked language models with masked word-definition pairs."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner",
        )
        parser.add_argument("--separator-token", type=none_or_str, default="</s>")
        parser.add_argument("--init-token", type=str, default="<s>")
        parser.add_argument("--surround-token", type=str, default="<sur>")
        parser.add_argument("--def-sep-token", type=str, default=":")
        parser.add_argument("--mask-token", type=str, default="<mask>")
        parser.add_argument(
            "--no-relation-prediction", action="store_true", default=False
        )
        parser.add_argument(
            "--no-lemma",
            action="store_true",
            default=False,
            help="do not add lemmas to the definitions",
        )
        parser.add_argument("--no-definition", action="store_true", default=False)
        parser.add_argument(
            "--no-masking-target-word", action="store_true", default=False
        )
        parser.add_argument(
            "--no-masking-definition", action="store_true", default=False
        )
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
        parser.add_argument(
            "--inspect-data",
            action="store_true",
            help="interactively inspect processed sentences",
        )

    def __init__(self, args, bpe, dictionary, label_dictionary, padding_factor=8):
        super().__init__(args)
        self.bpe = bpe
        self.dictionary = dictionary
        self._label_dictionary = label_dictionary
        self.seed = args.seed

        # add mask token
        self.mask_idx = dictionary.add_symbol(args.mask_token, verbose=True)

        # add other tokens
        self.init_idx = dictionary.add_symbol(args.init_token, verbose=True)
        self.separator_idx = dictionary.add_symbol(args.separator_token, verbose=True)
        self.surround_idx = dictionary.add_symbol(args.surround_token, verbose=True)
        self.def_sep_idx = dictionary.add_symbol(args.def_sep_token, verbose=True)

        print(
            f"| init {args.init_token} ({self.init_idx}) "
            f"| sep {args.separator_token} ({self.separator_idx}) "
            f"| sur {args.surround_token} ({self.surround_idx}) "
            f"| def_sep {args.def_sep_token} ({self.def_sep_idx})"
        )

        # lets pad embeddings indirectly
        if hasattr(self.dictionary, "pad_length_to_multiples"):
            self.dictionary.pad_length_to_multiples(
                padding_factor=padding_factor, verbose=True
            )

        # gpt2 dictionary is padded
        # then the <mask> token
        # our tokens
        # padded again
        # all those additional will be removed from vocabulary in fine-tuning

        self.word_definition_dataset = None
        self.lemmas = None

    @classmethod
    def setup_task(cls, args, **kwargs):
        assert (
            args.criterion == "masked_lm_prediction"
        ), "Must set criterion=masked_lm_prediction"

        # this will reflect in printed args
        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        assert args.max_positions >= args.tokens_per_sample
        # learned positional embedding is offset by padding_idx which is 1
        # num_positional_embedings = args.max_positions + 2
        # num_positional_embedings = (
        #     (num_positional_embedings + padding_factor - 1) // padding_factor
        # ) * padding_factor
        # args.max_positions = num_positional_embedings - 2
        # if max_positions is enlarged, we can actually take in more tokens per sample
        # is it helpful ?

        dictionary, bpe = load_dictionary(args, True, True, "dict.txt")
        print("| [input] dictionary: {} types".format(len(dictionary)))

        paths = args.data.split(os.pathsep)
        label_dict = Dictionary.load(os.path.join(paths[0], "label_dict.txt"))
        print("| [label] dictionary: {} types".format(len(label_dict)))

        return cls(args, bpe, dictionary, label_dict)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        if split.startswith("train"):
            self.load_mdef_dataset(split, epoch)
        else:
            self.load_mlm_dataset(split, epoch, combine)
        return self.datasets[split]

    def load_mdef_dataset(self, split, epoch=0):
        if self.word_definition_dataset is None:
            self.lemmas = Lemma.build_lexicon()
            self.word_definition_dataset = WordDefinitionDataset(
                self.lemmas,
                self.source_dictionary,
                bpe=self.bpe,
                surround_token=self.args.surround_token,
                separator_token=self.args.def_sep_token,
                no_lemma=self.args.no_lemma,
                no_definition=self.args.no_definition,
            )

        word_pairs = generate_word_pairs(self.lemmas, epoch=epoch, seed=self.args.seed,)

        print(f"| Sampled {len(word_pairs)} definition pairs")

        word_index_pairs = [(item["start"], item["end"]) for item in word_pairs]

        dataset = WordDefinitionPairDataset(
            self.word_definition_dataset,
            word_index_pairs,
            self.init_idx,
            self.separator_idx,
            return_types=False,
        )

        type_dataset = WordDefinitionPairDataset(
            self.word_definition_dataset,
            word_index_pairs,
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

        # to determine if a token belongs to a word or a definition
        # we use the special tokens, surround

        if self.surround_idx is not None:
            if self.args.no_masking_definition:

                def allow_function(x):
                    # tokens in the surrond_idx are target words
                    split_points = np.where(x == self.surround_idx)[0]
                    assert len(split_points) in [
                        2,
                        4,
                    ], f"surround_idx not matched {split_points}"

                    can_be_masked = np.arange(split_points[0] + 1, split_points[1])

                    if len(split_points) == 4:
                        can_be_masked = np.concatenate(
                            [
                                can_be_masked,
                                np.arange(split_points[2] + 1, split_points[3]),
                            ]
                        )
                    return can_be_masked

            elif self.args.no_masking_target_word:

                def allow_function(x):
                    # tokens outside the surrond_idx are defintions
                    split_points = np.where(x == self.surround_idx)[0]
                    assert len(split_points) in [
                        2,
                        4,
                    ], f"surround_idx not matched {split_points}"
                    end = len(x)
                    if len(split_points) == 2:
                        can_be_masked = np.arange(split_points[1] + 1, end)
                    else:
                        can_be_masked = np.concatenate(
                            [
                                np.arange(split_points[1] + 1, split_points[2]),
                                np.arange(split_points[3] + 1, end),
                            ]
                        )
                    return can_be_masked

            else:
                allow_function = None

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
                x
                for x in [
                    self.surround_idx,
                    self.init_idx,
                    self.separator_idx,
                    self.def_sep_idx,
                ]
                if x is not None
            ],
            allow_function=allow_function,
        )

        if getattr(self.args, "inspect_data", False):

            def decode(token_t):
                token_l = token_t.tolist()
                token_bpe = [self.source_dictionary[t] for t in token_l]
                string = " ".join(token_bpe)
                print(string)

            while True:
                ind = int(input("index> ").strip())
                print("src: ", end="")
                decode(src_dataset[ind])
                print("tgt: ", end="")
                decode(tgt_dataset[ind])
                print()

        if self.args.truncate_sequence:
            src_dataset = TruncateDataset(src_dataset, self.args.tokens_per_sample)
            tgt_dataset = TruncateDataset(tgt_dataset, self.args.tokens_per_sample)
            type_dataset = TruncateDataset(type_dataset, self.args.tokens_per_sample)

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
        }

        if not self.args.no_relation_prediction:
            labels = [item["relation"] for item in word_pairs]

            label_dataset = OffsetTokensDataset(
                IndexedRawLabelDataset(labels, self.label_dictionary),
                offset=-self.label_dictionary.nspecial,
            )
            dataset["target_label"] = label_dataset

        nested_dataset = NestedDictionaryDataset(dataset, sizes=[src_dataset.sizes])

        dataset = SortDataset(nested_dataset, sort_order=[shuffle, src_dataset.sizes])

        print(f"| Loaded {split} with {len(dataset)} samples")
        self.datasets[split] = dataset

    def load_mlm_dataset(self, split, epoch=0, combine=False):
        """Load a given preprocessed text dataset split.

        Args:
            split (str): name of the split (e.g., valid, test)
        """

        paths = self.args.data.split(os.pathsep)

        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]
        split_path = os.path.join(data_path, split)

        dataset = data_utils.load_indexed_dataset(
            split_path, self.source_dictionary, self.args.dataset_impl, combine=combine
        )
        if dataset is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path)
            )

        # create continuous blocks of tokens
        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.args.tokens_per_sample - 1,  # one less for <s>
            pad=self.source_dictionary.pad(),
            eos=self.source_dictionary.eos(),
            break_mode="complete",
        )
        print("| Loaded {} batches from: {}".format(len(dataset), split_path))

        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())

        # create masked input and targets
        if self.args.mask_whole_words:
            bpe = encoders.build_bpe(self.args)
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
        )

        with data_utils.numpy_seed(self.args.seed + epoch):
            shuffle = np.random.permutation(len(src_dataset))

        self.datasets[split] = SortDataset(
            NestedDictionaryDataset(
                {
                    "id": IdDataset(),
                    "net_input": {
                        "src_tokens": PadDataset(
                            src_dataset,
                            pad_idx=self.source_dictionary.pad(),
                            left_pad=False,
                        ),
                        "src_lengths": NumelDataset(src_dataset, reduce=False),
                    },
                    "target": PadDataset(
                        tgt_dataset,
                        pad_idx=self.source_dictionary.pad(),
                        left_pad=False,
                    ),
                    "nsentences": NumSamplesDataset(),
                    "ntokens": NumelDataset(src_dataset, reduce=True),
                },
                sizes=[src_dataset.sizes],
            ),
            sort_order=[shuffle, src_dataset.sizes],
        )

        print(f"| Loaded {split} with {len(dataset)} samples")

    def build_dataset_for_inference(self, src_tokens, src_lengths, sort=True):
        src_dataset = PadDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                self.args.tokens_per_sample - 1,  # one less for <s>
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            ),
            pad_idx=self.source_dictionary.pad(),
            left_pad=False,
        )
        src_dataset = PrependTokenDataset(src_dataset, self.source_dictionary.bos())
        src_dataset = NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": src_dataset,
                    "src_lengths": NumelDataset(src_dataset, reduce=False),
                },
            },
            sizes=src_lengths,
        )
        if sort:
            src_dataset = SortDataset(src_dataset, sort_order=[src_lengths])
        return src_dataset

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


@register_task("masked_simple_def_lm")
class MaskedSimDefLMTask(MaskedDefLMTask):
    """Task for fine-tuning masked language models with masked word-definitions."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        MaskedDefLMTask.add_args(parser)

        parser.add_argument(
            "--pos",
            default=None,
            help="allowed pos tag for words, a for adj. adv., n for noun, v for verb, x for all",
        )

        # parser.add_argument(
        #     "--no-lemma",
        #     action="store_true",
        #     default=False,
        #     help="do not add lemmas to the definitions",
        # )

    def __init__(self, args, bpe, dictionary, padding_factor=8):
        super().__init__(args, bpe, dictionary, None, padding_factor)

    @classmethod
    def setup_task(cls, args, **kwargs):
        assert args.criterion == "masked_lm", "Must set criterion=masked_lm"

        # this will reflect in printed args
        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        assert args.max_positions >= args.tokens_per_sample

        if args.pos == "x":
            args.pos = None

        dictionary, bpe = load_dictionary(args, True, True, "dict.txt")
        print("| [input] dictionary: {} types".format(len(dictionary)))

        return cls(args, bpe, dictionary)

    def load_mdef_dataset(self, split, epoch=0):
        if self.word_definition_dataset is None:
            self.lemmas = Lemma.build_lexicon(pos=self.args.pos)
            self.word_definition_dataset = WordDefinitionDataset(
                self.lemmas,
                self.source_dictionary,
                bpe=self.bpe,
                surround_token=self.args.surround_token,
                separator_token=self.args.def_sep_token,
                no_lemma=self.args.no_lemma,
            )

        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        dataset = PrependTokenDataset(
            self.word_definition_dataset, self.source_dictionary.bos()
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
                x
                for x in [self.surround_idx, self.init_idx, self.separator_idx]
                if x is not None
            ],
        )

        if self.args.truncate_sequence:
            src_dataset = TruncateDataset(src_dataset, self.args.tokens_per_sample)
            tgt_dataset = TruncateDataset(tgt_dataset, self.args.tokens_per_sample)

        with data_utils.numpy_seed(self.args.seed + epoch):
            shuffle = np.random.permutation(len(src_dataset))

        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": PadDataset(
                    src_dataset, pad_idx=self.source_dictionary.pad(), left_pad=False
                ),
                "src_lengths": NumelDataset(src_dataset, reduce=False),
            },
            "target": PadDataset(
                tgt_dataset, pad_idx=self.source_dictionary.pad(), left_pad=False
            ),
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_dataset, reduce=True),
        }

        nested_dataset = NestedDictionaryDataset(dataset, sizes=[src_dataset.sizes])

        dataset = SortDataset(nested_dataset, sort_order=[shuffle, src_dataset.sizes])

        print(f"| Loaded {split} with {len(dataset)} samples")
        self.datasets[split] = dataset

    def build_model(self, args):
        from fairseq import models

        model = models.build_model(args, self)

        return model

    @property
    def label_dictionary(self):
        raise NotImplementedError
