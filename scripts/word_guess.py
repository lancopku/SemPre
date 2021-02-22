"""
word guess with definitions for words in oxford3000
definitions may be from wordnet or oald

for word guess with wordnet definitions, as we use wordnet definitions as training sources,
we randomly split the oxford3000 and prevent 1000 of them from being trained (see the masked
definition new task and preprocess_data.py to see how it's actually done)
"""

import argparse
import os
import functools
import numpy as np
import json

import torch
from nltk.corpus import wordnet as wn
from tqdm import tqdm

from fairseq.data import data_utils, encoders
from fairseq.models.roberta import RobertaModel


TOP_WORDNET_SYNSET = 2
DEFAULT_DEF_SRC = "wndef"

TEMPLATES = [" {w} means {d}", " the definition of {w} is {d}", " {w} : {d}"]

NOUN_TEMPLATES = [" {w} is {d}", " {w} are {d}"]  # nouns only  # nouns only


TEMPLATES = ["{w} means {d}", "The definition of {w} is {d}", "{w} : {d}"]

NOUN_TEMPLATES = ["{w} is {d}", "{w} are {d}"]  # nouns only  # nouns only


def get_args():
    parser = argparse.ArgumentParser()

    # fmt: off
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-name", type=str, default="checkpoint_best.pt")
    parser.add_argument("--max-sentences", type=int, default=64)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument("--user-dir", default=None)
    parser.add_argument("--cpu", action='store_true')
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--no-form-variation", action='store_true', default=False)
    parser.add_argument("data", type=str)
    parser.add_argument('--def-src', type=str, default=DEFAULT_DEF_SRC)
    # fmt: on
    args = parser.parse_args()

    assert os.path.isfile(os.path.join(args.checkpoint_dir, args.checkpoint_name))
    assert os.path.isfile(args.data)
    assert args.def_src == DEFAULT_DEF_SRC or os.path.isfile(args.def_src)

    if args.save is None:

        def get_name(path):
            return os.path.basename(os.path.splitext(path)[0])

        form = "-base_form" if args.no_form_variation else "-full_form"

        args.save = os.path.join(
            os.path.dirname(args.data),
            f"wg-{get_name(args.data)}-{get_name(args.def_src)}{form}-{get_name(args.checkpoint_name)}.txt",
        )

    return args


def from_pretrained(args):
    # this is not a roberta model but a roberta hub interface
    kwargs = dict(
        checkpoint_file=args.checkpoint_name,
        data_name_or_path=".",
        gpt2_encoder_json=encoders.gpt2_bpe.DEFAULT_ENCODER_JSON,
        gpt2_vocab_bpe=encoders.gpt2_bpe.DEFAULT_VOCAB_BPE,
    )
    if args.user_dir is not None:
        kwargs["user_dir"] = args.user_dir
    return RobertaModel.from_pretrained(args.checkpoint_dir, **kwargs)


@functools.lru_cache()
def read_def_src(filepath):
    # currently, only for oxdef.jsonl. due to various reasons, we are unable to distribute this. please
    # contact the authors for more info.
    definitions = {}
    with open(filepath, "r", encoding="utf8") as f:
        for line in f:
            data = json.loads(line)
            definitions[data["word"]] = data["definitions"]
    return definitions


class Example:
    def __init__(self, word, pos, definition, template, sentence, tensor, target):

        self.word = word
        self.pos = pos
        self.definition = definition
        self.template = template
        self.sentence = sentence
        self.tensor = tensor
        self.target = target

        self.prediction = None
        self.rank = None

    def __str__(self):
        return (
            f"Example(word={self.word}, pos={self.pos}, definition={self.definition})"
        )

    def __len__(self):
        return self.tensor.size(0)


def generate_dataset(
    samples, dictionary, bpe, def_src=DEFAULT_DEF_SRC, variation=False
):

    dataset = []  # (word, pos, definition, template, tensor, target)

    def make_example(word, form, definition, template):
        form_bpe = bpe.encode(form).split()
        target_len = len(form_bpe)
        if target_len > 1:
            return None

        sentence = template.format(d=definition, w="{w}")
        offset = sentence.index("{w}")

        form_has_leading_space = form[0].isspace()
        template_has_leading_space = offset > 0 and sentence[offset - 1].isspace()

        if form_has_leading_space and template_has_leading_space:
            split_token = " {w}"
        else:
            split_token = "{w}"

        prefix, suffix = sentence.split(split_token)

        prefix_bpe, suffix_bpe = bpe.encode(prefix).split(), bpe.encode(suffix).split()
        offset = len(prefix_bpe)

        sent_bpe = prefix_bpe + form_bpe + suffix_bpe
        indices = (
            [dictionary.bos()]
            + [dictionary.index(token) for token in sent_bpe]
            + [dictionary.eos()]
        )
        offset += 1

        target_index = indices[offset]
        indices[offset] = dictionary.index("<mask>")
        return Example(
            word,
            pos,
            definition,
            template,
            sentence,
            torch.tensor(indices),
            target_index,
        )

    def get_definitions(word, pos):
        definitions = []
        if def_src == DEFAULT_DEF_SRC:
            synsets = wn.synsets(word, pos=pos)[:TOP_WORDNET_SYNSET]  # top 2 synset
            for synset in synsets:
                definitions.append(synset.definition())
        else:
            src = read_def_src(def_src)
            if word in src:
                definitions = src[word]
            else:
                print(f"word {word} def not found")
        return definitions

    for word, pos, _ in samples:

        for definition in get_definitions(word, pos):

            # synsets = wn.synsets(word, pos=pos)[:2]
            # for synset in synsets:
            #     definition = synset.definition()

            templates = TEMPLATES
            if pos.startswith("n"):
                templates = templates + NOUN_TEMPLATES

            for template in templates:
                # the base form: leading space, lowercased
                ex = make_example(word, " " + word, definition, template)
                if ex is not None:
                    dataset.append(ex)

                if variation:
                    # the following are other forms
                    ex = make_example(word, word, definition, template)
                    if ex is not None:
                        dataset.append(ex)

                    ex = make_example(
                        word, word[0].upper() + word[1:], definition, template
                    )
                    if ex is not None:
                        dataset.append(ex)

                    ex = make_example(
                        word, " " + word[0].upper() + word[1:], definition, template
                    )
                    if ex is not None:
                        dataset.append(ex)

    print(f"| Dataset loaded: {len(dataset)} examples")
    return dataset


def get_batch_iterator(
    dataset, max_tokens=None, max_sentences=None, required_batch_size_multiple=4
):
    indices = np.arange(len(dataset), dtype=np.int64)
    lengths = np.array([len(d) for d in dataset])

    def num_tokens(index):
        return lengths[index]

    batch_sampler = data_utils.batch_by_size(
        indices,
        num_tokens,
        max_tokens=max_tokens,
        max_sentences=max_sentences,
        required_batch_size_multiple=required_batch_size_multiple,
    )

    return batch_sampler


def read_oxford3000(path):
    sample_set = set()
    samples = []
    POS_DICT = {"v.": "v", "n.": "n", "adj.": "a", "adv.": "r"}
    with open(path, encoding="utf8") as f:
        for line in f:
            parts = line.strip("\n").split("\t")

            if not eval(parts[-2]):
                continue
            sample = (parts[0], POS_DICT.get(parts[3], None), parts[-1])
            if sample in sample_set:
                continue
            else:
                sample_set.add(sample)

            samples.append(sample)
    print(f"| Load {len(samples)} examples")
    return samples


def idx_to_token(idx, roberta):
    dict_decoded = roberta.task.source_dictionary[idx]
    bpe_decoded = roberta.bpe.bpe.decoder.get(
        int(dict_decoded) if dict_decoded.isdigit() else dict_decoded, dict_decoded
    )
    byte_decoded = bytearray(
        [roberta.bpe.bpe.byte_decoder[c] for c in bpe_decoded]
    ).decode("utf-8", errors="replace")
    return byte_decoded


def main():
    args = get_args()
    print(args)
    if args.save is not None and os.path.isfile(args.save):
        print("| Warning: remove existing save file")
        os.remove(args.save)

    roberta = from_pretrained(args)
    torch.set_grad_enabled(False)
    roberta.float()
    roberta.eval()

    if not args.cpu:
        roberta.cuda()

    samples = read_oxford3000(args.data)
    dataset = generate_dataset(
        samples,
        roberta.task.source_dictionary,
        roberta.bpe,
        args.def_src,
        not args.no_form_variation,
    )
    # print()

    index_to_token = {
        i: idx_to_token(i, roberta) for i in range(len(roberta.task.source_dictionary))
    }
    print("| dictionary index to bpe token mapping built")

    # predicitions = [None] * len(dataset)
    # ranks = [None] * len(dataset)

    last_word = None
    last_definition = None

    for batch in tqdm(
        get_batch_iterator(
            dataset, max_tokens=args.max_tokens, max_sentences=args.max_sentences
        ),
        ascii=True,
        dynamic_ncols=True,
        leave=True,
    ):
        src_tokens = [dataset[idx].tensor for idx in batch]
        src_tokens = data_utils.collate_tokens(
            src_tokens, roberta.task.source_dictionary.pad()
        )
        targets = [dataset[idx].target for idx in batch]
        targets = torch.tensor(targets)
        if not args.cpu:
            src_tokens = src_tokens.cuda()
            targets = targets.cuda()

        hiddens, _ = roberta.model(
            src_tokens, features_only=True, return_all_hiddens=False
        )

        masked_tokens = src_tokens.eq(roberta.task.source_dictionary.index("<mask>"))
        logits = roberta.model.decoder.output_layer(hiddens, masked_tokens).view(
            src_tokens.size(0), -1
        )

        sorted_index = logits.argsort(dim=-1, descending=True)
        top_k = sorted_index[:, :10]
        ranks = (sorted_index == targets.unsqueeze(1)).nonzero()[:, 1]

        # logits = logits.tolist()
        ranks = ranks.tolist()
        top_k = top_k.tolist()

        for i, j in enumerate(batch):
            dataset[j].rank = ranks[i]
            dataset[j].prediction = top_k[i]
            # print(
            #     f"| {dataset[j].sentence} | {[index_to_token[k] for k in dataset[j].sentence.tolist()]} | {ranks[i]} | {[index_to_token[k] for k in top_k[i]]} |"
            # )
            if args.save is not None:
                with open(args.save, "a", encoding="utf8") as f:
                    # input_sentence = [
                    #     index_to_token[k] for k in dataset[j].tensor.tolist()
                    # ]
                    # input_sentence = (
                    #     input_sentence[0]
                    #     + " "
                    #     + "".join(input_sentence[1:-1])
                    #     + " "
                    #     + input_sentence[-1]
                    # )
                    input_sentence = "_".join(
                        index_to_token[k] for k in dataset[j].tensor.tolist()
                    )
                    target = index_to_token[dataset[j].target]
                    input_sentence = input_sentence.replace("<mask>", f"[{target}]")

                    if (
                        last_definition is not None
                        and dataset[j].definition != last_definition
                    ):
                        print("", file=f)
                    last_definition = dataset[j].definition
                    if last_word is not None and dataset[j].word != last_word:
                        print("", file=f)
                    last_word = dataset[j].word

                    print(
                        f"| {ranks[i]:5d} | {input_sentence} | {[index_to_token[k] for k in top_k[i]]} |",
                        file=f,
                    )
            # dataset[j].prediction = logits[i]

    # print()

    ex_syn_temp = {}
    for ex in dataset:
        key = (ex.word, ex.pos)
        if key not in ex_syn_temp:
            ex_syn_temp[key] = {}
        if ex.definition not in ex_syn_temp[key]:
            ex_syn_temp[key][ex.definition] = []
        ex_syn_temp[key][ex.definition].append(ex.rank)

    # average word
    # average definition
    # mininum form and templates
    mean_rank = np.mean(
        [
            np.mean([np.min(temp) for temp in syn.values()])
            for syn in ex_syn_temp.values()
        ]
    )
    print(f"| mean rank | all {mean_rank}")

    splits = {}

    for word, pos, split in samples:
        if split not in splits:
            splits[split] = []
        if (word, pos) in ex_syn_temp:
            splits[split].append(ex_syn_temp[(word, pos)])

    assert sum(map(len, splits.values())) == len(ex_syn_temp)

    split_mean_rank = {
        key: np.mean(
            [np.mean([np.min(temp) for temp in syn.values()]) for syn in value]
        )
        for key, value in splits.items()
    }

    for key, value in split_mean_rank.items():
        print(f"| mean rank | {key} {value}")

    # results = []
    # with torch.no_grad():
    #     for word, pos in tqdm(samples):
    #         ranks = validate_step(
    #             args,
    #             roberta.model,
    #             roberta.task.source_dictionary,
    #             roberta.bpe,
    #             word,
    #             pos,
    #         )
    #         results.append(ranks)

    # # results: [each example, each synset, each template]
    # mean_rank = np.mean(
    #     [np.mean([ranks.min() for rank in rank_list]) for rank_list in results]
    # )

    print("| done")


if __name__ == "__main__":
    main()
