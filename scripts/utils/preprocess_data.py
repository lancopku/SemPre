"""
data preprocessing for training for word guess and testing for word guess
input: nltk wordnet, the oxford 3000.txt
output: wordnet.jsonl, oxford3000.tsv, oxford3000-wn1000.tsv, wn1000/{info,train,valid}.jsonl
"""
import argparse
import collections
import json
import random
import re

import tqdm
from fairseq.data import Dictionary
from fairseq.data.encoders.gpt2_bpe import GPT2BPE
from nltk.corpus import wordnet as wn

DICT_PATH = "dict.txt"  # this is the roberta dict


def is_in_wordnet(word, pos=None):
    if pos is not None:
        pos = pos.strip(".")
        if pos == "adj":
            pos = "a"
        elif pos == "adv":
            pos = "r"
        if pos not in ["n", "v", "a", "r"]:
            return False
    ss = wn.synsets(word, pos)
    if ss:
        return True
    return False


def get_antonyms(synset):
    antonym_synsets = set()
    # direct antonym
    for lemma in synset.lemmas():
        for antonym in lemma.antonyms():
            antonym_synsets.add(antonym.synset())
    # indirect antonym
    for similar_synset in synset.similar_tos():
        for lemma in similar_synset.lemmas():
            for antonym in lemma.antonyms():
                antonym_synsets.add(antonym.synset())
    return list(antonym_synsets)


def get_pertainyms(synset):
    pertainym_synsets = set()
    for lemma in synset.lemmas():
        for pertainym in lemma.pertainyms():
            pertainym_synsets.add(pertainym.synset())
    return list(pertainym_synsets)


def get_sister_terms(synset):
    return list(
        set(
            sister_term
            for hypernym in synset.hypernyms()
            for sister_term in hypernym.hyponyms()
        )
        - set([synset])
    )


# fmt: off
RELATIONS = {
    "antonyms": get_antonyms,                                  # opposite of,   happy           > unhappy
    "hypernyms": lambda x: x.hypernyms(),                      # more general,  project         > show
    "hyponyms": lambda x: x.hyponyms(),                        # more specific, show            > project
    "sister_terms": get_sister_terms,
    "member_holonyms": lambda x: x.member_holonyms(),          # member of,     faculty         > professor
    "substance_holonyms": lambda x: x.substance_holonyms() ,   # used in,       oxygen          > water
    "part_holonyms": lambda x: x.part_holonyms() ,             # part of,       feather         > bird
    "member_meronyms": lambda x: x.member_meronyms(),          # has member,    professor       > faculty
    "substance_meronyms": lambda x: x.substance_meronyms() ,   # contain,       water           > oxygen
    "part_meronyms": lambda x: x.part_meronyms() ,             # has part,      bird            > feather
    "attributes": lambda x: x.attributes() ,                   # express,       heavy          <> weight
    "entailments": lambda x: x.entailments() ,                 # entail         snore           > sleep
    "causes": lambda x: x.causes() ,                           # cause,         project         > appear
    "similar_tos": lambda x: x.similar_tos(),
    "pertainyms": get_pertainyms,                              # pertaining to,
}
# fmt: on

dictionary = Dictionary.load(DICT_PATH)
bpe = GPT2BPE(argparse.Namespace()).bpe


def bpe_encode(text, prepend_space=False):
    assert prepend_space in ["auto_len", "auto_cap", True, False]
    if prepend_space == "auto_len":
        # encode twice
        text_bpe_with_space = bpe.encode(" " + text)
        text_bpe_without_space = bpe.encode(text)
        if len(text_bpe_without_space) <= len(text_bpe_with_space):
            return " ".join(map(str, text_bpe_without_space))
        else:
            return " ".join(map(str, text_bpe_with_space))

    if prepend_space == "auto_cap":
        prepend_space = not text[0].isupper()

    if prepend_space:
        text = " " + text
    text_bpe = " ".join(map(str, bpe.encode(text)))
    return text_bpe
    # text_bin = dictionary.encode_line(
    #     text_bpe, add_if_not_exist=False, append_eos=append_eos
    # )
    # return " ".join(map(str, text_bin.tolist()))


def bpe_tokens(text):
    encoded = bpe_encode(text, prepend_space=False).split(" ")
    tokens = [bpe.decoder.get(int(dictionary[int(x)]), x) for x in encoded]
    tokens = [
        bytearray([bpe.byte_decoder[c] for c in text]).decode(
            "utf-8", errors=bpe.errors
        )
        for text in tokens
    ]
    return tokens


def parse_oxford_3000():
    """
    read from "The Oxford 3000.txt"
    """

    lines = []
    with open("The Oxford 3000.txt", "r", encoding="utf8") as f:
        # due to various reasons, we are unable to distribute this. please
        # contact the authors for more info.
        for line in f:
            if line.strip():
                lines.append(line.strip())
    lines = lines[2:]  # first two lines are not word lines

    lemma_dict = {}

    for line in lines:
        # the file format is highly irregular
        line = re.sub(r"\(.*\)", "", line)  # remove sense indicators
        line = re.sub(
            r"(A1|A2|B1|B2)", r" \1", line
        )  # ensure there is a space before word level

        # lemmas has pos tag and level
        parts = line.split()

        # space in lemma is replaced by . and multiple lemmas are separated by commas
        lemmas = parts[0].replace(".", " ").split(",")

        # consecutive pos tags of the same level are separated by comma or slash
        # same pos tag but with different senses and levels will have separate lines, not dealt with here
        tags = " ".join(parts[1:]).replace(",", " ").replace("/", " ").split()

        pos_level = set()

        level = None
        for tag in reversed(tags):
            if not tag.endswith("."):
                level = tag
            else:
                if level is None:
                    raise ValueError
                pos_level.add((tag, level))

        for lemma in lemmas:
            lemma = lemma.strip()  # remove space around comma separator
            # lemma could end with number to indicate same form word in dictionary
            lemma = re.sub(r"\d$", "", lemma)
            encoded = bpe_tokens(" " + lemma)
            if len(encoded) > 1:
                print(f"{lemma} will be splited in to {encoded}")
            encoded = bpe_tokens(lemma[0].upper() + lemma[1:])
            if len(encoded) > 1:
                print(f"{lemma} will be splited in to {encoded}")
            if lemma in lemma_dict:
                # this is a word with different levels of the same pos tag
                lemma_dict[lemma].update(pos_level)
            else:
                lemma_dict[lemma] = pos_level

    return lemma_dict


def do_oxford3000():
    """
    convert from "The Oxford 3000.txt" to "oxford3000.tsv"
    """
    lemma_dict = parse_oxford_3000()
    with open("oxford3000.tsv", "w", encoding="utf8") as f:
        for lemma in lemma_dict:
            for pos, level in sorted(
                list(lemma_dict[lemma]), key=lambda x: (x[1], x[0])
            ):
                f.write(
                    f"{lemma}\t{bpe_tokens(' ' + lemma)}\t{bpe_tokens(lemma[0].upper()+lemma[1:])}\t{pos}\t{level}\t{is_in_wordnet(lemma, pos)}\t\n"
                )


def do_wordnet():
    """
    read wordnet from nltk and write to "wordnet.jsonl"
    """
    synsets = sorted(list(wn.all_synsets()))
    all_synsets = []
    all_ids = set()

    for synset in tqdm.tqdm(synsets, total=len(synsets)):
        synset_id = synset.name()

        lemmas = [
            dict(
                raw=lemma.name().replace("_", " "),
                # bpe_encoded=bpe_encode(
                #     lemma.name().replace("_", " "), prepend_space="auto_len"
                # ),
            )
            for lemma in synset.lemmas()
            # if "_" not in lemma.name()
        ]

        if not lemmas:
            continue

        definition = dict(
            raw=synset.definition().strip(),
            # bpe_encoded=bpe_encode(synset.definition().strip(), prepend_space=True),
        )
        examples = [
            dict(
                raw=example.strip(),
                # bpe_encoded=bpe_encode(example.strip(), prepend_space="auto_cap"),
            )
            for example in synset.examples()
        ]

        relations = []
        for relation in RELATIONS:
            related_synsets = RELATIONS[relation](synset)
            relations.extend([(relation, synset.name()) for synset in related_synsets])

        if not relations:
            continue

        all_ids.add(synset_id)
        all_synsets.append(
            dict(
                id=synset_id,
                supersense=synset.lexname(),
                lemmas=lemmas,
                definition=definition,
                examples=examples,
                relations=relations,
            )
        )

    for synset in all_synsets:
        synset["relations"] = [
            relation for relation in synset["relations"] if relation[1] in all_ids
        ]

    with open("wordnet.jsonl", "w", encoding="utf8") as f:
        for synset in all_synsets:
            f.write(json.dumps(synset) + "\n")


VALID_SIZE = 1000
TOP_WORDNET_SYNSET = 100  # to disable set a large number


def do_wordnet_oxford():
    """
    sample 1000 words from oxford3000 and treat them as valiation in sempre new task
    """

    # random sample 1000 words from oxford 3000 and make them validation set
    # the format is rel word1 def1 word2 def2
    # so these should not be in the training set
    #    word1 word2
    #    def1 def2
    #    word1 def1
    #    word2 def2
    # the validation set is split into 3 parts:
    # both word1 and word2 are not seen, word1 and word2 is in [set]
    # only word1 def1 is seen, word2 is in [set]
    # only word2 def2 is seen, word1 is in [set]

    # these only affect masked definition training validation (masked predict and relation prediction)
    # no effects on word guess

    all_synsets = {}
    with open("wordnet.jsonl", "r", encoding="utf8") as f:
        for line in f:
            item = json.loads(line)
            all_synsets[item["id"]] = item

    lemmas_in_oxford3000 = []
    with open("oxford3000.tsv", "r", encoding="utf8") as f:
        for line in f:
            parts = line.strip().split()
            lemma = parts[0]
            in_wordnet = eval(parts[-1])
            if in_wordnet:
                lemmas_in_oxford3000.append(lemma)

    lemmas_in_oxford3000 = set(lemmas_in_oxford3000)
    all_relations = []
    excluded_synset = []
    synset_to_ox = {}

    for synset in all_synsets.values():
        all_relations.extend(
            [
                (synset["id"], r, s)
                for r, s in synset["relations"]
                if r != "sister_terms"
            ]
        )
        if any(lemma["raw"] in lemmas_in_oxford3000 for lemma in synset["lemmas"]):
            excluded_synset.append(synset["id"])
            if synset["id"] not in synset_to_ox:
                synset_to_ox[synset["id"]] = []
            for lemma in synset["lemmas"]:
                if lemma["raw"] in lemmas_in_oxford3000:
                    synset_to_ox[synset["id"]].append(lemma["raw"])

    excluded_synset = [
        sid for sid in excluded_synset if int(sid.split(".")[-1]) <= TOP_WORDNET_SYNSET
    ]

    all_relations = all_relations
    excluded_synset = set(excluded_synset)

    print(f"{len(all_synsets)} synsets, {len(all_relations)} relations")
    print(
        f"{len(lemmas_in_oxford3000)} lemmas, {len(excluded_synset)} synsets in oxford3000"
    )

    rel_both = []
    rel_head = []
    rel_tail = []
    rel_neither = []

    for rel in all_relations:
        head = rel[0] in excluded_synset
        tail = rel[2] in excluded_synset
        if head and tail:
            rel_both.append(rel)
        elif head:
            rel_head.append(rel)
        elif tail:
            rel_tail.append(rel)
        else:
            rel_neither.append(rel)

    print(
        f"relations: both {len(rel_both)} head {len(rel_head)} tail {len(rel_tail)} neither {len(rel_neither)}"
    )

    random.seed(1234)

    both_selected = random.sample(rel_both, VALID_SIZE)  # both is not seen
    head_selected = random.sample(rel_head, VALID_SIZE)  # head is not seen
    tail_selected = random.sample(rel_tail, VALID_SIZE)  # tail is not seen
    neither_selected = random.sample(
        rel_neither, VALID_SIZE
    )  # only the relation is not seen

    selected_relations = (
        sorted(both_selected)
        + sorted(head_selected)
        + sorted(tail_selected)
        + sorted(neither_selected)
    )

    selected_synset = set()
    selected_synset.update(
        [r[0] for r in both_selected] + [r[2] for r in both_selected]
    )
    selected_synset.update([r[0] for r in head_selected])
    selected_synset.update([r[2] for r in tail_selected])

    relations = collections.Counter([r[1] for r in selected_relations])
    print(
        f"select {len(selected_synset)} synset for {VALID_SIZE}*4 relations for validation"
    )
    print(f"select {len(relations)} relation types")
    for key, value in relations.most_common():
        print(f"{key}: {value}")

    # valid_data = {}
    # for head, rel, tail in rel_selected:
    #     if head not in valid_data:
    #         item  = {'id'}
    #         valid_data[head] = {"id": head}
    #         valid_data['']
    #         for lemma in all_synsets[head]['lemmas']:
    #             if lemma['raw'] in synset_to_ox[head]:

    with open(f"wn{VALID_SIZE}/info.jsonl", "w", encoding="utf8") as f:
        for sid, synset in all_synsets.items():
            r = synset["relations"]
            del synset["relations"]
            f.write(json.dumps(synset) + "\n")
            synset["relations"] = r

    with open(f"wn{VALID_SIZE}/valid.jsonl", "w", encoding="utf8") as f:
        for head, rel, tail in selected_relations:
            item = dict(relation=rel, head_id=head, tail_id=tail)
            if head in selected_synset:
                item["head_lemmas"] = [
                    lemma
                    for lemma in all_synsets[head]["lemmas"]
                    if lemma["raw"] in synset_to_ox[head]
                ]
            if tail in selected_synset:
                item["tail_lemmas"] = [
                    lemma
                    for lemma in all_synsets[tail]["lemmas"]
                    if lemma["raw"] in synset_to_ox[tail]
                ]
            f.write(json.dumps(item) + "\n")
        print(f"{4*VALID_SIZE} relations selected for valid")

    selected_relations = set(selected_relations)

    with open(f"wn{VALID_SIZE}/train.jsonl", "w", encoding="utf8") as f:
        count = 0
        count_re = 0
        for head, rel, tail in all_relations:
            hin = head in selected_synset
            tin = tail in selected_synset
            if hin and tin:
                continue
            if (head, rel, tail) in selected_relations:
                continue

            if tin and rel in ["hypernyms", "hyponyms"]:
                # if tail is head's hypernyms, tail's hypernyms are also head's hypernyms
                expected_rel = rel
                for rel2, tail2 in all_synsets[tail]["relations"]:
                    # this could be done recursively
                    if rel2 == expected_rel and tail2 not in selected_synset:
                        item = dict(
                            relation=rel,
                            head_id=head,
                            tail_id=tail2,
                            source=f"expanded_tail_{tail}",
                        )
                        f.write(json.dumps(item) + "\n")
                        count_re += 1
            elif hin and rel in ["hypernyms", "hyponyms"]:
                # if tail is head's hypernym, tail is also head's hyponyms' hypernym
                expected_rel = "hypernyms" if rel == "hyponyms" else "hyponyms"
                for rel2, head2 in all_synsets[head]["relations"]:
                    if rel2 == expected_rel and head2 not in selected_synset:
                        item = dict(
                            relation=rel,
                            head_id=head2,
                            tail_id=tail,
                            source=f"expanded_head_{head}",
                        )
                        f.write(json.dumps(item) + "\n")
                        count_re += 1
            else:
                item = dict(relation=rel, head_id=head, tail_id=tail, source="original")
                f.write(json.dumps(item) + "\n")
                count += 1

        print(
            f"{count} relations left for train, {count_re} relations regenerated for train, {len(all_relations)-4*VALID_SIZE-count} relations dropped"
        )


# do_oxford3000()
# do_wordnet()
# do_wordnet_oxford()
