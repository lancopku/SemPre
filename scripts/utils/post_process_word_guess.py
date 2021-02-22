import json


OX3000_TSV_PATH = "oxford3000.tsv"
VALID_JSONL = "wn1000/valid.jsonl"
TSV_SAVE_PATH = "roxford3000-wn1000.tsv"
# SAVE_VALID = False


def extract_raw(item, name):
    return [lemma["raw"] for lemma in item.get(name, [])]


num_item = 0
lemmas = set()
with open(VALID_JSONL, "r", encoding="utf8") as f:
    for line in f:
        item = json.loads(line)
        lemmas.update(extract_raw(item, "head_lemmas"))
        lemmas.update(extract_raw(item, "tail_lemmas"))
        num_item += 1
print(f"{num_item} line {len(lemmas)} lemma")


samples = []
with open(OX3000_TSV_PATH, encoding="utf8") as f:
    for line in f:
        parts = line.strip("\n").split("\t")

        if not eval(parts[-2]):
            continue

        valid = parts[0] in lemmas

        if valid:
            parts[-1] = "valid"
        else:
            parts[-1] = "train"

        line = "\t".join(parts) + "\n"

        # if valid == SAVE_VALID:
        #     # when they are the same save
        samples.append(line)

print(f"| Load {len(samples)} examples")

with open(TSV_SAVE_PATH, "w", encoding="utf8") as f:
    f.writelines(samples)
