"""
Commonsense Ability Tests
(Zhou et. al., 2020, Evaluating Commonsense in Pre-trained Language Models, https://arxiv.org/abs/1911.11931)


for FILE in ca.txt wsc.txt sm.txt smr.txt swag.txt hella_swag.txt arct_1.txt arct_2.txt; do
    CUDA_VISIBLE_DEVICES=$GPU python making_sense.py --data-dir $DATA_ROOT --user-dir ../sempre --checkpoint-dir $CKPT_ROOT --checkpoint-name $CKPT_NAME $FILE | tee -a cats.txt
done
"""

import argparse
import os
from collections import OrderedDict

import torch
import torch.nn.functional as F
from fairseq.data import encoders
from fairseq.models.roberta import RobertaHubInterface, RobertaModel
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()

    # fmt: off
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-name", type=str, default="checkpoint_best.pt")
    parser.add_argument("--user-dir", default=None)
    parser.add_argument("--cpu", action='store_true')
    parser.add_argument('--data-dir', type=str, default='cats')
    parser.add_argument("data", type=str)
    # fmt: on
    args = parser.parse_args()

    assert os.path.isfile(os.path.join(args.checkpoint_dir, args.checkpoint_name))
    assert os.path.isfile(os.path.join(args.data_dir, args.data))

    return args


def from_pretrained(args):
    # this is not a roberta model but a roberta hub interface
    kwargs = dict(
        checkpoint_file=args.checkpoint_name,
        data_name_or_path=args.data_dir,
        gpt2_encoder_json=encoders.gpt2_bpe.DEFAULT_ENCODER_JSON,
        gpt2_vocab_bpe=encoders.gpt2_bpe.DEFAULT_VOCAB_BPE,
    )
    if args.user_dir is not None:
        kwargs["user_dir"] = args.user_dir
    return RobertaModel.from_pretrained(args.checkpoint_dir, **kwargs)


def roberta_sentence_score(text, roberta: RobertaHubInterface, skip_invalid_size=True):
    # some files containing " [SEP]"
    if " [SEP]" in text:
        text = text.replace(" [SEP]", "")
    text_bpe = roberta.bpe.encode(text).split()
    sent_bpe = ["<s>"] + text_bpe + ["</s>"]
    sent_ind = [roberta.task.source_dictionary.index(token) for token in sent_bpe]
    mask_index = roberta.task.source_dictionary.index("<mask>")

    if len(sent_bpe) > 128 and skip_invalid_size:
        return None

    tensor = (
        torch.tensor(sent_ind, dtype=torch.int64, device=roberta.device)
        .unsqueeze(0)
        .expand(len(sent_ind) - 2, -1)
    ).contiguous()

    mask = torch.eye(len(sent_ind), dtype=torch.bool, device=roberta.device)[1:-1, :]
    tensor[mask] = mask_index

    targets = torch.tensor(sent_ind[1:-1], dtype=torch.int64, device=roberta.device)

    hiddens, _ = roberta.model(tensor, features_only=True, return_all_hiddens=False)
    logits = roberta.model.output_layer(hiddens, masked_tokens=mask)

    logits = logits.view(len(sent_ind) - 2, logits.size(-1))

    score = -F.cross_entropy(logits, targets, reduction="mean").item()

    return score


def argmax(l):
    return max(range(len(l)), key=lambda i: l[i])


def main():
    args = get_args()
    print(args)
    roberta = from_pretrained(args)
    torch.set_grad_enabled(False)
    roberta.float()
    roberta.eval()
    if not args.cpu and torch.cuda.is_available():
        roberta.cuda()
    with open(os.path.join(args.data_dir, args.data), "r", encoding="utf8") as f:

        total = 0
        count = 0
        correct = 0

        pbar = tqdm(list(f), leave=False, ascii=True)
        for line in pbar:
            total += 1

            parts = line.strip().split("\001")
            label1 = int(parts[0])
            if args.robust:
                label2 = int(parts[3])

            scores = []

            texts = (
                ([part for part in parts if len(part) > 1])
                if args.robust
                else parts[1:]
            )

            for text in texts:
                if len(text) == 1:
                    continue

                score = roberta_sentence_score(text, roberta, True)
                scores.append(score)

            if None in scores:
                continue
            count += 1

            if args.robust:
                predict1 = argmax(scores[:2])
                predict2 = argmax(scores[2:])
                correct += (predict1 == label1) + (predict2 == label2) != 1
            else:
                predict1 = argmax(scores)
                correct += predict1 == label1

            pbar.set_postfix(
                OrderedDict(
                    correct=correct,
                    count=count,
                    skipped=total - count,
                    accuracy=f"{correct/count:.4f}",
                ),
                refresh=False,
            )

    print(
        f"| model {os.path.basename(args.checkpoint_dir)}/{os.path.basename(args.checkpoint_name)} | data {args.data}"
    )
    print(
        f"| correct {correct} | all {count} | accuracy {correct/count:.6f} | skipped {total-count} "
    )
    print("| done")


if __name__ == "__main__":
    main()
