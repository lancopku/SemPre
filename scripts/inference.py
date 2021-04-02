import argparse
import json
import os
import sys

import numpy as np

# import scipy.stats
# import sklearn.metrics
import torch
from fairseq import hub_utils, tasks, utils
from fairseq.models.roberta import RobertaHubInterface


def get_args():
    parser = argparse.ArgumentParser()

    # fmt: off
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-name", type=str, default="checkpoint_best.pt")
    parser.add_argument("--mode", type=str, default="validate", help='validate and provide the spilt in data-split or the test file path')
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--metric", type=str, default=["acc"], nargs="+",
                        choices=["acc", "mcc", "pcc", "scc", "f1"], help='metric used in validation mode')
    parser.add_argument("--user-dir", default=None)
    parser.add_argument("--max-tokens", default=None)
    parser.add_argument("--max-sentences", type=int, default=32)
    parser.add_argument("--task", default='classification')
    parser.add_argument("--cpu", action='store_true', default=False)
    parser.add_argument('--dummy-test', action='store_true', default=False)
    parser.add_argument('--analysis', action='store_true', default=False)
    parser.add_argument("--valid-subset", type=str, default='valid')
    parser.add_argument("data", type=str)
    # fmt: on
    args = parser.parse_args()

    assert args.mode in ["validate"] or (
        os.path.isfile(args.mode) and (args.mode.endswith(".jsonl"))
    )

    if not os.path.isfile(os.path.join(args.checkpoint_dir, args.checkpoint_name)) and (
        args.mode.endswith(".jsonl")
    ):
        raise ValueError("checkpoint does not exist; please use --dummy-test")

    args.choices = 1
    if args.task == "WiC":
        args.metric = ["acc"]
    elif args.task == "PIQA":
        args.metric = ["acc"]

    return args


def from_pretrained(args):
    # this is not a roberta model but a roberta hub interface
    kwargs = {}
    if args.user_dir is not None:
        kwargs["user_dir"] = args.user_dir
    kwargs["load_checkpoint_name"] = True

    x = hub_utils.from_pretrained(
        args.checkpoint_dir, args.checkpoint_name, args.data, **kwargs
    )

    return RobertaHubInterface(x["args"], x["task"], x["models"][0])


def wic_validate(args, roberta):
    # special treatment for wic
    roberta_args = roberta.args
    roberta_args.data = args.data
    roberta_args.skip_invalid_size_inputs_valid_test = (
        False  # never drop data for testing purpose
    )
    if args.cpu:
        roberta_args.cpu = True

    print(roberta_args)
    print(roberta.model)

    # wsc and winogrande have different procedures
    # use task to unify the code
    task = tasks.setup_task(roberta_args)

    # to get accuracy, reuse the criterion class, this will compute loss
    # so can't be used in inference
    criterion = task.build_criterion(roberta_args)

    # the data will be shuffled
    dataset = task.load_dataset(args.valid_subset, combine=False, epoch=0)
    # {id, net_input: {src_tokens, src_lengths}, nsentences, ntokens, target}

    itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), roberta.model.max_positions()
        ),
        ignore_invalid_inputs=False,
        required_batch_size_multiple=roberta_args.required_batch_size_multiple,
        seed=roberta_args.seed,
        num_workers=roberta_args.num_workers,
    ).next_epoch_itr(shuffle=False)

    roberta.eval()
    if not args.cpu:
        roberta.cuda()

    # logging_outputs = []
    ncorrect = 0
    nsample = 0
    with torch.no_grad():
        for sample in itr:
            # we only use sample['net_input']['src_tokens'] and sample['target_labels']
            # but less is more
            if not args.cpu:
                sample = utils.move_to_cuda(sample)
            _, _, logging_output = criterion(roberta.model, sample, reduce=True)
            # logging_outputs.append(criterion(roberta.model, sample, reduce=True)[2])
            ncorrect += logging_output["ncorrect"]
            nsample += logging_output["nsentences"]

    # log = criterion.aggregate_logging_outputs(logging_outputs)
    # ncorrect = log["ncorrect"]
    # nsample = log["nsentences"]
    print(f"| Accuracy {ncorrect/nsample:.6f} | Correct {ncorrect} | All {nsample}")


def roberta_predict(args, roberta, net_input):

    roberta_args = roberta.args

    if hasattr(roberta_args, "classification_head_name"):
        head_name = roberta_args.classification_head_name
    elif hasattr(roberta_args, "ranking_head_name"):
        head_name = roberta_args.ranking_head_name
    else:
        head_name = "sentence_classification_head"

    if not args.cpu:
        net_input = utils.move_to_cuda(net_input)
    features, _ = roberta.model(
        **net_input, features_only=True, return_all_hiddens=False
    )
    logits = roberta.model.classification_heads[head_name](features)
    return logits


def validate(args, roberta):
    roberta_args = roberta.args
    roberta_args.data = args.data
    roberta_args.no_shuffle = True  # don't shuffle, as we may need to output the result
    roberta_args.skip_invalid_size_inputs_valid_test = (
        False  # never drop data for testing purpose
    )
    roberta_args.truncate_sequence = True
    if args.cpu:
        roberta_args.cpu = True

    roberta_args.regression_target = getattr(roberta_args, "regression_target", False)

    print(roberta_args)
    print(roberta.model)

    # task contains dataset processing, loading and batching,
    # so we don't need to do ourselves
    # task args are already in the checkpoints anyway, just use them
    task = tasks.setup_task(roberta_args)

    dataset = task.load_dataset(args.valid_subset, combine=False, epoch=0)
    # {id, net_input: {src_tokens, src_lengths}, nsentences, ntokens, target}

    itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), roberta.model.max_positions()
        ),
        ignore_invalid_inputs=False,
        required_batch_size_multiple=roberta_args.required_batch_size_multiple,
        seed=roberta_args.seed,
        num_workers=roberta_args.num_workers,
    ).next_epoch_itr(shuffle=False)

    ground_truth = []
    prediction = []

    roberta.eval()
    if not args.cpu:
        roberta.cuda()
    with torch.no_grad():
        for sample in itr:

            scores = []
            # taking ranking into account
            # name format is net_input\d
            for key in sample.keys():

                if not key.startswith("net_input"):
                    continue

                # net_inputs
                logits = roberta_predict(args, roberta, sample[key])

                scores.append(logits)

            if len(scores) == 1:
                # classification
                logits = scores[0]
            else:
                # ranking
                logits = torch.cat(scores, dim=1)

            ground_truth.extend(sample["target"].squeeze().tolist())

            if roberta_args.regression_target:
                prediction.extend(logits.squeeze(dim=1).tolist())
            elif roberta_args.criterion == "sentence_prediction" and args.choices > 1:
                # multiple choices to binary classification
                # to calculate accuracy needs more handling
                prediction.extend(logits.squeeze(1).tolist())
            elif roberta_args.criterion == "sentence_ranking":
                # ranking
                prediction.extend(logits.argmax(dim=1).tolist())
            else:
                # classification
                prediction.extend(logits.argmax(dim=1).tolist())

    assert len(ground_truth) == len(prediction)

    ground_truth = np.array(ground_truth)
    prediction = np.array(prediction)

    nsample = len(ground_truth)
    for metric in args.metric:
        if metric == "acc":
            if roberta_args.criterion == "sentence_prediction" and args.choices > 1:
                # classification
                ground_truth = ground_truth.reshape((-1, roberta_args.num_classes))
                # coarse check that each column is actually the same example
                assert np.all(
                    np.max(ground_truth, axis=1) == np.min(ground_truth, axis=1)
                )
                ground_truth = ground_truth[:, 0]

                prediction = prediction.reshape((-1, roberta_args.num_classes))
                prediction = np.argmax(prediction, axis=1)

            ncorrect = (ground_truth == prediction).sum()
            print(
                f"| Accuracy {ncorrect/nsample:.6f} | Correct {ncorrect} | All {nsample}"
            )
        # elif metric == "mcc":
        #     ncorrect = (ground_truth == prediction).sum()
        #     mcc = sklearn.metrics.matthews_corrcoef(ground_truth, prediction)
        #     print(f"| MatthewCC {mcc:.6f} | Correct {ncorrect} | All {nsample}")
        # elif metric == "pcc":
        #     pcc, p = scipy.stats.pearsonr(ground_truth, prediction)
        #     print(f"| PearsonCC {pcc:.6f} | p-value {p} | All {nsample}")
        # elif metric == "scc":
        #     scc, p = scipy.stats.spearmanr(ground_truth, prediction)
        #     print(f"| SpearmanCC {scc:.6f} | p-value {p} | All {nsample}")
        elif metric == "f1":
            # this is a little tricky
            # in preprocessing, label mapping is based on frequency, meaning label 0 could have index 1
            # this affects f1 but not accuracy and mcc
            pos_label = (
                roberta.task.label_dictionary.index("1")
                - roberta.task.label_dictionary.nspecial
            )
            f1 = sklearn.metrics.f1_score(ground_truth, prediction, pos_label=pos_label)
            print(f"| F1 {f1:.6f} | All {nsample}")


def test_dummy(args, path, save_path=None):
    if save_path is not None:
        fout = open(save_path, "w", encoding="utf8")
    else:
        fout = sys.stdout
    if path.endswith(".jsonl"):
        if args.task in ["WiC"]:
            default = "False"
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                item = json.loads(line)
                print(json.dumps(dict(idx=item["idx"], label=default)), file=fout)


def analysis_jsonl(args, roberta):
    roberta_args = roberta.args
    roberta_args.no_shuffle = True  # don't shuffle, as we may need to output the result
    roberta_args.skip_invalid_size_inputs_valid_test = (
        False  # never drop data for testing purpose
    )
    roberta_args.truncate_sequence = True

    print(roberta_args)
    print(roberta.model)

    torch.set_grad_enabled(False)

    task = tasks.setup_task(roberta_args)

    raw_data = [json.loads(line) for line in open(args.mode, "r", encoding="utf8")]

    # provide data_path, override split argument
    dataset = task.load_dataset(
        "placeholder", data_path=args.mode, combine=False, epoch=0
    )

    itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), roberta.model.max_positions()
        ),
        ignore_invalid_inputs=roberta_args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=roberta_args.required_batch_size_multiple,
        seed=roberta_args.seed,
        num_workers=roberta_args.num_workers,
    ).next_epoch_itr(shuffle=False)

    roberta.eval()

    if not args.cpu:
        roberta.cuda()

    if args.save_path is not None:
        fout = open(args.save_path, "w", encoding="utf8")
    else:
        fout = sys.stdout

    def label_fn(label):
        if roberta_args.criterion == "wic":
            return str(label == 1)
        try:
            return roberta.task.label_dictionary.string(
                [label + roberta.task.label_dictionary.nspecial]
            )
        except AttributeError:
            return label

    idx = 0
    ncorrect = 0
    for sample in itr:
        if roberta_args.criterion == "sentence_prediction":

            logits = roberta_predict(args, roberta, sample["net_input"])

            predictions = (
                logits.squeeze(dim=1).tolist()
                if roberta_args.regresion_target
                else logits.argmax(dim=1).tolist()
            )

        elif roberta_args.criterion == "sentence_ranking":
            scores = []
            for i in range(roberta_args.num_classes):
                score = roberta_predict(args, roberta, sample[f"net_input{i+1}"])
                scores.append(score)

            logits = torch.cat(scores, dim=1)
            predictions = logits.argmax(dim=1).tolist()
        elif roberta_args.criterion == "wic":
            sample = utils.move_to_cuda(sample)
            hiddens, _ = roberta.model(**sample["net_input"], features_only=True)
            embeddings = []
            embeddings.append(hiddens[:, 0, :])
            for i in range(2):
                index = (
                    sample["net_input"]["src_ranges"][f"range{i+1}"]
                    .unsqueeze(-1)
                    .expand([-1, -1, hiddens.size(-1)])
                )
                mask = index != 0
                embedding = hiddens.gather(dim=1, index=index) * mask
                embedding = embedding.sum(dim=1) / mask.sum(dim=1)
                embeddings.append(embedding)

            ctx_emb = hiddens.gather(dim=1, index=index)
            concat = torch.cat(ctx_emb, dim=1)
            logits = roberta.model.classification_heads["sentence_classification_head"](
                concat.unsqueeze(1)
            )
            predictions = logits.argmax(dim=1).tolist()
        else:
            raise NotImplementedError

        for prediction in predictions:
            print(
                "| ".join(f"{key}: {value} " for key, value in raw_data[idx].items()),
                file=fout,
            )
            print(
                f"| idx {idx} | predict_label {label_fn(prediction)} | correct {str(label_fn(prediction))==str(raw_data[idx]['label'])}",
                file=fout,
            )
            ncorrect += str(label_fn(prediction)) == str(raw_data[idx]["label"])
            print("", file=fout)
            idx += 1

    print(
        f"| accuracy {ncorrect/idx:.4f} | total {idx} | correct {ncorrect}", file=fout
    )


def test_jsonl(args, roberta):

    roberta_args = roberta.args
    roberta_args.no_shuffle = True  # don't shuffle, as we may need to output the result
    roberta_args.skip_invalid_size_inputs_valid_test = (
        False  # never drop data for testing purpose
    )
    roberta_args.truncate_sequence = True

    print(roberta_args)
    print(roberta.model)

    torch.set_grad_enabled(False)

    task = tasks.setup_task(roberta_args)

    # provide data_path, override split argument
    dataset = task.load_dataset(
        "placeholder", data_path=args.mode, combine=False, epoch=0
    )

    itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), roberta.model.max_positions()
        ),
        ignore_invalid_inputs=roberta_args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=roberta_args.required_batch_size_multiple,
        seed=roberta_args.seed,
        num_workers=roberta_args.num_workers,
    ).next_epoch_itr(shuffle=False)

    roberta.eval()

    if not args.cpu:
        roberta.cuda()

    if args.save_path is not None:
        fout = open(args.save_path, "w", encoding="utf8")
    else:
        fout = sys.stdout

    def label_fn(label):
        if roberta_args.criterion == "wic":
            return str(label == 1)
        try:
            return roberta.task.label_dictionary.string(
                [label + roberta.task.label_dictionary.nspecial]
            )
        except AttributeError:
            return label

    idx = 0
    for sample in itr:
        if roberta_args.criterion == "sentence_prediction":

            logits = roberta_predict(args, roberta, sample["net_input"])

            predictions = (
                logits.squeeze(dim=1).tolist()
                if roberta_args.regresion_target
                else logits.argmax(dim=1).tolist()
            )

        elif roberta_args.criterion == "sentence_ranking":
            scores = []
            for i in range(roberta_args.num_classes):
                score = roberta_predict(args, roberta, sample[f"net_input{i+1}"])
                scores.append(score)

            logits = torch.cat(scores, dim=1)
            predictions = logits.argmax(dim=1).tolist()
        elif roberta_args.criterion == "wic":
            sample = utils.move_to_cuda(sample)
            hiddens, _ = roberta.model(**sample["net_input"], features_only=True)
            embeddings = []
            embeddings.append(hiddens[:, 0, :])
            for i in range(2):
                index = (
                    sample["net_input"]["src_ranges"][f"range{i+1}"]
                    .unsqueeze(-1)
                    .expand([-1, -1, hiddens.size(-1)])
                )
                mask = index != 0
                embedding = hiddens.gather(dim=1, index=index) * mask
                embedding = embedding.sum(dim=1) / mask.sum(dim=1)
                embeddings.append(embedding)
            # ctx_emb = hiddens.gather(dim=1, index=index)
            ctx_emb = embeddings
            concat = torch.cat(ctx_emb, dim=1)
            logits = roberta.model.classification_heads["sentence_classification_head"](
                concat.unsqueeze(1)
            )
            predictions = logits.argmax(dim=1).tolist()
        else:
            raise NotImplementedError

        for prediction in predictions:

            print(json.dumps(dict(idx=idx, label=label_fn(prediction))), file=fout)
            idx += 1


def main():
    args = get_args()
    print(args)
    if args.mode not in ["validate", "inference"] and args.dummy_test:
        test_dummy(args, args.mode, args.save_path)
        return

    roberta = from_pretrained(args)

    if args.mode.endswith(".jsonl"):
        if args.analysis:
            analysis_jsonl(args, roberta)
        else:
            test_jsonl(args, roberta)
    elif roberta.args.task == "wic":
        wic_validate(args, roberta)
    else:
        validate(args, roberta)
    print("| done ")


if __name__ == "__main__":
    main()
