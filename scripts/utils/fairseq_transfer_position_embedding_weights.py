"""
Revert the rest position embeddings from checkpoints, as we truncate position embeddings
in sempre training
"""
import argparse
import os
import time

import torch


def type_as_fp16(sample):
    # adapted from fairseq
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x) and x.is_floating_point():
            return x.type(torch.float16)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source_checkpoint",
        type=str,
        help="fairseq checkpoint that provides the extra position embeddings",
    )
    parser.add_argument(
        "target_checkpoint",
        type=str,
        help="the checkpoint that the extra position embeddings will transfer to",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="whether to save the new checkpoint as fp16",
    )

    args = parser.parse_args()

    start_time = time.time()

    src = torch.load(args.source_checkpoint, map_location="cpu")
    tgt = torch.load(args.target_checkpoint, map_location="cpu")

    src_position_embed: torch.Tensor = src["model"][
        "decoder.sentence_encoder.embed_positions.weight"
    ]
    tgt_position_embed: torch.Tensor = tgt["model"][
        "decoder.sentence_encoder.embed_positions.weight"
    ]
    print(
        f"source position embedding is of size {src_position_embed.shape} and data "
        f"type {src_position_embed.dtype}"
    )

    print(
        f"target position embedding is of size {tgt_position_embed.shape} and data "
        f"type {tgt_position_embed.dtype}"
    )

    if tgt_position_embed.size(1) != src_position_embed.size(1):
        print("position embedding size not compatible")
        return
    if tgt_position_embed.size(0) >= src_position_embed.size(0):
        print("target position embedding has more positions than source")
        return

    if args.fp16:
        print("changing target type to fp16")
        tgt = type_as_fp16(tgt)
        src_position_embed = src["model"][
            "decoder.sentence_encoder.embed_positions.weight"
        ]
        tgt_position_embed = tgt["model"][
            "decoder.sentence_encoder.embed_positions.weight"
        ]

    # change args, don't touch tokens_per_sample, which is for the task, by design, this is 2 larger
    # than positional embeddings
    print(
        f"changing max_positions from {tgt['args'].max_positions} to {src['args'].max_positions}"
    )
    tgt["args"].max_positions = src["args"].max_positions

    new_num_positions = src_position_embed.size(0)

    new_base_name, new_ext = os.path.splitext(args.target_checkpoint)
    new_checkpoint = new_base_name + f"-pos{new_num_positions}" + new_ext

    # copy embedding
    if tgt_position_embed.dtype != src_position_embed.dtype:
        print("using data dtype of target position embedding")
    new_position_embed = src_position_embed.type_as(
        tgt_position_embed
    )  # this could be an inplace op
    new_position_embed[: tgt_position_embed.size(0)] = tgt_position_embed
    tgt["model"]["decoder.sentence_encoder.embed_positions.weight"] = new_position_embed

    # clear optimizer state if there is any
    if "last_optimizer_state" in tgt:
        print("deleting optimizer state, as we are doing hacks here")
        del tgt["last_optim_state"]

    torch.save(tgt, new_checkpoint, _use_new_zipfile_serialization=False)
    print(f"finished in {time.time() - start_time} seconds")


if __name__ == "__main__":
    main()
