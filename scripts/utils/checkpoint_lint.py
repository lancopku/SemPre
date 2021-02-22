import argparse
import os

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, metavar="FILE")
    args = parser.parse_args()

    ckpt = torch.load(args.path)
    # del ckpt["optimizer_history"]
    del ckpt["last_optimizer_state"]
    prefix, suffix = os.path.splitext(args.path)
    torch.save(ckpt, prefix + "_lint" + suffix)


if __name__ == "__main__":
    main()
