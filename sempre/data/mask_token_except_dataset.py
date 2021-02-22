from typing import Callable, List, Optional

import numpy as np
import torch
from fairseq.data import BaseWrapperDataset, Dictionary, LRUCacheDataset, data_utils


class MaskTokenExceptDataset(BaseWrapperDataset):
    """
    A wrapper Dataset for masked language modeling.

    Input items are masked according to the specified masking probability.
    keep_words are not maksed

    Args:
        dataset: Dataset to wrap.
        vocab: Dictionary with the vocabulary and special tokens.
        pad_idx: Id of pad token in vocab
        mask_idx: Id of mask token in vocab
        return_masked_tokens: controls whether to return the non-masked tokens
            (the default) or to return a tensor with the original masked token
            IDs (and *pad_idx* elsewhere). The latter is useful as targets for
            masked LM training.
        seed: Seed for random number generator for reproducibility.
        mask_prob: probability of replacing a token with *mask_idx*.
        leave_unmasked_prob: probability that a masked token is unmasked.
        random_token_prob: probability of replacing a masked token with a
            random token from the vocabulary.
        freq_weighted_replacement: sample random replacement words based on
            word frequencies in the vocab.
        mask_whole_words: only mask whole words. This should be a byte mask
            over vocab indices, indicating whether it is the beginning of a
            word. We will extend any mask to encompass the whole word.
        keep_words: do not mask these words. This should be a list of the indices
            of the words that should not be masked
        allow_function: a function that outputs allowed positions for masking. This
            takes an np array as input and outputs the position indices.
    """

    @classmethod
    def apply_mask(cls, dataset: torch.utils.data.Dataset, *args, **kwargs):
        """Return the source and target datasets for masked LM training."""
        dataset = LRUCacheDataset(dataset)
        return (
            LRUCacheDataset(cls(dataset, *args, **kwargs, return_masked_tokens=False)),
            LRUCacheDataset(cls(dataset, *args, **kwargs, return_masked_tokens=True)),
        )

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        vocab: Dictionary,
        pad_idx: int,
        mask_idx: int,
        return_masked_tokens: bool = False,
        seed: int = 1,
        mask_prob: float = 0.15,
        leave_unmasked_prob: float = 0.1,
        random_token_prob: float = 0.1,
        freq_weighted_replacement: bool = False,
        mask_whole_words: torch.Tensor = None,
        keep_words: list = None,
        allow_function: Optional[Callable[[List], List]] = None,
    ):
        assert 0.0 < mask_prob < 1.0
        assert 0.0 <= random_token_prob <= 1.0
        assert 0.0 <= leave_unmasked_prob <= 1.0
        assert random_token_prob + leave_unmasked_prob <= 1.0

        self.dataset = dataset
        self.vocab_size = len(vocab)
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        self.return_masked_tokens = return_masked_tokens
        self.seed = seed
        self.mask_prob = mask_prob
        self.leave_unmasked_prob = leave_unmasked_prob
        self.random_token_prob = random_token_prob
        self.mask_whole_words = mask_whole_words
        if keep_words is not None:
            self.keep_words = np.array(keep_words)
        else:
            self.keep_words = None
        self.allow_function = allow_function

        if random_token_prob > 0.0:
            if freq_weighted_replacement:
                weights = np.array(vocab.count)
            else:
                weights = np.ones(self.vocab_size)
            weights[: vocab.nspecial] = 0
            if self.keep_words is not None:
                weights[self.keep_words] = 0
            self.weights = weights / weights.sum()

        self.epoch = 0

    def set_epoch(self, epoch, **unused):
        self.epoch = epoch

    # @lru_cache(maxsize=8) # why cache the results?
    def __getitem__(self, index: int):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            item = self.dataset[index]
            size = len(item)

            assert (
                self.mask_idx not in item
            ), "Dataset contains mask_idx (={}), this is not expected!".format(
                self.mask_idx
            )

            if self.mask_whole_words is not None:
                word_begins_mask = self.mask_whole_words.gather(0, item)
                word_begins_idx = word_begins_mask.nonzero().view(-1)
                size = len(word_begins_idx)
                words = np.split(word_begins_mask, word_begins_idx)[1:]
                assert len(words) == size
                word_lens = list(map(len, words))

            # decide elements to mask
            mask = np.full(size, False)

            # can_be_masked is a list of indices that can be masked
            # do not mask keep_words
            if self.keep_words is not None:
                can_be_masked = np.where(np.isin(item, self.keep_words, invert=True))[0]
            else:
                can_be_masked = None

            # only mask allowed_words
            if self.allow_function is not None:
                if can_be_masked is not None:
                    can_be_masked_2 = self.allow_function(item)
                    can_be_masked = np.intersect1d(
                        can_be_masked, can_be_masked_2, assume_unique=True
                    )
                else:
                    can_be_masked = self.allow_function(item)

            nsz = size
            if can_be_masked is not None:
                nsz -= len(item) - len(can_be_masked)

            # at least mask one and at most mask all except one
            num_mask = min(
                max(
                    int(
                        # add a random number for probabilistic rounding
                        self.mask_prob * nsz
                        + np.random.rand()
                    ),
                    1,
                ),
                nsz - 1,
            )

            if can_be_masked is not None:
                mask[np.random.choice(can_be_masked, num_mask, replace=False)] = True
            else:
                mask[np.random.choice(size, num_mask, replace=False)] = True

            if self.return_masked_tokens:
                # exit early if we're just returning the masked tokens
                # (i.e., the targets for masked LM training)
                if self.mask_whole_words is not None:
                    mask = np.repeat(mask, word_lens)
                new_item = np.full(len(mask), self.pad_idx)
                new_item[mask] = item[torch.from_numpy(mask.astype(np.uint8)) == 1]
                return torch.from_numpy(new_item)

            # decide unmasking and random replacement
            rand_or_unmask_prob = self.random_token_prob + self.leave_unmasked_prob
            if rand_or_unmask_prob > 0.0:
                rand_or_unmask = mask & (np.random.rand(size) < rand_or_unmask_prob)
                if self.random_token_prob == 0.0:
                    unmask = rand_or_unmask
                    rand_mask = None
                elif self.leave_unmasked_prob == 0.0:
                    unmask = None
                    rand_mask = rand_or_unmask
                else:
                    unmask_prob = self.leave_unmasked_prob / rand_or_unmask_prob
                    decision = np.random.rand(size) < unmask_prob
                    unmask = rand_or_unmask & decision
                    rand_mask = rand_or_unmask & (~decision)
            else:
                unmask = rand_mask = None

            if unmask is not None:
                mask = mask ^ unmask

            if self.mask_whole_words is not None:
                mask = np.repeat(mask, word_lens)

            new_item = np.copy(item)
            new_item[mask] = self.mask_idx
            if rand_mask is not None:
                num_rand = rand_mask.sum()
                if num_rand > 0:
                    if self.mask_whole_words is not None:
                        rand_mask = np.repeat(rand_mask, word_lens)
                        num_rand = rand_mask.sum()

                    new_item[rand_mask] = np.random.choice(
                        self.vocab_size, num_rand, p=self.weights
                    )

            return torch.from_numpy(new_item)
