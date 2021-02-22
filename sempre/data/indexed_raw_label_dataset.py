import torch

from fairseq.data import FairseqDataset


class IndexedRawLabelDataset(FairseqDataset):
    def __init__(self, labels, dictionary):
        super().__init__()
        self.labels = [dictionary.index(label) for label in labels]

    def __getitem__(self, index):
        return self.labels[index]

    def __len__(self):
        return len(self.labels)

    def collater(self, samples):
        return torch.tensor(samples)  # pylint: disable=not-callable
