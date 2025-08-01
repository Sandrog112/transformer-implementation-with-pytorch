import torch

class RandomSequenceDataset:
    def __init__(self, vocab_size, seq_length, dataset_size):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # Generate random src and tgt sequences of token ids
        src = torch.randint(1, self.vocab_size, (self.seq_length,))  # avoid 0 if padding token is 0
        tgt = torch.randint(1, self.vocab_size, (self.seq_length,))
        return src, tgt


def collate_fn(batch):
    # batch is list of tuples (src, tgt)
    src_batch = torch.stack([item[0] for item in batch])
    tgt_batch = torch.stack([item[1] for item in batch])
    return src_batch, tgt_batch
