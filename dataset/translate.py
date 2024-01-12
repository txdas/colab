import random

import torch
import jsonlines
from torch.utils.data.dataset import T_co
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data import get_tokenizer
from torch.utils.data import Dataset, DataLoader
from functools import partial
import os
en_tokenizer = get_tokenizer("basic_english")
BASE = os.path.dirname(os.path.dirname(__file__))
print(BASE)


def ch_tokenizer(text):
    return [x for x in text]


def yield_en_tokens():
    with jsonlines.open(os.path.join(BASE,"./data/Corpus/corpus.json")) as fp:
        for v in fp:
            yield [x for x in en_tokenizer(v["en"])]


def yield_ch_tokens():
    with jsonlines.open(os.path.join(BASE,"./data/Corpus/corpus.json")) as fp:
        for v in fp:
            yield [x for x in ch_tokenizer(v["ch"])]


def main():
    vocab = build_vocab_from_iterator(yield_ch_tokens(), specials=["<unk>", "<pad>", "<sep>", '<sos>', '<eos>'])
    vocab.set_default_index(vocab["<unk>"])
    torch.save(vocab, os.path.join(BASE, "./data/Corpus/vocab_ch"))
    vocab = build_vocab_from_iterator(yield_en_tokens(), specials=["<unk>","<pad>", "<sep>", '<sos>', '<eos>'])
    vocab.set_default_index(vocab["<unk>"])
    torch.save(vocab, os.path.join(BASE, "./data/Corpus/vocab_en"))


class TranslateDataset(Dataset):

    def __init__(self, fn="data/Corpus/corpus.json"):
        super(TranslateDataset, self).__init__()
        self.fn = os.path.join(BASE, fn)
        self.content = self.init()
        self.vocab_ch = torch.load(os.path.join(BASE, "data/Corpus/vocab_ch"))
        self.vocab_ch.set_default_index(self.vocab_ch["<unk>"])
        self.vocab_en = torch.load(os.path.join(BASE, "data/Corpus/vocab_en"))
        self.vocab_en.set_default_index(self.vocab_en["<unk>"])

    def init(self):
        ret = []
        with jsonlines.open(self.fn) as fp:
            for v in fp:
                ret.append(v)
        return ret

    def __getitem__(self, index) -> T_co:
        return self.content[index]["en"], self.content[index]["ch"]

    def __len__(self):
        return len(self.content)


def collate_batch(batch, vocab_en, vocab_ch):
    src_tensors, trg_tensors = [], []
    maxsrc, maxtrg, batch_size = 0, 0, 0
    for src, trg in batch:
        src_tensor = [vocab_en["sos"]]+vocab_en(en_tokenizer(src))+[vocab_en["eos"]]
        trg_tensor = [vocab_en["sos"]]+vocab_ch(ch_tokenizer(trg))+[vocab_en["eos"]]
        maxsrc = max(maxsrc, len(src_tensor))
        maxtrg = max(maxtrg, len(trg_tensor))
        src_tensors.append(src_tensor)
        trg_tensors.append(trg_tensor)
        batch_size += 1
    src_tensors = [x+(maxsrc-len(x))*[vocab_en["<pad>"]] for x in src_tensors]
    trg_tensors = [x+(maxtrg-len(x))*[vocab_ch["<pad>"]] for x in trg_tensors]
    src_len = torch.tensor([maxsrc]*batch_size)
    return torch.tensor(src_tensors, dtype=torch.long).permute(1,0), src_len, torch.tensor(trg_tensors, dtype=torch.long).permute(1,0)


def evalate():
    dataset = TranslateDataset()
    loader = DataLoader(dataset, batch_size=8, shuffle=True,
                        collate_fn=partial(collate_batch, vocab_en=dataset.vocab_en, vocab_ch=dataset.vocab_ch))
    # for v in dataset:
    #     print(v)
    for v in loader:
        print(v)
        break


def train_test_split():
    lst = []
    with jsonlines.open("./data/Corpus/corpus.json") as fp:
        for v in fp:
            lst.append(v)
    size = len(lst)
    idxs = set(random.sample(list(range(size)), int(size*0.8)))
    train_fp = jsonlines.open("./data/Corpus/train.json", "w")
    test_fp = jsonlines.open("./data/Corpus/test.json", "w")
    for i, v in enumerate(lst):
        if i in idxs:
            train_fp.write(v)
        else:
            test_fp.write(v)


if __name__ == '__main__':
    main()
    # train_test_split()
    # evalate()