from models.seq2seq import Attention, Encoder, Decoder, Seq2Seq
import torch
import math
from torch import nn, optim
from dataset.translate import TranslateDataset, collate_batch
from torch.utils.data import DataLoader
from functools import partial
import time
import tqdm


def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(tqdm.tqdm(iterator, total=len(iterator))):
        src, src_len, trg = batch
        optimizer.zero_grad()
        output = model(src, src_len, trg)
        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)

        trg = trg[1:].reshape(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src, src_len, trg = batch

            output = model(src, src_len, trg, 0)  # turn off teacher forcing

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].reshape(-1)

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    train_, test = TranslateDataset("data/Corpus/train.json"), TranslateDataset("data/Corpus/test.json")
    print("test", len(test), "train", len(train_))
    collate_fn = partial(collate_batch, vocab_en=train_.vocab_en, vocab_ch=train_.vocab_ch)
    train_loader = DataLoader(train_, batch_size=8, shuffle=True, collate_fn=collate_fn, drop_last=True)
    test_loader = DataLoader(test, batch_size=8, shuffle=True, collate_fn=collate_fn, drop_last=True)
    INPUT_DIM = len(train_.vocab_en)
    OUTPUT_DIM = len(train_.vocab_ch)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    ENC_HID_DIM = 512
    DEC_HID_DIM = 512
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    SRC_PAD_IDX = train_.vocab_en["<pad>"]

    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2Seq(enc, dec, SRC_PAD_IDX, device).to(device)

    def init_weights(m):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)
    model.apply(init_weights)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    PAD_IDX = train_.vocab_ch["<pad>"]
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters())
    N_EPOCHS = 1
    CLIP = 1

    best_valid_loss = float('inf')
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, test_loader, criterion)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut4-model.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    model.load_state_dict(torch.load('tut4-model.pt'))
    test_loss = evaluate(model, test_loader, criterion)
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')


if __name__ == '__main__':
    main()