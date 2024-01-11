import warnings
warnings.filterwarnings("ignore")
import numpy as np
import scipy.stats as stats
import pandas as pd
from sklearn import preprocessing
from dataset.utr import VEE5UTRDataset
from torch.utils.data import DataLoader
import lightning as L
import torch
import random
import tqdm
from models.cnn import LitCNN, MyPrintingCallback, CNN


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)


def evaluate(df, model, test_seq, obs_col, output_col='pred'):
    '''Predict mean ribosome load using model and test set UTRs'''

    # Scale the test set mean ribosome load
    scaler = preprocessing.StandardScaler()
    scaler.fit(df[obs_col].values.reshape(-1, 1))
    model.eval()
    # Make predictions
    test_seq = torch.tensor(test_seq, dtype=torch.float)
    predictions = model(test_seq).reshape(-1, 1).detach().numpy()
    # Inverse scaled predicted mean ribosome load and return in a column labeled 'pred'
    df.loc[:, output_col] = scaler.inverse_transform(predictions)
    return df


def one_hot_encode(df, col='seq', seq_len=44):
    # Dictionary returning one-hot encoding of nucleotides.
    nuc_d = {'a' :[1 ,0 ,0 ,0] ,'c' :[0 ,1 ,0 ,0] ,'g' :[0 ,0 ,1 ,0] ,'t' :[0 ,0 ,0 ,1], 'n' :[0 ,0 ,0 ,0]}

    # Creat empty matrix.
    vectors = np.empty([len(df), seq_len, 4])

    # Iterate through UTRs and one-hot encode
    for i, seq in enumerate(df[col].str[:seq_len]):
        seq = seq.lower()
        a = np.array([nuc_d[x] for x in seq])
        vectors[i] = a
    return vectors


def r2(x,y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    return r_value**2

def foo():
    setup_seed(1337)
    # df = pd.read_csv("./data/VEE_5UTR_0429/VEE_3UTR.csv")
    df = pd.read_csv("../data/VEE_5UTR_0611/VEE-0611.csv")
    plasmid_gate, rna_gate = 30, 5
    df = df[(df["rna_counts"] > rna_gate) & (df["plasmid_counts"] > plasmid_gate)]
    e_train = df.sample(frac=0.8)
    e_test = df[~df.index.isin(e_train.index)]
    seq_e_train = one_hot_encode(e_train, seq_len=44)
    seq_e_test = one_hot_encode(e_test, seq_len=44)

    # Scale the training mean ribosome load values
    e_train.loc[:, 'scaled_rl'] = preprocessing.StandardScaler().fit_transform(
        e_train.loc[:, 'score'].values.reshape(-1, 1))
    train = VEE5UTRDataset(seq_e_train, e_train["scaled_rl"])
    train_loader = DataLoader(train, batch_size=128)
    model = CNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08)
    criterion = torch.nn.MSELoss()
    epochs = 3
    train_epochs_loss = []
    for epoch in range(epochs):
        model.train()
        train_epoch_loss = []
        for idx, (data_x, data_y) in tqdm.tqdm(enumerate(train_loader),total=len(train_loader)):
            data_x = data_x.to(torch.float32)
            data_y = data_y.to(torch.float32)
            outputs = model(data_x)
            optimizer.zero_grad()
            loss = criterion(data_y, outputs)
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())
        train_epochs_loss.append(np.average(train_epoch_loss))
        print("epoch={}/{} of train, loss={}".format(epoch+1, epochs, np.average(train_epoch_loss)))
    model.eval()
    e_test = evaluate(e_test, model, seq_e_test, 'score', output_col='pred')
    r = r2(e_test['pred'], e_test['score'])
    print('r-squared = ', r)


def main():
    setup_seed(1337)
    df = pd.read_csv("./data/VEE_5UTR_0429/VEE_3UTR.csv")
    plasmid_gate, rna_gate = 30, 5
    df = df[(df["rna_counts"] > rna_gate) & (df["plasmid_counts"] > plasmid_gate)]
    e_train = df.sample(frac=0.8)
    e_test = df[~df.index.isin(e_train.index)]
    seq_e_train = one_hot_encode(e_train, seq_len=44)
    seq_e_test = one_hot_encode(e_test, seq_len=44)

    # Scale the training mean ribosome load values
    e_train.loc[:, 'scaled_rl'] = preprocessing.StandardScaler().fit_transform(
        e_train.loc[:, 'score'].values.reshape(-1, 1))
    train = VEE5UTRDataset(seq_e_train, e_train["scaled_rl"])
    train_loader = DataLoader(train, batch_size=128)
    trainer = L.Trainer(limit_train_batches=128, max_epochs=5, callbacks=MyPrintingCallback())
    model = LitCNN()
    trainer.fit(model=model, train_dataloaders=train_loader)
    model.eval()
    e_test = evaluate(e_test, model.model, seq_e_test, 'score', output_col='pred')
    r = r2(e_test['pred'], e_test['score'])
    print('r-squared = ', r)


if __name__ == '__main__':
    # main()
    foo()