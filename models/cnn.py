from torch import nn, optim, utils
import torch
import lightning as L
from lightning.pytorch.callbacks import Callback


class CNN(nn.Module):

    def __init__(self, input_size=44, filters=120, kernel_size=8, layers=3, dropouts=(0, 0, 0.2)):
        super(CNN, self).__init__()
        self.conv = nn.Sequential()
        self.input_size = input_size
        for i in range(layers):
            input_size = filters if i else 4
            self.conv.append(nn.Sequential(
                nn.Conv1d(in_channels=input_size, out_channels=filters, kernel_size=(kernel_size,), padding="same"),
                nn.LeakyReLU(),
                # nn.Dropout(p=dropouts[i])
                ))
        # self.pool = nn.MaxPool1d(kernel_size=(self.input_size,))
        self.dense = nn.Linear(filters*self.input_size, 40)
        self.dropout = nn.Dropout(p=0.2)
        self.output = nn.Linear(40, 1)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                # nn.init.normal_(param.data, mean=0, std=0.01)
                nn.init.kaiming_uniform_(self.dense.weight)
                # nn.init.xavier_uniform_(self.dense.weight, gain=nn.init.calculate_gain('relu'))
            else:
                nn.init.constant_(param.data, 0)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        # x = self.pool(x)
        # x = torch.squeeze(x, dim=2)
        x = self.dense(x)
        x = torch.relu(x)
        # x = self.dropout(x)
        x = self.output(x)
        # x = torch.tanh(x)
        return x


# define the LightningModule
class LitCNN(L.LightningModule):
    def __init__(self,  input_size=44, filters=120, kernel_size=8, layers=3, dropout=(0, 0, 0)):
        super().__init__()
        self.model = CNN(input_size, filters, kernel_size, layers, dropout)
        self.training_step_outputs = []

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        z = self.model(x)
        loss = nn.functional.mse_loss(z, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        self.training_step_outputs.append(loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08)
        return optimizer


class MyPrintingCallback(Callback):

    def on_train_epoch_end(self, trainer, pl_module):
        # do something with all training_step outputs, for example:
        epoch_mean = torch.stack(pl_module.training_step_outputs).mean()
        print("training_epoch_mean", epoch_mean.detach().tolist())
        pl_module.log("training_epoch_mean", epoch_mean)
        # free up the memory
        pl_module.training_step_outputs.clear()


def main():
    x = torch.rand((6, 44, 4))
    model = CNN()
    y= model(x)
    print(y.shape)


if __name__ == '__main__':
    main()