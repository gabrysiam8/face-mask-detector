from pathlib import Path
from typing import Dict, List, Union
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.init as init
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.nn import (Conv2d, CrossEntropyLoss, Linear, MaxPool2d, ReLU, Sequential, Sigmoid)
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor
from mask_dataset import MaskDataset


def init_weights(layer):
    if isinstance(layer, (Linear, Conv2d)):
        init.xavier_uniform_(layer.weight)


class Model(pl.LightningModule):
    def __init__(self, dataframe_path: Path = None):
        super(Model, self).__init__()

        self.dataframe_path = dataframe_path
        self.dataframe = None
        self.trainDF = None
        self.validateDF = None
        self.cross_entropy_loss = None
        self.learning_rate = 1e-3
        
        self.convLayer1 = convLayer1 = Sequential(
            Conv2d(3, 32, kernel_size=(3, 3), padding=(1, 1)),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2))
        )
        
        self.convLayer2 = convLayer2 = Sequential(
            Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2))
        )
        
        self.convLayer3 = convLayer3 = Sequential(
            Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1), stride=(3,3)),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2))
        )
        
        self.linearLayers = linearLayers = Sequential(
            Linear(in_features=2048, out_features=1024),
            Sigmoid(),
            Linear(in_features=1024, out_features=2),
        )

        for sequential in [convLayer1, convLayer2, convLayer3, linearLayers]:
            sequential.apply(init_weights)
    
    def forward(self, x: Tensor):
        out = self.convLayer1(x)
        out = self.convLayer2(out)
        out = self.convLayer3(out)
        out = out.view(-1, 2048)
        out = self.linearLayers(out)
        return out
    
    def prepare_data(self) -> None:
        self.dataframe = dataframe = pd.read_pickle(self.dataframe_path)
        train, validate = train_test_split(dataframe, test_size=0.3, random_state=0, stratify=dataframe['mask'])

        transformations = Compose([
            ToPILImage(),
            Resize((100, 100)),
            ToTensor()
        ])
        self.trainDF = MaskDataset(dataframe=train, transform=transformations)
        self.validateDF = MaskDataset(dataframe=validate, transform=transformations)

        mask_counter = dataframe[dataframe['mask'] == 1].shape[0]
        nonmask_counter = dataframe[dataframe['mask'] == 0].shape[0]
        counter_list = [mask_counter, nonmask_counter]
        weights = [(counter / sum(counter_list)) for counter in counter_list]
        self.cross_entropy_loss = CrossEntropyLoss(weight=torch.tensor(weights))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.trainDF, batch_size=100, num_workers=8)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.validateDF, batch_size=100, num_workers=8)
    
    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=self.learning_rate)
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch['image'], batch['mask']
        labels = labels.flatten()
        outputs = self.forward(inputs)
        loss = self.cross_entropy_loss(outputs, labels)
        return {
            'loss': loss,
            'log': {'train_loss': loss}
        }

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch['image'], batch['mask']
        labels = labels.flatten()
        outputs = self.forward(inputs)
        loss = self.cross_entropy_loss(outputs, labels)
        outputs = torch.argmax(outputs, dim=1)
        val_acc = accuracy_score(outputs.cpu(), labels.cpu())
        return {
            'val_loss': loss,
            'val_acc': torch.tensor(val_acc)
        }

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        return {
            'val_loss': mean_loss,
            'log': {'val_loss': mean_loss, 'val_acc': mean_acc}
        }


if __name__ == '__main__':
    model = Model(Path('data/mask_df.pickle'))
    
    checkpoint_callback = ModelCheckpoint(
        filepath='checkpoints/weights.ckpt',
        save_weights_only=True,
        verbose=True,
        monitor='val_acc',
        mode='max'
    )
    trainer = Trainer(gpus=1 if torch.cuda.is_available() else 0,
                      max_epochs=1,
                      checkpoint_callback=checkpoint_callback)
    trainer.fit(model)
