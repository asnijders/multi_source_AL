import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
from src.utils import get_trainer

from pytorch_lightning import Trainer
from torch.optim import Adam
import torchmetrics

"""
This module implements several helper functions for the Discriminative Active Learning (DAL) acquisition function
"""

def get_MLP_trainer(config, logger):

    mode = "max"

    # Init early stopping
    early_stopping_callback = EarlyStopping(monitor="discriminative_train_acc_epoch",
                                            min_delta=0.000,
                                            patience=0,
                                            verbose=True,
                                            mode=mode,
                                            stopping_threshold=0.90)

    # Init ModelCheckpoint callback, monitoring 'config.monitor'
    run_dir = config.checkpoint_dir + '/' + config.array_uid.replace(' ','_') + '/' + config.acquisition_fn + '/' + str(config.seed) + '/'
    checkpoint_callback = ModelCheckpoint(monitor="discriminative_train_acc_epoch",
                                          mode=mode,
                                          save_top_k=1,
                                          dirpath=run_dir,
                                          filename='discriminative-{epoch}-{step}-{discriminative_loss_epoch:.2f}-{discriminative_train_acc_epoch:.2f}',
                                          verbose=True)

    callbacks = [early_stopping_callback, checkpoint_callback]
    epochs = 5

    trainer = Trainer(gpus=config.gpus,
                      strategy=config.strategy,
                      logger=logger,
                      callbacks=callbacks,
                      log_every_n_steps=1,#config.log_every,
                      accelerator=config.accelerator,
                      max_epochs=epochs,
                      deterministic=True,
                      enable_checkpointing=True,
                      enable_model_summary=False,
                      num_sanity_val_steps=0,
                      progress_bar_refresh_rate=1,#config.refresh_rate,
                      enable_progress_bar=True,
                      precision=config.precision)

    return trainer


class EmbeddingPool(Dataset):
    """
    This class serves to provide iterables when seeking to train on learned representations
    """

    def __init__(self, config, dm, model):
        def get_features(encoder, dataloader):
            """
            function for performing inference on labeled and unlabeled data
            takes a model and a dataloader, returns a np array of embeddings
            """
            trainer = get_trainer(config, logger=None)
            predictions = trainer.predict(model, dataloader)
            predictions = torch.cat(predictions, dim=0)
            embeddings = predictions.squeeze(1)
            return embeddings

        # get embeddings for labeled data
        labeled_embeddings = get_features(encoder=model,
                                          dataloader=dm.labelled_dataloader(shuffle=False))
        # get embeddings for unlabeled data
        unlabeled_embeddings = get_features(encoder=model,
                                            dataloader=dm.unlabelled_dataloader())

        # add labels and compile into single train set
        unlabeled_data = [(example, 0) for example in unlabeled_embeddings]
        labeled_data = [(example, 1) for example in labeled_embeddings]
        self.train_data = unlabeled_data + labeled_data
        self.unlabeled_data = unlabeled_data
        self.data = self.train_data

    def set_data(self, split):
        if split == 'all':
            self.data = self.train_data
        elif split == 'U':
            self.data = self.unlabeled_data

    def label_instances(self, indices):
        """
        Takes a list of integer indices;
        iterates over training set and adds label for unlabeled examples
        :param indices: list of integers
        :return: None
        """

        for idx in indices:
            sample = self.train_data[idx]
            example = sample[0]
            label = 1

            self.train_data[idx] = (example, label)

        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        example = self.data[idx]

        embedding = example[0]
        label = example[1]

        sample = {'embedding': embedding,
                  'label': label}

        return sample


class DiscriminativeDataModule(pl.LightningDataModule):

    def __init__(self, config, dm, model, disc_batch_size):
        super().__init__()

        self.config = config
        self.train = EmbeddingPool(config=config,
                                   dm=dm,
                                   model=model)
        self.batch_size = disc_batch_size

    def train_loader(self, shuffle=True):
        self.train.set_data('all')
        return DataLoader(self.train,
                          shuffle=shuffle,
                          batch_size=self.batch_size,
                          num_workers=self.config.num_workers,
                          drop_last=False)

    def unlabeled_loader(self):
        self.train.set_data('U')
        return DataLoader(self.train,
                          shuffle=False,
                          batch_size=self.batch_size,
                          num_workers=self.config.num_workers,
                          drop_last=False)


class DiscriminativeMLP(pl.LightningModule):

    def __init__(self, input_dim, batch_size=64):
        super().__init__()

        self.input_dim = input_dim

        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )

        self.ce = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.batch_size = batch_size

        # hparams
        self.lr = 0.0000001  # learning rate
        self.train_acc = torchmetrics.Accuracy()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):

        # get x, y from batch
        examples = batch['embedding']
        labels = batch['label']

        # forward pass + loss
        outputs = self(examples)
        loss = self.ce(outputs, labels)

        # get preds, compute acc
        preds = self.softmax(outputs)
        acc = self.train_acc(preds, labels)

        # log metrics and return
        metrics = {'discriminative_train_acc': acc, 'discriminative_loss': loss}
        self.log_dict(metrics,
                      batch_size=self.batch_size,
                      on_step=True,
                      on_epoch=True,
                      prog_bar=True,
                      logger=True,
                      sync_dist=False)

        return loss

    def predict_step(self, batch, batch_idx):

        examples = batch['embedding']
        predictions = self.softmax(self(examples))

        return predictions.detach().cpu().numpy()
