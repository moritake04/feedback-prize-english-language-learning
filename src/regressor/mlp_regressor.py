import numpy as np
import pytorch_lightning as pl
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger


def mcrmse(y_trues, y_preds):
    scores = []
    idxes = y_trues.shape[1]
    for i in range(idxes):
        y_true = y_trues[:, i]
        y_pred = y_preds[:, i]
        score = metrics.mean_squared_error(y_true, y_pred, squared=False)  # RMSE
        scores.append(score)
    mcrmse_score = np.mean(scores)
    return mcrmse_score, scores


class MLP(pl.LightningModule):
    def __init__(self, input_size, output_size, cfg):
        super().__init__()
        print(f"input size: {input_size}, output size: {output_size}")
        self.cfg = cfg
        self.criterion = nn.__dict__[cfg["criterion"]]()

        self.network1 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(input_size, 1024), dim=None),
            nn.BatchNorm1d(1024),
            nn.Mish(),
        )
        self.network2 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(1024, 512), dim=None),
            nn.BatchNorm1d(512),
            nn.Mish(),
        )
        self.network3 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(512, 256), dim=None),
            nn.BatchNorm1d(256),
            nn.Mish(),
        )
        self.network4 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(256, 128), dim=None),
            nn.BatchNorm1d(128),
            nn.Mish(),
        )
        self.fc = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.network1(x)
        x = self.network2(x)
        x = self.network3(x)
        x = self.network4(x)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        X, y = batch
        pred_y = self.forward(X)
        loss = self.criterion(pred_y, y)
        return loss

    def training_epoch_end(self, outputs):
        loss_list = [x["loss"] for x in outputs]
        avg_loss = torch.stack(loss_list).mean()
        self.log("train_avg_loss", avg_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        X, y = batch
        pred_y = self.forward(X)
        loss = self.criterion(pred_y, y)
        return {"valid_loss": loss, "preds": pred_y, "targets": y}

    def validation_epoch_end(self, outputs):
        loss_list = [x["valid_loss"] for x in outputs]
        preds = torch.cat([x["preds"] for x in outputs], dim=0).cpu().detach().numpy()
        targets = (
            torch.cat([x["targets"] for x in outputs], dim=0).cpu().detach().numpy()
        )
        avg_loss = torch.stack(loss_list).mean()
        score, scores = mcrmse(targets, preds)
        self.log("valid_avg_loss", avg_loss, prog_bar=True)
        self.log("valid_score", score, prog_bar=True)
        self.log("valid_cohesion", scores[0], prog_bar=True)
        self.log("valid_syntax", scores[1], prog_bar=True)
        self.log("valid_vocabulary", scores[2], prog_bar=True)
        self.log("valid_phraseology	", scores[3], prog_bar=True)
        self.log("valid_grammar", scores[4], prog_bar=True)
        self.log("valid_conventions", scores[5], prog_bar=True)
        return avg_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        X, _ = batch
        pred_y = self.forward(X)
        return pred_y

    def configure_optimizers(self):
        optimizer = optim.__dict__[self.cfg["optimizer"]["name"]](
            self.parameters(), **self.cfg["optimizer"]["params"]
        )
        if self.cfg["scheduler"] is None:
            return [optimizer]
        else:
            if self.cfg["scheduler"]["name"] == "OneCycleLR":
                scheduler = optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    steps_per_epoch=self.cfg["len_train_loader"],
                    **self.cfg["scheduler"]["params"],
                )
                scheduler = {"scheduler": scheduler, "interval": "step"}
            elif self.cfg["scheduler"]["name"] == "ReduceLROnPlateau":
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    **self.cfg["scheduler"]["params"],
                )
                scheduler = {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "monitor": "valid_avg_loss",
                }
            else:
                scheduler = optim.lr_scheduler.__dict__[self.cfg["scheduler"]["name"]](
                    optimizer, **self.cfg["scheduler"]["params"]
                )
            return [optimizer], [scheduler]


class TableDataset(torch.utils.data.Dataset):
    def __init__(self, X, y=None):
        if y is None:
            self.X = X
            self.y = torch.zeros(len(self.X), dtype=torch.float32)
        else:
            self.X = X
            self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        return X, y


class MLPRegressor:
    def __init__(self, cfg, train_X, train_y, valid_X=None, valid_y=None):
        input_size = train_X.shape[1]
        output_size = train_y.shape[1]
        train_X = torch.tensor(train_X, dtype=torch.float32)
        train_y = torch.tensor(train_y.values, dtype=torch.float32)
        self.train_dataset = TableDataset(train_X, train_y)
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            **cfg["train_loader"],
        )
        cfg["len_train_loader"] = len(self.train_dataloader)
        if valid_X is None:
            self.valid_dataloader = None
        else:
            valid_X = torch.tensor(valid_X, dtype=torch.float32)
            valid_y = torch.tensor(valid_y.values, dtype=torch.float32)
            self.valid_dataset = TableDataset(valid_X, valid_y)
            self.valid_dataloader = torch.utils.data.DataLoader(
                self.valid_dataset,
                **cfg["valid_loader"],
            )

        self.callbacks = [pl.callbacks.LearningRateMonitor(logging_interval="step")]

        if cfg["early_stopping"] is not None:
            self.callbacks.append(
                pl.callbacks.EarlyStopping(
                    "valid_avg_loss",
                    patience=cfg["early_stopping"]["patience"],
                )
            )

        if cfg["model_save"]:
            self.callbacks.append(
                pl.callbacks.ModelCheckpoint(
                    dirpath=f"{cfg['save_weight_folder']}",
                    filename=f"last_epoch_fold{cfg['fold_n']}"
                    if cfg["general"]["cv"]
                    else f"last_epoch",
                    save_weights_only=True,
                )
            )

        self.logger = WandbLogger(
            project=cfg["general"]["project_name"],
            name=f"{cfg['general']['save_name']}_{cfg['fold_n']}",
            group=f"{cfg['general']['save_name']}_cv"
            if cfg["general"]["cv"]
            else f"{cfg['general']['save_name']}_all",
            job_type=cfg["job_type"],
            mode="disabled" if cfg["general"]["wandb_desabled"] else "online",
            config=cfg,
        )

        self.model = MLP(input_size, output_size, cfg)
        self.cfg = cfg

        self.trainer = Trainer(
            callbacks=self.callbacks, logger=self.logger, **self.cfg["pl_params"]
        )

    def train(self, weight_path=None):
        if self.valid_dataloader is None:
            self.trainer.fit(
                self.model,
                train_dataloaders=self.train_dataloader,
                ckpt_path=weight_path,
            )
        else:
            self.trainer.fit(
                self.model,
                train_dataloaders=self.train_dataloader,
                val_dataloaders=self.valid_dataloader,
                ckpt_path=weight_path,
            )

    def predict(self, test_X, weight_path=None):
        preds = []
        test_dataset = TableDataset(test_X)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            **self.cfg["test_loader"],
        )
        preds = self.trainer.predict(
            self.model,
            dataloaders=test_dataloader,
            ckpt_path="best" if self.cfg["early_stopping"] is not None else weight_path,
        )
        preds = torch.cat(preds, axis=0)
        preds = preds.cpu().detach().numpy()
        return preds

    def load_weight(self, weight_path):
        self.model.model = self.model.model.load_from_checkpoint(
            checkpoint_path=weight_path,
            cfg=self.cfg,
        )
        print(f"loaded model ({weight_path})")
