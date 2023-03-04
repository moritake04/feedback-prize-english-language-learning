import numpy as np
import pytorch_lightning as pl
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


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


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights=None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = (
            nn.Parameter(torch.tensor(layer_weights, dtype=torch.float))
            if layer_weights is not None
            else nn.Parameter(
                torch.tensor(
                    [1] * (num_hidden_layers + 1 - layer_start), dtype=torch.float
                )
            )
        )
        print(self.layer_weights)

    def forward(self, all_hidden_states):
        all_layer_embedding = all_hidden_states[self.layer_start :, :, :, :]
        weight_factor = (
            self.layer_weights.unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(all_layer_embedding.size())
        )
        weighted_average = (weight_factor * all_layer_embedding).sum(
            dim=0
        ) / self.layer_weights.sum()
        return weighted_average


class AttentionHead(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionHead, self).__init__()
        self.att = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, last_hidden_state):
        att_weights = self.att(last_hidden_state)
        feature = torch.sum(att_weights * last_hidden_state, dim=1)
        return feature


class TransformersModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.criterion = nn.__dict__[cfg["criterion"]]()

        # awp
        if cfg["awp"] is not None:
            self.automatic_optimization = False
            self.adv_param = cfg["awp"]["adv_param"]
            self.adv_lr = cfg["awp"]["adv_lr"]
            self.adv_eps = cfg["awp"]["adv_eps"]
            self.adv_step = cfg["awp"]["adv_step"]
            self.backup = {}
            self.backup_eps = {}
            self.awp_accumulate_grad_batches = cfg["awp"]["accumulate_grad_batches"]
            # self.awp_scaler = torch.cuda.amp.GradScaler(enabled=cfg["awp"]["amp"])
            if cfg["awp"]["gradient_clip_val"] is not None:
                self.awp_max_grad_norm = cfg["awp"]["gradient_clip_val"]
            else:
                self.awp_max_grad_norm = None
            self.awp_start_epoch = cfg["awp"]["start_epoch"]

        # model
        if cfg["mlm"]:
            print(f"../weights/mlm_model/{cfg['model_name']}")
            self.tr_config = AutoConfig.from_pretrained(
                f"../weights/mlm_model/{cfg['model_name']}", output_hidden_states=True
            )
        else:
            self.tr_config = AutoConfig.from_pretrained(
                cfg["model_name"], output_hidden_states=True
            )

        self.tr_config.hidden_dropout = 0.0
        self.tr_config.hidden_dropout_prob = 0.0
        self.tr_config.attention_dropout = 0.0
        self.tr_config.attention_probs_dropout_prob = 0.0

        if cfg["mlm"]:
            self.model = AutoModel.from_pretrained(
                f"../weights/mlm_model/{cfg['model_name']}", config=self.tr_config
            )
        elif cfg["pretrained"]:
            self.model = AutoModel.from_pretrained(
                cfg["model_name"], config=self.tr_config
            )
        else:
            self.model = AutoModel(self.tr_config)
        if self.cfg["transformers_params"]["gradient_checkpointing"]:
            self.model.gradient_checkpointing_enable()
        if cfg["preprocess"]:
            self.model.resize_token_embeddings(len(cfg["tokenizer"]))

        # freeze for very large model
        if "deberta-v2-xxlarge" in cfg["model_name"]:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.layer[:24].requires_grad_(False)
        if "deberta-v2-xlarge" in cfg["model_name"]:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.layer[:12].requires_grad_(False)
        if "deberta-xlarge" in cfg["model_name"]:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.layer[:24].requires_grad_(False)

        # header
        if cfg["header"] == "cls":
            self.fc = nn.Linear(self.tr_config.hidden_size, 6)
            self._init_weights(self.fc)
        elif cfg["header"] == "mean_pooling":
            self.pool = MeanPooling()
            self.fc = nn.Linear(self.tr_config.hidden_size, 6)
            self._init_weights(self.fc)
        elif cfg["header"] == "mean_pooling_simple":
            # attention maskを使わない方法
            self.fc = nn.Linear(self.tr_config.hidden_size, 6)
            self._init_weights(self.fc)
        elif cfg["header"] == "attention":
            self.attention = AttentionHead(self.tr_config.hidden_size)
            self.fc = nn.Linear(self.tr_config.hidden_size, 6)
            self._init_weights(self.fc)
        elif cfg["header"] == "weighted_average_pooling":
            layer_start = (
                self.tr_config.num_hidden_layers + 1
            ) - 4  # use last 4-layers
            self.wl_pool = WeightedLayerPooling(
                self.tr_config.num_hidden_layers,
                layer_start=layer_start,
                layer_weights=None,  # [0.1, 0.1, 0.1, 0.7],
            )
            self.m_pool = MeanPooling()
            self.fc = nn.Linear(self.tr_config.hidden_size, 6)
            self._init_weights(self.fc)
        elif cfg["header"] == "weighted_average_pooling_attention":
            layer_start = (
                self.tr_config.num_hidden_layers + 1
            ) - 4  # use last 4-layers
            self.wl_pool = WeightedLayerPooling(
                self.tr_config.num_hidden_layers,
                layer_start=layer_start,
                layer_weights=None,  # [0.1, 0.1, 0.1, 0.7],
            )
            self.attention = AttentionHead(self.tr_config.hidden_size)
            self.fc = nn.Linear(self.tr_config.hidden_size, 6)
            self._init_weights(self.fc)
        elif cfg["header"] == "cls_concatenate":
            self.fc = nn.Linear(self.tr_config.hidden_size, 6)
            self._init_weights(self.fc)
        elif cfg["header"] == "mean_pooling_concatenate":
            self.pool = MeanPooling()
            self.fc = nn.Linear(self.tr_config.hidden_size, 6)
            self._init_weights(self.fc)
        elif cfg["header"] == "1dcnn":
            self.cnn1 = nn.Conv1d(
                self.tr_config.hidden_size, 256, kernel_size=2, padding=1
            )
            self.cnn2 = nn.Conv1d(256, 6, kernel_size=2, padding=1)
            self.fc = nn.Linear(self.tr_config.hidden_size, 6)
            self._init_weights(self.fc)
        elif cfg["header"] == "lstm":
            self.lstm = nn.LSTM(
                self.tr_config.hidden_size,
                self.tr_config.hidden_size,
                batch_first=True,
            )
            self.fc = nn.Linear(self.tr_config.hidden_size, 6)
            self._init_weights(self.fc)
        elif cfg["header"] == "gru":
            self.gru = nn.GRU(
                self.tr_config.hidden_size,
                self.tr_config.hidden_size,
                batch_first=True,
            )
            self.fc = nn.Linear(self.tr_config.hidden_size, 6)
            self._init_weights(self.fc)

        # reinit same layers
        if cfg["transformers_params"]["reinit_layers"] is not None:
            print(f"reinit {cfg['transformers_params']['reinit_layers']} layers")
            self._reinit_layer(self.model)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.tr_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.tr_config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _reinit_layer(self, model):
        if "funnel" in self.cfg["model_name"] and "base" in self.cfg["model_name"]:
            for layer in model.encoder.blocks[
                -self.cfg["transformers_params"]["reinit_layers"] :
            ]:
                for module in layer.modules():
                    if isinstance(module, nn.Linear):
                        module.weight.data.normal_(
                            mean=0.0, std=model.config.initializer_range
                        )
                        if module.bias is not None:
                            module.bias.data.zero_()
                    elif isinstance(module, nn.Embedding):
                        module.weight.data.normal_(
                            mean=0.0, std=model.config.initializer_range
                        )
                        if module.padding_idx is not None:
                            module.weight.data[module.padding_idx].zero_()
                    elif isinstance(module, nn.LayerNorm):
                        module.bias.data.zero_()
                        module.weight.data.fill_(1.0)
        else:
            for layer in model.encoder.layer[
                -self.cfg["transformers_params"]["reinit_layers"] :
            ]:
                for module in layer.modules():
                    if isinstance(module, nn.Linear):
                        module.weight.data.normal_(
                            mean=0.0, std=model.config.initializer_range
                        )
                        if module.bias is not None:
                            module.bias.data.zero_()
                    elif isinstance(module, nn.Embedding):
                        module.weight.data.normal_(
                            mean=0.0, std=model.config.initializer_range
                        )
                        if module.padding_idx is not None:
                            module.weight.data[module.padding_idx].zero_()
                    elif isinstance(module, nn.LayerNorm):
                        module.bias.data.zero_()
                        module.weight.data.fill_(1.0)

    def set_head(self, head):
        # header
        print(f"set head: {head}")
        if head == "cls":
            self.cfg["header"] = "cls"
            self.fc = nn.Linear(self.tr_config.hidden_size, 6)
            self._init_weights(self.fc)
        elif head == "mean_pooling":
            self.cfg["header"] = "mean_pooling"
            self.pool = MeanPooling()
            self.fc = nn.Linear(self.tr_config.hidden_size, 6)
            self._init_weights(self.fc)
        elif head == "mean_pooling_simple":
            self.cfg["header"] = "mean_pooling_simple"
            # attention maskを使わない方法
            self.fc = nn.Linear(self.tr_config.hidden_size, 6)
            self._init_weights(self.fc)
        elif head == "attention":
            self.cfg["header"] = "attention"
            self.attention = AttentionHead(self.tr_config.hidden_size)
            self.fc = nn.Linear(self.tr_config.hidden_size, 6)
            self._init_weights(self.fc)
        elif head == "weighted_average_pooling":
            self.cfg["header"] = "weighted_average_pooling"
            layer_start = (
                self.tr_config.num_hidden_layers + 1
            ) - 4  # use last 4-layers
            self.wl_pool = WeightedLayerPooling(
                self.tr_config.num_hidden_layers,
                layer_start=layer_start,
                layer_weights=None,  # [0.1, 0.1, 0.1, 0.7],
            )
            self.m_pool = MeanPooling()
            self.fc = nn.Linear(self.tr_config.hidden_size, 6)
            self._init_weights(self.fc)
        elif head == "weighted_average_pooling_attention":
            self.cfg["header"] = "weighted_average_pooling_attention"
            layer_start = (
                self.tr_config.num_hidden_layers + 1
            ) - 4  # use last 4-layers
            self.wl_pool = WeightedLayerPooling(
                self.tr_config.num_hidden_layers,
                layer_start=layer_start,
                layer_weights=None,  # [0.1, 0.1, 0.1, 0.7],
            )
            self.attention = AttentionHead(self.tr_config.hidden_size)
            self.fc = nn.Linear(self.tr_config.hidden_size, 6)
            self._init_weights(self.fc)
        elif head == "cls_concatenate":
            self.cfg["header"] = "cls_concatenate"
            self.fc = nn.Linear(self.tr_config.hidden_size, 6)
            self._init_weights(self.fc)
        elif head == "mean_pooling_concatenate":
            self.cfg["header"] = "mean_pooling_concatenate"
            self.pool = MeanPooling()
            self.fc = nn.Linear(self.tr_config.hidden_size, 6)
            self._init_weights(self.fc)
        elif head == "1dcnn":
            self.cfg["header"] = "1dcnn"
            self.cnn1 = nn.Conv1d(
                self.tr_config.hidden_size, 256, kernel_size=2, padding=1
            )
            self.cnn2 = nn.Conv1d(256, 6, kernel_size=2, padding=1)
            self.fc = nn.Linear(self.tr_config.hidden_size, 6)
            self._init_weights(self.fc)
        elif head == "lstm":
            self.cfg["header"] = "lstm"
            self.lstm = nn.LSTM(
                self.tr_config.hidden_size,
                self.tr_config.hidden_size,
                batch_first=True,
            )
            self.fc = nn.Linear(self.tr_config.hidden_size, 6)
            self._init_weights(self.fc)
        elif head == "gru":
            self.cfg["header"] = "gru"
            self.gru = nn.GRU(
                self.tr_config.hidden_size,
                self.tr_config.hidden_size,
                batch_first=True,
            )
            self.fc = nn.Linear(self.tr_config.hidden_size, 6)
            self._init_weights(self.fc)

    def forward(self, inputs):
        outputs = self.model(**inputs)
        if self.cfg["header"] == "cls":
            last_hidden_states = outputs[0][:, 0]
            if self.cfg["retrieve_embeddings"]:
                outputs = F.normalize(outputs, p=2, dim=1)
            else:
                outputs = self.fc(last_hidden_states)
        elif self.cfg["header"] == "mean_pooling":
            last_hidden_states = outputs[0]
            feature = self.pool(last_hidden_states, inputs["attention_mask"])
            if self.cfg["retrieve_embeddings"]:
                outputs = F.normalize(feature, p=2, dim=1)
            else:
                outputs = self.fc(feature)
        elif self.cfg["header"] == "mean_pooling_simple":
            last_hidden_states = outputs[0]
            feature = torch.mean(last_hidden_states, 1)
            if self.cfg["retrieve_embeddings"]:
                outputs = F.normalize(feature, p=2, dim=1)
            else:
                outputs = self.fc(feature)
        elif self.cfg["header"] == "attention":
            last_hidden_states = outputs[0]
            feature = self.attention(last_hidden_states)
            if self.cfg["retrieve_embeddings"]:
                outputs = F.normalize(feature, p=2, dim=1)
            else:
                outputs = self.fc(feature)
        elif self.cfg["header"] == "weighted_average_pooling":
            all_hidden_states = torch.stack(outputs["hidden_states"])
            weighted_pooling_embeddings = self.wl_pool(all_hidden_states)
            feature = self.m_pool(weighted_pooling_embeddings, inputs["attention_mask"])
            if self.cfg["retrieve_embeddings"]:
                outputs = F.normalize(feature, p=2, dim=1)
            else:
                outputs = self.fc(feature)
        elif self.cfg["header"] == "weighted_average_pooling_attention":
            all_hidden_states = torch.stack(outputs["hidden_states"])
            weighted_pooling_embeddings = self.wl_pool(all_hidden_states)
            feature = self.attention(weighted_pooling_embeddings)
            if self.cfg["retrieve_embeddings"]:
                outputs = F.normalize(feature, p=2, dim=1)
            else:
                outputs = self.fc(feature)
        elif self.cfg["header"] == "cls_concatenate":
            feature = torch.cat(
                [outputs["hidden_states"][-1 * i][:, 0] for i in range(1, 4 + 1)], dim=1
            )
            if self.cfg["retrieve_embeddings"]:
                outputs = F.normalize(feature, p=2, dim=1)
            else:
                outputs = self.fc(feature)
        if self.cfg["header"] == "mean_pooling_concatenate":
            features = []
            for i in range(1, 4 + 1):
                last_hidden_states = outputs["hidden_states"][-1 * i]
                feature = self.pool(last_hidden_states, inputs["attention_mask"])
                features.append(feature)
            feature = torch.cat(features, dim=1)
            if self.cfg["retrieve_embeddings"]:
                outputs = F.normalize(feature, p=2, dim=1)
            else:
                outputs = self.fc(feature)
        elif self.cfg["header"] == "1d_cnn":
            last_hidden_states = outputs["last_hidden_state"].permute(0, 2, 1)
            if self.cfg["retrieve_embeddings"]:
                outputs = F.normalize(last_hidden_states, p=2, dim=1)
            else:
                cnn_embeddings = F.relu(self.cnn1(last_hidden_states))
                cnn_embeddings = self.cnn2(cnn_embeddings)
                outputs, _ = torch.max(cnn_embeddings, 2)
        elif self.cfg["header"] == "lstm":
            feature, _ = self.lstm(outputs["last_hidden_state"], None)
            feature = feature[:, -1, :]
            if self.cfg["retrieve_embeddings"]:
                outputs = F.normalize(feature, p=2, dim=1)
            else:
                outputs = self.fc(feature)
        elif self.cfg["header"] == "gru":
            feature, _ = self.gru(outputs["last_hidden_state"], None)
            feature = feature[:, -1, :]
            if self.cfg["retrieve_embeddings"]:
                outputs = F.normalize(feature, p=2, dim=1)
            else:
                outputs = self.fc(feature)
        return outputs

    def collate(self, inputs):
        # 一番長いtokenへ合わせる
        mask_len = int(inputs["attention_mask"].sum(axis=1).max())
        for k, v in inputs.items():
            inputs[k] = inputs[k][:, :mask_len]
        return inputs

    def training_step(self, batch, batch_idx):
        X, y = batch
        X = self.collate(X)
        if self.cfg["awp"] is not None:
            # awp step
            opt = self.optimizers()
            sch = self.lr_schedulers()

            # with torch.cuda.amp.autocast(enabled=self.cfg["awp"]["amp"]):
            pred_y = self.forward(X)
            loss = self.criterion(pred_y, y)

            if self.awp_accumulate_grad_batches > 1:
                loss = loss / self.awp_accumulate_grad_batches
            # self.awp_scaler.scale(loss).backward()
            self.manual_backward(loss)

            if (batch_idx + 1) % self.awp_accumulate_grad_batches == 0:
                if self.trainer.current_epoch >= self.awp_start_epoch:
                    self._awp_save()
                    for _ in range(self.adv_step):
                        self._awp_attack_step()
                        # with torch.cuda.amp.autocast(enabled=self.cfg["awp"]["amp"]):
                        pred_y = self.forward(X)
                        adv_loss = self.criterion(pred_y, y)
                        opt.zero_grad()
                        # self.awp_scaler.scale(adv_loss).backward()
                        self.manual_backward(adv_loss)
                    self._awp_restore()

                # self.awp_scaler.unscale_(opt)
                # torch.nn.utils.clip_grad_norm_(
                #    self.parameters(), self.awp_max_grad_norm
                # )
                # self.awp_scaler.step(opt)
                opt.step()
                # self.awp_scaler.update()
                opt.zero_grad()
                sch.step()
        else:
            # normal step
            pred_y = self.forward(X)
            loss = self.criterion(pred_y, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def training_epoch_end(self, outputs):
        loss_list = [x["loss"] for x in outputs]
        avg_loss = torch.stack(loss_list).mean()
        self.log("train_avg_loss", avg_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        X, y = batch
        X = self.collate(X)
        # if self.cfg["awp"] is not None:
        #    with torch.cuda.amp.autocast(enabled=self.cfg["awp"]["amp"]):
        #        pred_y = self.forward(X)
        #        loss = self.criterion(pred_y, y)
        # else:
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
        X = self.collate(X)
        pred_y = self.forward(X)
        return pred_y

    def get_scheduler(self, optimizer, num_train_steps):
        if self.cfg["transformers_params"]["scheduler"] == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.cfg["transformers_params"]["warmup_ratio"]
                * num_train_steps,
                num_training_steps=num_train_steps,
            )
        elif self.cfg["transformers_params"]["scheduler"] == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.cfg["transformers_params"]["warmup_ratio"]
                * num_train_steps,
                num_training_steps=num_train_steps,
                num_cycles=self.cfg["transformers_params"]["num_cycles"],
            )
        return scheduler

    def configure_optimizers(self):
        # ↓ decayする層を選択 & header (decoder) には別の学習率を設定
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if "model" not in n],
                "lr": self.cfg["transformers_params"]["decoder_lr"],
                "weight_decay": 0.0,
            },
        ]

        if "funnel" in self.cfg["model_name"] and "base" in self.cfg["model_name"]:
            layers = [self.model.embeddings] + list(self.model.encoder.blocks)
        else:
            layers = [self.model.embeddings] + list(self.model.encoder.layer)
        layers.reverse()
        lr = self.cfg["transformers_params"]["encoder_lr"]
        lr_decay = self.cfg["transformers_params"]["lr_decay_final"] ** (
            1.0 / len(layers)
        )
        for layer in layers:
            optimizer_parameters += [
                {
                    "params": [
                        p
                        for n, p in layer.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.cfg["transformers_params"]["weight_decay"],
                    "lr": lr,
                },
                {
                    "params": [
                        p
                        for n, p in layer.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                    "lr": lr,
                },
            ]
            lr *= lr_decay

        optimizer = optim.AdamW(
            optimizer_parameters, lr=self.cfg["transformers_params"]["encoder_lr"],
        )

        if self.cfg["awp"] is None:
            num_train_steps = self.trainer.estimated_stepping_batches
        else:
            num_train_steps = (
                self.trainer.estimated_stepping_batches
                / self.awp_accumulate_grad_batches
            )

        scheduler = self.get_scheduler(optimizer, num_train_steps)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    ###############################################################################
    # awp -------------------------------------------------------------------------
    ###############################################################################
    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        if self.cfg["awp"] is not None and self.awp_max_grad_norm is not None:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=self.awp_max_grad_norm,
                gradient_clip_algorithm=None,
            )

    def _awp_attack_step(self):
        e = 1e-6
        for name, param in self.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]),
                        self.backup_eps[name][1],
                    )
                # param.data.clamp_(*self.backup_eps[name])

    def _awp_save(self):
        for name, param in self.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _awp_restore(self):
        for name, param in self.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


class TableDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, X, y=None):
        self.cfg = cfg
        if y is None:
            self.X = X.values
            self.y = torch.zeros(len(self.X), dtype=torch.float32)
        else:
            self.X = X.values
            self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X = self._prepare_input(self.X[index])
        y = self.y[index]
        return X, y

    def _prepare_input(self, X):
        if self.cfg["token_cut_head_and_tail"]:
            X = self.cut_head_and_tail(X)
        else:
            X = self.cfg["tokenizer"].encode_plus(
                X,
                return_tensors=None,
                add_special_tokens=True,
                max_length=self.cfg["tokenizer_params"]["max_length"],
                padding="max_length",
                truncation=True,
            )
        for k, v in X.items():
            X[k] = torch.tensor(v, dtype=torch.long)
        return X

    def cut_head_and_tail(self, text):
        # まずは限界を設定せずにトークナイズする
        max_len = self.cfg["tokenizer_params"]["max_length"]
        input_ids = self.cfg["tokenizer"].encode(text)
        n_token = len(input_ids)

        # トークン数が最大数と同じ場合
        if n_token == max_len:
            input_ids = input_ids
            attention_mask = [1 for _ in range(max_len)]
            token_type_ids = [1 for _ in range(max_len)]
        # トークン数が最大数より少ない場合
        elif n_token < max_len:
            pad = [1 for _ in range(max_len - n_token)]
            input_ids = input_ids + pad
            attention_mask = [1 if n_token > i else 0 for i in range(max_len)]
            token_type_ids = [1 if n_token > i else 0 for i in range(max_len)]
        # トークン数が最大数より多い場合
        else:
            harf_len = (max_len - 2) // 2
            _input_ids = input_ids[1:-1]
            input_ids = [0] + _input_ids[:harf_len] + _input_ids[-harf_len:] + [2]
            attention_mask = [1 for _ in range(max_len)]
            token_type_ids = [1 for _ in range(max_len)]

        d = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

        return d


class TransformersRegressor:
    def __init__(self, cfg, train_X, train_y, valid_X=None, valid_y=None):
        # tokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
        if cfg["preprocess"]:
            tokenizer.add_tokens("[BR]", special_tokens=True)
        cfg["tokenizer"] = tokenizer

        # create datasets
        self.train_dataset = TableDataset(cfg, train_X, train_y)
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, **cfg["train_loader"],
        )
        if valid_X is None:
            self.valid_dataloader = None
        else:
            self.valid_dataset = TableDataset(cfg, valid_X, valid_y)
            self.valid_dataloader = torch.utils.data.DataLoader(
                self.valid_dataset, **cfg["valid_loader"],
            )

        self.callbacks = [pl.callbacks.LearningRateMonitor(logging_interval="step")]

        if cfg["early_stopping"] is not None:
            self.callbacks.append(
                pl.callbacks.EarlyStopping(
                    "valid_avg_loss", patience=cfg["early_stopping"]["patience"],
                )
            )

        if cfg["model_save"]:
            self.callbacks.append(
                pl.callbacks.ModelCheckpoint(
                    dirpath=f"../weights/{cfg['general']['save_name']}",
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

        self.model = TransformersModel(cfg)
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
        test_dataset = TableDataset(self.cfg, test_X)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, **self.cfg["test_loader"],
        )
        preds = self.trainer.predict(
            self.model, dataloaders=test_dataloader, ckpt_path=weight_path
        )
        preds = torch.cat(preds, axis=0)
        preds = preds.cpu().detach().numpy()
        return preds

    def load_weight(self, weight_path):
        self.model = self.model.load_from_checkpoint(
            checkpoint_path=weight_path, cfg=self.cfg,
        )
        print(f"loaded model ({weight_path})")


class TransformersRegressorInference:
    def __init__(self, cfg, weight_path=None):
        # tokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
        if cfg["preprocess"]:
            tokenizer.add_tokens("[BR]", special_tokens=True)
        cfg["tokenizer"] = tokenizer

        self.model = TransformersModel(cfg)
        self.weight_path = weight_path
        self.cfg = cfg
        self.trainer = Trainer(**self.cfg["pl_params"])

    def predict(self, test_X):
        preds = []
        test_dataset = TableDataset(self.cfg, test_X)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, **self.cfg["test_loader"],
        )
        preds = self.trainer.predict(
            self.model, dataloaders=test_dataloader, ckpt_path=self.weight_path
        )
        preds = torch.cat(preds, axis=0)
        preds = preds.cpu().detach().numpy()
        return preds
