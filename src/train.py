import argparse

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import torch
import yaml
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from pytorch_lightning import seed_everything
from tqdm import tqdm

import wandb
from regressor import TransformersRegressor


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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to config (.yaml)")
    parser.add_argument("-f", "--fold", type=int, help="fold")
    args = parser.parse_args()
    return args


def wandb_start(cfg):
    wandb.init(
        project=cfg["general"]["project_name"],
        name=f"{cfg['general']['save_name']}_{cfg['fold_n']}",
        group=f"{cfg['general']['save_name']}_cv" if cfg["general"]["cv"] else "all",
        job_type=cfg["job_type"],
        mode="disabled" if cfg["general"]["wandb_desabled"] else "online",
        config=cfg,
    )


def preprocess_text(data):
    for i, t in enumerate(data["full_text"]):
        data.loc[i, "full_text"] = data.loc[i, "full_text"].replace("\n\n", "[BR]")
        data.loc[i, "full_text"] = data.loc[i, "full_text"].replace("\r\n", "[BR]")
        data.loc[i, "full_text"] = data.loc[i, "full_text"].replace("\n", "[BR]")
        data.loc[i, "full_text"] = data.loc[i, "full_text"].replace("\r", "[BR]")
    return data


def train_and_predict(cfg, train_X, train_y, valid_X=None, valid_y=None):

    if cfg["pseudo"]:
        if cfg['fold_n'] == "all":
            ext = pd.read_csv(f"../data/pseudo/pseudo_fold_mean.csv")
            # split X/y
            ext_X = ext["full_text"]
            ext_y = ext.drop(["text_id", "full_text"], axis=1)
        else:
            ext = pd.read_csv(f"../data/pseudo/pseudo_fold_{cfg['fold_n']}.csv")
            # split X/y
            ext_X = ext["full_text"]
            ext_y = ext.drop(["text_id", "full_text"], axis=1)

        print("using pseudo labeling data")
        train_X = pd.concat([train_X, ext_X])
        train_y = pd.concat([train_y, ext_y])

    model = TransformersRegressor(
        cfg, train_X, train_y, valid_X=valid_X, valid_y=valid_y
    )
    
    if cfg["pretrained_path"] is not None:
        head = cfg["header"]
        if head != "mean_pooling":
            model.model.set_head("mean_pooling") # pretrainedモデルは基本mean poolingなので
        if cfg['fold_n'] == "all":
            print(f"using pretrained ({cfg['pretrained_path']})")
            model.load_weight(cfg["pretrained_path"] + f"last_epoch.ckpt")
        else:
            print(f"using pretrained ({cfg['pretrained_path']})")
            model.load_weight(cfg["pretrained_path"] + f"last_epoch_fold{cfg['fold_n']}.ckpt")
        if head != "mean_pooling":
            model.model.set_head(head) # pretrainedモデルは基本mean poolingなので

    model.train()

    if valid_X is None:
        del model
        torch.cuda.empty_cache()
        return
    else:
        valid_preds = model.predict(valid_X)
        del model
        torch.cuda.empty_cache()
        return valid_preds


def one_fold(skf, cfg, train_X, train_y, fold_n):
    print(f"[fold_{fold_n}] start")
    seed_everything(cfg["general"]["seed"], workers=True)
    train_indices, valid_indices = list(skf.split(train_X, train_y))[fold_n]
    train_X_cv, train_y_cv = (
        train_X.iloc[train_indices].reset_index(drop=True),
        train_y.iloc[train_indices].reset_index(drop=True),
    )
    valid_X_cv, valid_y_cv = (
        train_X.iloc[valid_indices].reset_index(drop=True),
        train_y.iloc[valid_indices].reset_index(drop=True),
    )

    # train and valid
    valid_preds = train_and_predict(cfg, train_X_cv, train_y_cv, valid_X_cv, valid_y_cv)

    score, scores = mcrmse(valid_y_cv.values, valid_preds)
    print(f"[fold_{fold_n}] finished, mcrmse_score:{score}, mcrmse_scores:{scores}")
    wandb.log({"mcrmse_score": score, "mcrmse_scores": scores})

    torch.cuda.empty_cache()
    wandb.finish()

    return valid_preds, score, scores


def all_train(cfg, train_X, train_y):
    print("[all_train] start")

    seed_everything(cfg["general"]["seed"], workers=True)

    # train
    train_and_predict(cfg, train_X, train_y)

    return


def main():
    # read config
    args = get_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.fold is not None:
        cfg["general"]["fold"] = [args.fold]
    print(f"fold: {cfg['general']['fold']}")

    # set jobtype for wandb
    cfg["job_type"] = "train"

    # random seed setting
    seed_everything(cfg["general"]["seed"], workers=True)

    # read csv
    train = pd.read_csv("../data/input/train.csv")

    # preprocess csv
    if cfg["preprocess"]:
        preprocess_text(train)

    # split X/y
    train_X = train["full_text"]
    train_y = train.drop(["text_id", "full_text"], axis=1)

    if cfg["general"]["cv"]:
        # multilabel stratified k-fold cross-validation
        skf = MultilabelStratifiedKFold(
            n_splits=5, shuffle=True, random_state=cfg["general"]["seed"]
        )
        valid_mcrmse_score_list = []
        valid_mcrmse_scores_list = []
        for fold_n in tqdm(cfg["general"]["fold"]):
            cfg["fold_n"] = fold_n
            valid_preds, score, scores = one_fold(skf, cfg, train_X, train_y, fold_n)
            valid_mcrmse_score_list.append(score)
            valid_mcrmse_scores_list.append(scores)

        valid_mcrmse_score_mean = np.mean(valid_mcrmse_score_list, axis=0)
        valid_mcrmse_scores_mean = np.mean(valid_mcrmse_scores_list, axis=0)
        print(f"cv mean mcrmse score:{valid_mcrmse_score_mean}")
        print(f"cv mean mcrmse score:{valid_mcrmse_scores_mean}")
    else:
        # train all data
        cfg["fold_n"] = "all"
        all_train(cfg, train_X, train_y)


if __name__ == "__main__":
    main()
