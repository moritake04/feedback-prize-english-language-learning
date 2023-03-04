import argparse
import gc

import joblib
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import torch
import yaml
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from pytorch_lightning import seed_everything
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding

from regressor import (
    RapidsSVR,
    RapidsSVRInference,
    TransformersRegressor,
    TransformersRegressorInference,
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to config (.yaml)")
    parser.add_argument("mode", type=str, help="valid or test or both")
    parser.add_argument("-f", "--fold", type=int, help="fold")
    parser.add_argument(
        "-s",
        "--save_preds",
        action="store_true",
        help="Whether to save the predicted value or not.",
    )
    parser.add_argument("-a", "--amp", action="store_true")
    args = parser.parse_args()
    return args


def preprocess_text(data):
    for i, t in enumerate(data["full_text"]):
        data.loc[i, "full_text"] = data.loc[i, "full_text"].replace("\n\n", "[BR]")
        data.loc[i, "full_text"] = data.loc[i, "full_text"].replace("\r\n", "[BR]")
        data.loc[i, "full_text"] = data.loc[i, "full_text"].replace("\n", "[BR]")
        data.loc[i, "full_text"] = data.loc[i, "full_text"].replace("\r", "[BR]")
    return data


def one_fold_valid(skf, cfg, train_X, train_y, fold_n):
    print(f"[fold_{fold_n}]")
    seed_everything(cfg["general"]["seed"], workers=True)
    _, valid_indices = list(skf.split(train_X, train_y))[fold_n]
    valid_X_cv, valid_y_cv = (
        train_X.iloc[valid_indices].reset_index(drop=True),
        train_y.iloc[valid_indices].reset_index(drop=True),
    )

    if cfg["pretrained_path"] is not None:
        ckpt_path = cfg["pretrained_path"] + f"last_epoch_fold{cfg['fold_n']}.ckpt"
        model = TransformersRegressorInference(cfg, ckpt_path)
    else:
        model = TransformersRegressorInference(cfg)

    emb_valid = model.predict(valid_X_cv)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    model = RapidsSVRInference(cfg, f"{cfg['ckpt_path']}.ckpt")
    valid_preds = model.predict(emb_valid)

    if cfg["save_preds"]:
        print("save_preds!")
        joblib.dump(
            valid_preds,
            f"../data/preds/valid_{cfg['general']['seed']}_{cfg['general']['save_name']}_{fold_n}.preds",
            compress=3,
        )

    score, scores = mcrmse(valid_y_cv.values, valid_preds)
    print(f"mcrmse_score:{score}, mcrmse_scores:{scores}")

    return valid_preds, score, scores


def one_fold_test(cfg, test_X, fold_n):
    print(f"[fold_{fold_n}]")
    seed_everything(cfg["general"]["seed"], workers=True)

    if cfg["pretrained_path"] is not None:
        ckpt_path = cfg["pretrained_path"] + f"last_epoch_fold{cfg['fold_n']}.ckpt"
        model = TransformersRegressorInference(cfg, ckpt_path)
    else:
        model = TransformersRegressorInference(cfg)

    emb_test = model.predict(test_X)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    model = RapidsSVRInference(cfg, f"{cfg['ckpt_path']}.ckpt")
    test_preds = model.predict(emb_test)

    return test_preds


def one_fold_both(skf, cfg, test_X, train_X, train_y, fold_n):
    print(f"[fold_{fold_n}]")
    seed_everything(cfg["general"]["seed"], workers=True)
    _, valid_indices = list(skf.split(train_X, train_y))[fold_n]
    valid_X_cv, valid_y_cv = (
        train_X.iloc[valid_indices].reset_index(drop=True),
        train_y.iloc[valid_indices].reset_index(drop=True),
    )

    if cfg["pretrained_path"] is not None:
        ckpt_path = cfg["pretrained_path"] + f"last_epoch_fold{cfg['fold_n']}.ckpt"
        model = TransformersRegressorInference(cfg, ckpt_path)
    else:
        model = TransformersRegressorInference(cfg)

    emb_valid = model.predict(valid_X_cv)
    emb_test = model.predict(test_X)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    model = RapidsSVRInference(cfg, f"{cfg['ckpt_path']}.ckpt")
    valid_preds = model.predict(emb_valid)
    test_preds = model.predict(emb_test)

    if cfg["save_preds"]:
        print("save_preds!")
        joblib.dump(
            valid_preds,
            f"../data/preds/valid_{cfg['general']['seed']}_{cfg['general']['save_name']}_{fold_n}.preds",
            compress=3,
        )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    score, scores = mcrmse(valid_y_cv.values, valid_preds)
    print(f"mcrmse_score:{score}, mcrmse_scores:{scores}")

    return test_preds, valid_preds, score, scores


def all_train_test(cfg, test_X):
    print("[all_train]")

    seed_everything(cfg["general"]["seed"], workers=True)

    if cfg["pretrained_path"] is not None:
        ckpt_path = cfg["pretrained_path"] + f"last_epoch_fold{cfg['fold_n']}.ckpt"
        model = TransformersRegressorInference(cfg, ckpt_path)
    else:
        model = TransformersRegressorInference(cfg)

    emb_test = model.predict(test_X)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    model = RapidsSVRInference(cfg, f"{cfg['ckpt_path']}.ckpt")
    test_preds = model.predict(emb_test)

    return test_preds


def main():
    # read config
    args = get_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.fold is not None:
        cfg["general"]["fold"] = [args.fold]
    print(f"fold: {cfg['general']['fold']}")
    cfg["mode"] = args.mode
    cfg["save_preds"] = args.save_preds

    if args.amp:
        print("fp16")
        cfg["pl_params"]["precision"] = 16
    else:
        print("fp32")
        cfg["pl_params"]["precision"] = 32

    # random seed setting
    seed_everything(cfg["general"]["seed"], workers=True)

    # read csv
    train = pd.read_csv("../data/input/train.csv")
    test = pd.read_csv("../data/input/test.csv")

    # preprocess csv
    if cfg["preprocess"]:
        preprocess_text(train)

    # split X/y
    train_X = train["full_text"]
    train_y = train.drop(["text_id", "full_text"], axis=1)
    test_X = test["full_text"]

    if cfg["general"]["cv"]:
        if cfg["mode"] == "valid":
            # multilabel stratified k-fold cross-validation
            skf = MultilabelStratifiedKFold(
                n_splits=5, shuffle=True, random_state=cfg["general"]["seed"]
            )
            valid_mcrmse_score_list = []
            valid_mcrmse_scores_list = []
            for fold_n in tqdm(cfg["general"]["fold"]):
                cfg["fold_n"] = fold_n
                """
                cfg[
                    "ckpt_path"
                ] = f"../weights/{cfg['general']['save_name']}/last_epoch_fold{fold_n}"
                """
                cfg[
                    "ckpt_path"
                ] = f"../../weights/{cfg['general']['save_name']}/last_epoch_fold{fold_n}"
                valid_preds, score, scores = one_fold_valid(
                    skf, cfg, train_X, train_y, fold_n
                )
                valid_mcrmse_score_list.append(score)
                valid_mcrmse_scores_list.append(scores)

            valid_mcrmse_score_mean = np.mean(valid_mcrmse_score_list, axis=0)
            valid_mcrmse_scores_mean = np.mean(valid_mcrmse_scores_list, axis=0)
            print(f"cv mean mcrmse score:{valid_mcrmse_score_mean}")
            print(f"cv mean mcrmse score:{valid_mcrmse_scores_mean}")

        elif cfg["mode"] == "test":
            test_preds_list = []
            for fold_n in tqdm(cfg["general"]["fold"]):
                cfg["fold_n"] = fold_n
                cfg[
                    "ckpt_path"
                ] = f"../weights/{cfg['general']['save_name']}/last_epoch_fold{fold_n}"
                test_preds = one_fold_test(cfg, test_X, fold_n)
                test_preds_list.append(test_preds)
                print(test_preds)

            final_test_preds = np.mean(test_preds_list, axis=0)
            print(final_test_preds)

        elif cfg["mode"] == "both":
            # multilabel stratified k-fold cross-validation
            skf = MultilabelStratifiedKFold(
                n_splits=5, shuffle=True, random_state=cfg["general"]["seed"]
            )
            valid_mcrmse_score_list = []
            valid_mcrmse_scores_list = []
            test_preds_list = []
            for fold_n in tqdm(cfg["general"]["fold"]):
                cfg["fold_n"] = fold_n
                cfg[
                    "ckpt_path"
                ] = f"../weights/{cfg['general']['save_name']}/last_epoch_fold{fold_n}"
                test_preds, valid_preds, score, scores = one_fold_both(
                    skf, cfg, test_X, train_X, train_y, fold_n
                )
                valid_mcrmse_score_list.append(score)
                valid_mcrmse_scores_list.append(scores)
                test_preds_list.append(test_preds)
                print(test_preds)

            valid_mcrmse_score_mean = np.mean(valid_mcrmse_score_list, axis=0)
            valid_mcrmse_scores_mean = np.mean(valid_mcrmse_scores_list, axis=0)
            final_test_preds = np.mean(test_preds_list, axis=0)
            print(f"cv mean mcrmse score:{valid_mcrmse_score_mean}")
            print(f"cv mean mcrmse score:{valid_mcrmse_scores_mean}")
            print(final_test_preds)

    else:
        # train all data
        cfg["fold_n"] = "all"
        cfg["ckpt_path"] = f"../weights/{cfg['general']['save_name']}/last_epoch"
        final_test_preds = all_train_test(cfg, test_X)
        print(final_test_preds)


if __name__ == "__main__":
    main()
