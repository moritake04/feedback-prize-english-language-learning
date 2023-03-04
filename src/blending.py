import argparse
import gc

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import torch
import yaml
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from pytorch_lightning import seed_everything
from scipy.optimize import minimize
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding

from regressor import TransformersModel, TransformersRegressorInference


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
    args = parser.parse_args()
    return args


def preprocess_text(data):
    for i, t in enumerate(data["full_text"]):
        data.loc[i, "full_text"] = data.loc[i, "full_text"].replace("\n\n", "[BR]")
        data.loc[i, "full_text"] = data.loc[i, "full_text"].replace("\r\n", "[BR]")
        data.loc[i, "full_text"] = data.loc[i, "full_text"].replace("\n", "[BR]")
        data.loc[i, "full_text"] = data.loc[i, "full_text"].replace("\r", "[BR]")
    return data


def one_fold_valid(skf, cfg, valid_X, valid_y, fold_n):
    print(f"[fold_{fold_n}]")
    seed_everything(cfg["general"]["seed"], workers=True)

    model = TransformersRegressorInference(cfg, f"{cfg['ckpt_path']}.ckpt")
    valid_preds = model.predict(valid_X)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    score, scores = mcrmse(valid_y.values, valid_preds)
    print(f"mcrmse_score:{score}, mcrmse_scores:{scores}")

    return valid_preds, score, scores


def one_fold_test(cfg, test_X, fold_n):
    print(f"[fold_{fold_n}]")
    seed_everything(cfg["general"]["seed"], workers=True)

    model = TransformersRegressorInference(cfg, f"{cfg['ckpt_path']}.ckpt")
    test_preds = model.predict(test_X)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return test_preds


def one_fold_both(skf, cfg, test_X, train_X, train_y, fold_n):
    print(f"[fold_{fold_n}]")
    seed_everything(cfg["general"]["seed"], workers=True)
    _, valid_indices = list(skf.split(train_X, train_y))[fold_n]
    valid_X_cv, valid_y_cv = (
        train_X.iloc[valid_indices].reset_index(drop=True),
        train_y.iloc[valid_indices].reset_index(drop=True),
    )

    model = TransformersRegressorInference(cfg, f"{cfg['ckpt_path']}.ckpt")
    valid_preds = model.predict(valid_X_cv)
    test_preds = model.predict(test_X)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    score, scores = mcrmse(valid_y_cv.values, valid_preds)
    print(f"mcrmse_score:{score}, mcrmse_scores:{scores}")

    return test_preds, valid_preds, score, scores


def all_train_test(cfg, test_X):
    print("[all_train]")

    seed_everything(cfg["general"]["seed"], workers=True)

    model = TransformersRegressorInference(cfg, f"{cfg['ckpt_path']}.ckpt")
    test_preds = model.predict(test_X)

    return test_preds


def main():
    args = get_args()
    fold_list = range(5)
    if args.fold is not None:
        fold_list = [args.fold]

    # random seed setting
    seed_everything(cfg["general"]["seed"], workers=True)

    # read csv
    train = pd.read_csv(f"{args.data_folder_path}/data/input/train.csv")
    test = pd.read_csv(f"{args.data_folder_path}/data/input/test.csv")

    # split X/y
    train_X = train["full_text"]
    train_y = train.drop(["text_id", "full_text"], axis=1)
    test_X = test["full_text"]

    cfg_list = []
    weights_list = [[]] * 6
    skf = MultilabelStratifiedKFold(
        n_splits=5, shuffle=True, random_state=cfg["general"]["seed"]
    )
    for j, fold_n in enumerate(fold_list):
        preds_list = []
        _, valid_indices = list(skf.split(train_X, train_y))[fold_n]
        valid_X_cv, valid_y_cv = (
            train_X.iloc[valid_indices].reset_index(drop=True),
            train_y.iloc[valid_indices].reset_index(drop=True),
        )

        for i, c in enumerate(cfg_list):
            with open(c + ".yaml", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            valid_preds = one_fold_valid(skf, cfg, valid_X_cv, valid_y_cv, fold_n)
            preds_list.append(valid_preds)
            del valid_preds
            gc.collect()

        score, scores = mcrmse(valid_y_cv.values, np.mean(preds_list, axis=0))
        print(f"mcrmse_score:{score}, mcrmse_scores:{scores}")
        for i, p in enumerate(preds_list):
            score, scores = mcrmse(valid_y_cv.values, p)
            print(f"mcrmse_score:{score}, mcrmse_scores:{scores}")

        cohesion = preds_list[:, 0]
        syntax = preds_list[:, 1]
        vocabulary = preds_list[:, 2]
        phraseology = preds_list[:, 3]
        grammar = preds_list[:, 4]
        conventions = preds_list[:, 5]
        target_list = [cohesion, syntax, vocabulary, phraseology, grammar, conventions]

        for t_idx, target in enumerate(target_list):

            def f(x):
                pred = 0
                for i, p in enumerate(target):
                    if i < len(x):
                        pred += p * x[i]
                    else:
                        pred += p * (1 - sum(x))
                score = metrics.mean_squared_error(
                    valid_y_cv.values[:, t_idx], pred, squared=False
                )
                return score

            init_state = [round(1 / len(target), 3) for _ in range(len(target) - 1)]
            result = minimize(f, init_state, method="Nelder-Mead")
            print(f"optimized_corr:{-result['fun']}")

            weights = [0] * len(target)
            for i in range(len(target) - 1):
                weights[i] = result["x"][i]
            weights[len(target) - 1] = 1 - sum(result["x"])
            weights_list[t_idx].append(weights)
            print(f"weights:{weights}")

    avg_weights_cohesion = np.mean(weights_list[0], axis=0)
    avg_weights_syntax = np.mean(weights_list[1], axis=0)
    avg_weights_vocabulary = np.mean(weights_list[2], axis=0)
    avg_weights_phraseology = np.mean(weights_list[3], axis=0)
    avg_weights_grammar = np.mean(weights_list[4], axis=0)
    avg_weights_conventions = np.mean(weights_list[5], axis=0)

    print(f"averaged_weights:{avg_weights_cohesion}")
    print(f"averaged_weights:{avg_weights_syntax}")
    print(f"averaged_weights:{avg_weights_vocabulary}")
    print(f"averaged_weights:{avg_weights_phraseology}")
    print(f"averaged_weights:{avg_weights_grammar}")
    print(f"averaged_weights:{avg_weights_conventions}")


if __name__ == "__main__":
    main()
