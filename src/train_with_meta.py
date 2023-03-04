import argparse
import gc
import re
import string

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import torch
import yaml
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from nltk.corpus import stopwords
from pytorch_lightning import seed_everything
from sklearn.preprocessing import StandardScaler
from spellchecker import SpellChecker
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding

import wandb
from regressor import MLPRegressor, RFRegressor, TransformersModel


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
    parser.add_argument("config_2nd", type=str, help="path to config 2nd (.yaml)")
    parser.add_argument("-f", "--fold", type=int, help="fold")
    args = parser.parse_args()
    return args


def wandb_start(cfg):
    wandb.init(
        project=cfg["general"]["project_name"],
        name=f"{cfg['general']['save_name']}_{cfg['fold_n']}",
        group=f"{cfg['general']['save_name']}_cv" if cfg["general"]["cv"] else "all",
        job_type=cfg["job_type"],
        # mode="disabled" if cfg["general"]["wandb_desabled"] else "online",
        mode="disabled",
        config=cfg,
    )


def preprocess_text(data):
    for i, t in enumerate(data["full_text"]):
        data.loc[i, "full_text"] = data.loc[i, "full_text"].replace("\n\n", "[BR]")
        data.loc[i, "full_text"] = data.loc[i, "full_text"].replace("\r\n", "[BR]")
        data.loc[i, "full_text"] = data.loc[i, "full_text"].replace("\n", "[BR]")
        data.loc[i, "full_text"] = data.loc[i, "full_text"].replace("\r", "[BR]")
    return data


"""
def pos_count(sent):
    nn_count = 0
    pr_count = 0
    vb_count = 0
    jj_count = 0
    uh_count = 0
    cd_count = 0
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    for token in sent:
        if token[1] in ["NN", "NNP", "NNS"]:
            nn_count += 1
        if token[1] in ["PRP", "PRP$"]:
            pr_count += 1
        if token[1] in ["VB", "VBD", "VBG", "VBN"]:
            vb_count += 1
        if token[1] in ["JJ", "JJR", "JJS"]:
            jj_count += 1
        if token[1] in ["UH"]:
            uh_count += 1
        if token[1] in ["CD"]:
            cd_count += 1
    return pd.Series([nn_count, pr_count, vb_count, jj_count, uh_count, cd_count])
"""


def contraction_count(sent):
    count = 0
    count += re.subn(r"won\'t", "", sent)[1]
    count += re.subn(r"can\'t", "", sent)[1]
    count += re.subn(r"\'re", "", sent)[1]
    count += re.subn(r"\'s", "", sent)[1]
    count += re.subn(r"\'d", "", sent)[1]
    count += re.subn(r"\'ll", "", sent)[1]
    count += re.subn(r"\'t", "", sent)[1]
    count += re.subn(r"\'ve", "", sent)[1]
    count += re.subn(r"\'m", "", sent)[1]
    return count


def text_features(df, col):
    df[f"{col}_num_words"] = df[col].apply(
        lambda x: len(str(x).split())
    )  # num_words count

    df[f"{col}_num_unique_words"] = df[col].apply(
        lambda x: len(set(str(x).split()))
    )  # num_unique_words count

    df[f"{col}_num_chars"] = df[col].apply(lambda x: len(str(x)))  # num_chars count

    df[f"{col}_num_stopwords"] = df[col].apply(
        lambda x: len(
            [w for w in str(x).lower().split() if w in stopwords.words("english")]
        )
    )  # stopword count 冠詞

    df[f"{col}_num_punctuations"] = df[col].apply(
        lambda x: len([c for c in str(x) if c in list(string.punctuation)])
    )  # num_punctuations count 句読点

    df[f"{col}_num_words_upper"] = df[col].apply(
        lambda x: len([w for w in str(x) if w.isupper()])
    )  # num_words_upper count 大文字

    df[f"{col}_mean_word_len"] = df[col].apply(
        lambda x: np.mean([len(w) for w in x.split()])
    )  # mean_word_len

    df[f"{col}_num_paragraphs"] = df[col].apply(
        lambda x: len(list(filter(None, re.split("\n\n|\r\n|\r|\n", x))))
    )  # num_paragraphs count

    df[f"{col}_num_contractions"] = df[col].apply(
        contraction_count
    )  # num_contractions count

    """
    df[f"{col}_polarity"] = df[col].apply(
        lambda x: TextBlob(x).sentiment[0]
    )  # TextBlob 感情分析

    df[f"{col}_subjectivity"] = df[col].apply(
        lambda x: TextBlob(x).sentiment[1]
    )  # TextBlob 感情分析

    df[
        [
            f"{col}_nn_count",
            f"{col}_pr_count",
            f"{col}_vb_count",
            f"{col}_jj_count",
            f"{col}_uh_count",
            f"{col}_cd_count",
        ]
    ] = df[col].apply(
        pos_count
    )  # pos count 品詞
    """

    return df


def check_spell(df, col):
    spell = SpellChecker()
    df["misspelled"] = 0
    for idx, text in enumerate(tqdm(df[col])):
        text = re.split(
            '[\n\n|\r\n|\r|\n| |!|#|$|%|&|(|)|*|+|,|-|.|/|:|;|<|=|>|?|@|\|^|_|`|{|}|~|"|\||\[|\]]',
            text,
        )
        misspelled = spell.unknown(text)
        if "" in misspelled:
            misspelled.remove("")
        df.loc[idx, "misspelled"] = len(misspelled)

    return df


def check_error(df, col):
    tl = language_tool_python.LanguageTool("en-US")
    df["GRAMMAR"] = 0  #
    df["TYPOS"] = 0  #
    df["TYPOGRAPHY"] = 0  #
    # df["REDUNDANCY"] = 0
    df["PUNCTUATION"] = 0  #
    # df["STYLE"] = 0
    # df["MISC"] = 0
    df["CASING"] = 0  #
    # df["CONFUSED_WORDS"] = 0
    # df["COLLOCATIONS"] = 0
    # df["NONSTANDARD_PHRASES"] = 0
    # df["BRITISH_ENGLISH"] = 0
    # df["SEMANTICS"] = 0
    # df["COMPOUNDING"] = 0
    # df["AMERICAN_ENGLISH_STYLE"] = 0
    df["cnt_error"] = 0

    use_category = ["GRAMMAR", "TYPOS", "TYPOGRAPHY", "PUNCTUATION", "CASING"]

    for idx, text in enumerate(tqdm(df[col])):
        for error in tl.check(text):
            if error.category in use_category:
                df.loc[idx, error.category] += 1
            df.loc[idx, "cnt_error"] += 1

    return df


class TableDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, X):
        self.cfg = cfg
        self.X = X.values

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X = self._prepare_input(self.X[index])
        return X

    def _prepare_input(self, X):
        if self.cfg["tokenizer_params"]["max_length"] == 512:
            X = self.cfg["tokenizer"].encode_plus(
                X,
                return_tensors=None,
                add_special_tokens=True,
                max_length=self.cfg["tokenizer_params"]["max_length"],
                padding="max_length",
                truncation=True,
            )
        else:
            X = self.cfg["tokenizer"].encode_plus(
                X,
                return_tensors=None,
                add_special_tokens=True,
                # max_length=self.cfg["tokenizer_params"]["max_length"],
                # padding="max_length",
                # truncation=True,
            )
        for k, v in X.items():
            X[k] = torch.tensor(v, dtype=torch.long)
        return X


def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.to("cpu").numpy())
    predictions = np.concatenate(preds)
    return predictions


def train_and_predict(cfg, cfg_2nd, train_X, train_y, valid_X, valid_y):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg["tokenizer"] = AutoTokenizer.from_pretrained(cfg["model_name"])
    train_dataloader = torch.utils.data.DataLoader(
        TableDataset(cfg, train_X["full_text"]),
        collate_fn=DataCollatorWithPadding(
            tokenizer=cfg["tokenizer"],
            padding="longest",
        ),
        **cfg["test_loader"],
    )
    valid_dataloader = torch.utils.data.DataLoader(
        TableDataset(cfg, valid_X["full_text"]),
        collate_fn=DataCollatorWithPadding(
            tokenizer=cfg["tokenizer"],
            padding="longest",
        ),
        **cfg["test_loader"],
    )
    if "base" in cfg["ckpt_path"]:
        model = TransformersModel(cfg).load_from_checkpoint(
            f"{cfg['ckpt_path']}-v1.ckpt", cfg=cfg
        )
    elif "large" in cfg["ckpt_path"]:
        model = TransformersModel(cfg).load_from_checkpoint(
            f"{cfg['ckpt_path']}.ckpt", cfg=cfg
        )
    train_preds = inference_fn(train_dataloader, model, device)
    valid_preds = inference_fn(valid_dataloader, model, device)

    del model, train_dataloader, valid_dataloader
    gc.collect()
    torch.cuda.empty_cache()

    train_X = train_X.drop(["full_text"], axis=1).values
    valid_X = valid_X.drop(["full_text"], axis=1).values

    train_X = np.hstack([train_X, train_preds])
    valid_X = np.hstack([valid_X, valid_preds])
    # train_X = train_preds
    # valid_X = valid_preds

    """
    train_X["cohesion"] = train_preds[:, 0]
    train_X["syntax"] = train_preds[:, 1]
    train_X["vocabulary"] = train_preds[:, 2]
    train_X["phraseology"] = train_preds[:, 3]
    train_X["grammar"] = train_preds[:, 4]
    train_X["conventions"] = train_preds[:, 5]

    valid_X["cohesion"] = valid_preds[:, 0]
    valid_X["syntax"] = valid_preds[:, 1]
    valid_X["vocabulary"] = valid_preds[:, 2]
    valid_X["phraseology"] = valid_preds[:, 3]
    valid_X["grammar"] = valid_preds[:, 4]
    valid_X["conventions"] = valid_preds[:, 5]
    """

    if cfg_2nd["model"] == "rf":
        wandb_start(cfg)
        model = RFRegressor(cfg_2nd, train_X, train_y)
        model.train()
        valid_preds = model.predict(valid_X)
        # wandb.sklearn.plot_feature_importances(model.rf, train_X.columns)
    elif cfg_2nd["model"] == "mlp":
        # normalization
        ss = StandardScaler()
        train_X = ss.fit_transform(train_X)
        valid_X = ss.transform(valid_X)

        model = MLPRegressor(cfg_2nd, train_X, train_y, valid_X, valid_y)
        model.train()
        valid_preds = model.predict(valid_X)

    return valid_preds


def train(cfg, cfg_2nd, train_X, train_y):
    pass


def one_fold(skf, cfg, cfg_2nd, train_X, train_y, fold_n):
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
    valid_preds = train_and_predict(
        cfg, cfg_2nd, train_X_cv, train_y_cv, valid_X_cv, valid_y_cv
    )

    score, scores = mcrmse(valid_y_cv.values, valid_preds)
    print(f"[fold_{fold_n}] finished, mcrmse_score:{score}, mcrmse_scores:{scores}")
    wandb.log({"mcrmse_score": score, "mcrmse_scores": scores})

    torch.cuda.empty_cache()
    wandb.finish()

    return valid_preds, score, scores


def all_train(cfg, cfg_2nd, train_X, train_y):
    print("[all_train] start")

    seed_everything(cfg["general"]["seed"], workers=True)

    # train
    train_and_predict(cfg, cfg_2nd, train_X, train_y)

    return


def main():
    # read config
    args = get_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.fold is not None:
        cfg["general"]["fold"] = [args.fold]
    print(f"fold: {cfg['general']['fold']}")

    # read 2nd config
    with open(args.config_2nd, encoding="utf-8") as f:
        cfg_2nd = yaml.safe_load(f)

    # embedding or predict
    cfg["retrieve_embeddings"] = False

    cfg_2nd["general"]["save_name"] = cfg["general"]["save_name"]
    cfg["job_type"] = "train_meta"
    cfg_2nd["job_type"] = "train_meta"

    # random seed setting
    seed_everything(cfg["general"]["seed"], workers=True)

    # read csv
    train_X = pd.read_csv("../data/input/train_with_meta.csv")
    train_y = pd.read_csv("../data/input/train.csv").drop(
        ["text_id", "full_text"], axis=1
    )

    # preprocess csv
    if cfg["preprocess"]:
        preprocess_text(train)

    # create features
    # train_X = text_features(train, "full_text")
    # train_X = check_spell(train_X, "full_text")
    train_X = train_X.drop(
        [
            "text_id",
            "cohesion",
            "syntax",
            "vocabulary",
            "phraseology",
            "grammar",
            "conventions",
        ],
        axis=1,
    )

    if cfg["general"]["cv"]:
        # multilabel stratified k-fold cross-validation
        skf = MultilabelStratifiedKFold(
            n_splits=5, shuffle=True, random_state=cfg["general"]["seed"]
        )
        valid_mcrmse_score_list = []
        valid_mcrmse_scores_list = []
        for fold_n in tqdm(cfg["general"]["fold"]):
            cfg["fold_n"] = fold_n
            cfg_2nd["fold_n"] = fold_n
            cfg[
                "ckpt_path"
            ] = f"../weights/{cfg['general']['save_name']}/last_epoch_fold{fold_n}"
            valid_preds, score, scores = one_fold(
                skf, cfg, cfg_2nd, train_X, train_y, fold_n
            )
            valid_mcrmse_score_list.append(score)
            valid_mcrmse_scores_list.append(scores)

        valid_mcrmse_score_mean = np.mean(valid_mcrmse_score_list, axis=0)
        valid_mcrmse_scores_mean = np.mean(valid_mcrmse_scores_list, axis=0)
        print(f"cv mean mcrmse score:{valid_mcrmse_score_mean}")
        print(f"cv mean mcrmse score:{valid_mcrmse_scores_mean}")
    else:
        # train all data
        cfg["fold_n"] = "all"
        cfg_2nd["fold_n"] = "all"
        cfg["ckpt_path"] = f"../weights/{cfg['general']['save_name']}/last_epoch"
        all_train(cfg, cfg_2nd, train_X, train_y)


if __name__ == "__main__":
    # nltk.download("punkt")
    # nltk.download("wordnet")
    # nltk.download("stopwords")
    # nltk.download("omw-1.4")
    # nltk.download("averaged_perceptron_tagger")
    main()
