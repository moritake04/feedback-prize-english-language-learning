import cuml
import joblib
from cuml.svm import SVR
from sklearn.multioutput import MultiOutputRegressor


class RapidsSVR:
    def __init__(
        self, cfg, train_X, train_y,
    ):
        self.train_X = train_X
        self.train_y = train_y
        self.cfg = cfg
        self.svr = MultiOutputRegressor(SVR())
        # self.svr = SVR(**self.cfg["rapids_svr_params"])

    def train(self):
        self.svr.fit(self.train_X, self.train_y)
        if self.cfg["model_save"]:
            os.makedirs(f"../weights/{self.cfg['general']['save_name']}", exist_ok=True)
            joblib.dump(
                self.svr,
                f"../weights/{self.cfg['general']['save_name']}/fold{self.cfg['fold_n']}.ckpt",
                compress=3,
            )

    def predict(self, test_X):
        preds = self.svr.predict(test_X)
        return preds


class RapidsSVRInference:
    def __init__(self, cfg, weight_path):
        self.cfg = cfg
        self.svr = joblib.load(weight_path)

    def predict(self, test_X):
        preds = self.svr.predict(test_X)
        return preds
