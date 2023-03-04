from sklearn.ensemble import RandomForestRegressor


class RFRegressor:
    def __init__(
        self,
        cfg,
        train_X,
        train_y,
    ):
        self.train_X = train_X
        self.train_y = train_y
        self.cfg = cfg
        self.rf = RandomForestRegressor(**self.cfg["randomforest_params"])

    def train(self):
        self.rf.fit(self.train_X, self.train_y)

    def predict(self, test_X):
        preds = self.rf.predict(test_X)
        return preds
