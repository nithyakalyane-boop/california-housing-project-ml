import optuna
import numpy as np
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor


def objective(trial, X, y):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "random_state": 42,
        "n_jobs": -1
    }

    model = XGBRegressor(**params)

    scores = cross_val_score(
        model,
        X,
        y,
        cv=3,
        scoring="neg_root_mean_squared_error"
    )

    return -np.mean(scores)


def tune_model(X, y, n_trials=30):
    study = optuna.create_study(direction="minimize")

    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)

    print("\n🔥 Best Parameters:")
    print(study.best_params)

    return study.best_params

