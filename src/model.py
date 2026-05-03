from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor


def get_model(model_type="xgboost", params=None):

    if model_type == "xgboost":
        default_params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1
        }

        if params:
            default_params.update(params)

        return XGBRegressor(**default_params)

    elif model_type == "rf":
        return RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )

