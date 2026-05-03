import joblib
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

from src.data_preprocessing import load_data, clean_data, split_data
from src.feature_engineering import build_preprocessor
from src.model import get_model
from src.optuna_tuning import tune_model


def train(data_path: str, target: str, use_tuning=True, model_path="models/model.pkl"):

    df = load_data(data_path)
    df = clean_data(df)

    X_train, X_test, y_train, y_test = split_data(df, target)

    preprocessor = build_preprocessor(X_train)

    # Transform data BEFORE Optuna (important for speed + correctness)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 🔥 OPTUNA TUNING
    if use_tuning:
        best_params = tune_model(X_train_processed, y_train, n_trials=30)
    else:
        best_params = None

    # 🧠 MODEL (XGBoost)
    model = get_model(model_type="xgboost", params=best_params)

    model.fit(X_train_processed, y_train)

    preds = model.predict(X_test_processed)

    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, preds)

    print("\n📊 FINAL MODEL PERFORMANCE")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # Save model + preprocessing together
    pipeline = {
        "preprocessor": preprocessor,
        "model": model
    }

    joblib.dump(pipeline, model_path)

    print(f"\n✅ Saved full pipeline to {model_path}")

    return pipeline

