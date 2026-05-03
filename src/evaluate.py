import joblib
from sklearn.metrics import mean_squared_error, r2_score


def evaluate(model_path, X_test, y_test):
    model = joblib.load(model_path)

    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, preds)

    print("\n📊 Evaluation Results")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

