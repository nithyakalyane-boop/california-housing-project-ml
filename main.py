import argparse
import os

from src.train import train


def main():

    parser = argparse.ArgumentParser(description="California Housing ML Pipeline")

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "predict"],
        help="train or predict mode"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/raw/housing.csv",
        help="Path to dataset"
    )

    parser.add_argument(
        "--target",
        type=str,
        default="median_house_value",
        help="Target column name"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="models/model.pkl",
        help="Path to save/load model"
    )

    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)

    if args.mode == "train":
        print("\n🚀 Starting Training Pipeline...\n")

        train(
            data_path=args.data_path,
            target=args.target,
            use_tuning=True,
            model_path=args.model_path
        )

        print("\n✅ Training Complete!\n")

    elif args.mode == "predict":
        print("\n🔮 Prediction mode not implemented yet (next step we can add API/inference)\n")


if __name__ == "__main__":
    main()

