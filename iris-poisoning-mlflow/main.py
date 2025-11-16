import argparse
from src.utils import load_iris_data_csv
from src.poison import poison_data
from src.train import train_and_log_model
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--poison", type=float, default=0.0)
    args = parser.parse_args()

    # Load from CSV with label encoding
    X, y, feature_cols, label_mapping = load_iris_data_csv("data/iris.csv")

    # Poison dataset
    Xp, yp = poison_data(X, pd.Series(y), args.poison)

    # Train & Log
    train_and_log_model(Xp, yp, args.poison)
