import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_iris_data_csv(path="data/iris.csv"):
    df = pd.read_csv(path)

    # Feature columns
    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    X = df[feature_cols]

    # Encode species labels
    le = LabelEncoder()
    y = le.fit_transform(df["species"])

    # Save label mapping (optional, but good practice)
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    return X, y, feature_cols, label_mapping
