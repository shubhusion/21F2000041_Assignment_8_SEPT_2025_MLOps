import os
import json
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from src.plot import save_confusion_matrix


def train_and_log_model(X, y, poison_fraction, label_mapping=None):
    """
    Trains a RandomForest model on the provided dataset
    and logs metrics, artifacts, and the trained model to MLflow.

    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series or np.array): Labels
        poison_fraction (float): Fraction of poisoned samples (for logging)
        label_mapping (dict): Mapping from string labels to encoded integers
    """

    # Create or set experiment
    mlflow.set_experiment("Iris_Poisoning_Experiment")

    # Ensure y is a Pandas Series
    try:
        y = y.reset_index(drop=True)
    except:
        pass

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    with mlflow.start_run(run_name=f"poison_{poison_fraction}"):

        # --- Log Parameters ---
        params = {
            "model_type": "RandomForestClassifier",
            "n_estimators": 100,
            "random_state": 42,
            "poison_fraction": poison_fraction
        }
        mlflow.log_params(params)

        # Log label mapping if available
        if label_mapping:
            mlflow.log_dict(label_mapping, "label_mapping.json")

        # --- Train Model ---
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        model.fit(X_train, y_train)

        # --- Predictions ---
        preds = model.predict(X_val)

        # --- Metrics ---
        accuracy = accuracy_score(y_val, preds)
        cm = confusion_matrix(y_val, preds)
        report = classification_report(y_val, preds, output_dict=True)

        mlflow.log_metric("val_accuracy", float(accuracy))

        # --- Artifacts Directory ---
        os.makedirs("artifacts", exist_ok=True)

        # Save confusion matrix
        cm_path = "artifacts/confusion_matrix.png"
        save_confusion_matrix(cm, labels=list(range(len(report) - 3)), out_path=cm_path)
        mlflow.log_artifact(cm_path)

        # Save detailed classification report
        report_path = "artifacts/classification_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        mlflow.log_artifact(report_path)

        # Save model
        mlflow.sklearn.log_model(model, "model")

        # --- Print Summary ---
        print("=" * 60)
        print(f" MLflow Run Logged Successfully")
        print(f" Poison Fraction  : {poison_fraction}")
        print(f" Validation Acc   : {accuracy:.4f}")
        print("=" * 60)
