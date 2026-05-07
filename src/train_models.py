"""Train embedding-based classifiers for AI fingerprinting."""

import json
import os

import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC


TRAIN_DATA = "data/processed/train.csv"
TEST_DATA = "data/processed/test.csv"
RESULTS_DIR = "results"
MODELS_DIR = "models"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RANDOM_STATE = 0


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    train = pd.read_csv(TRAIN_DATA)
    test = pd.read_csv(TEST_DATA)

    x_train = train["generated_text"].astype(str).tolist()
    y_train = train["source_label"].values

    x_test = test["generated_text"].astype(str).tolist()
    y_test = test["source_label"].values

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    x_train_encoded = embedding_model.encode(
        x_train,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    x_test_encoded = embedding_model.encode(
        x_test,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    classifiers = {
        "logistic_regression": LogisticRegression(
            max_iter=3000,
            random_state=RANDOM_STATE,
        ),
        "linear_svm": LinearSVC(random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
        ),
        "mlp_classifier": MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=500,
            random_state=RANDOM_STATE,
        ),
    }

    results = []
    best_model_name = None
    best_model = None
    best_macro_f1 = -1

    for model_name, classifier in classifiers.items():
        print(f"\nTraining {model_name}...")
        classifier.fit(x_train_encoded, y_train_encoded)

        y_pred_test = classifier.predict(x_test_encoded)

        acc = accuracy_score(y_test_encoded, y_pred_test)
        macro_f1 = f1_score(y_test_encoded, y_pred_test, average="macro")
        report = classification_report(
            y_test_encoded,
            y_pred_test,
            target_names=label_encoder.classes_,
            zero_division=0,
        )
        confusion = confusion_matrix(y_test_encoded, y_pred_test)

        results.append(
            {
                "model": model_name,
                "accuracy": acc,
                "macro_f1": macro_f1,
            }
        )

        joblib.dump(classifier, f"{MODELS_DIR}/{model_name}.joblib")

        print(report)
        print(confusion)

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_model_name = model_name
            best_model = classifier

    results_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    results_df.to_csv(f"{RESULTS_DIR}/metrics_summary.csv", index=False)

    joblib.dump(best_model, f"{MODELS_DIR}/best_model.joblib")
    joblib.dump(label_encoder, f"{MODELS_DIR}/label_encoder.joblib")

    model_info = {
        "best_model": best_model_name,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "normalize_embeddings": True,
        "labels": label_encoder.classes_.tolist(),
        "best_macro_f1": best_macro_f1,
    }
    with open(f"{MODELS_DIR}/model_info.json", "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2)

    print("\nFinal results:")
    print(results_df)
    print(f"\nBest model: {best_model_name}")
    print(f"Best macro F1: {best_macro_f1:.4f}")
    print("Saved best model to models/best_model.joblib")


if __name__ == "__main__":
    main()
