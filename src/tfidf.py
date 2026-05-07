# This file instead of using embedding system will use tfidf as the 
# data similarities lie more on external structures rather than 
# semantics and syntax, so with this will perform better 
# distinguishing between models 


"""
IMPORTS
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import pandas as pd
import os
import joblib

"""
PATHS 
"""

TRAIN_DATA = "data/processed/train_v2.csv"
TEST_DATA = "data/processed/test_v2.csv"
RESULTS_DIR = "results/v2_ai_tfidf"
MODELS_DIR = "models/v2_ai_tfidf"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

"""
TRAIN / TEST SPLIT 
"""
train = pd.read_csv(TRAIN_DATA)
test = pd.read_csv(TEST_DATA)

x_train = train["generated_text"].astype(str)
y_train = train["source_label"]

x_test = test["generated_text"].astype(str)
y_test = test["source_label"]

"""
MODEL 
"""

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=50000,
        lowercase=True
    )),
    ("clf", LinearSVC())
])

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

"""
SCORE AND RESULTS 
"""

acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
report = classification_report(y_test, y_pred, zero_division=0)
confusion = confusion_matrix(y_test, y_pred, labels=model.classes_)

print(report)
print(confusion)

pd.DataFrame([{
    "model": "char_tfidf_linear_svm",
    "accuracy": acc,
    "macro_f1": macro_f1
}]).to_csv(f"{RESULTS_DIR}/metrics_summary.csv", index=False)

joblib.dump(model, f"{MODELS_DIR}/best_model.joblib")

print("\nAccuracy:", acc)
print("Macro F1:", macro_f1)
print("Saved model to:", f"{MODELS_DIR}/best_model.joblib")
print("Labels:", model.classes_)