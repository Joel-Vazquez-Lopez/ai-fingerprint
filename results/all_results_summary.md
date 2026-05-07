# Consolidated Model Results

This file summarizes the main model results collected so far for the AI Fingerprinting project.

## Best Results

| Experiment | Task | Best model | Accuracy | Macro F1 | Random baseline |
|---|---|---:|---:|---:|---:|
| v1 main dataset | 7-class source classification | MLP classifier | 50.71% | 50.54% | 14.29% |
| v2 AI combined | 5-class AI model attribution | Logistic regression | 47.50% | 47.30% | 20.00% |
| v2 AI TF-IDF | 5-class AI model attribution | Character TF-IDF + Linear SVM | 84.50% | 84.40% | 20.00% |

## v1 Main Dataset

Dataset: `data/processed/train.csv` + `data/processed/test.csv`

Task: classify text into seven labels:

`chatgpt`, `claude`, `deepseek`, `gemini`, `grok`, `human_ai_polished`, `human_original`

Features: sentence embeddings

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| MLP classifier | 50.71% | 50.54% |
| Linear SVM | 48.21% | 48.09% |
| Logistic regression | 47.14% | 46.91% |
| Random forest | 33.21% | 32.37% |

Interpretation: the classifier performs clearly above the 14.29% random baseline, showing that the dataset contains learnable signal. The task remains difficult because it mixes AI model attribution with human-vs-AI detection.

## v2 AI Combined Dataset

Dataset: `data/processed/train_v2_combined.csv` + `data/processed/test_v2.csv`

Task: classify text into five AI model labels:

`chatgpt`, `claude`, `deepseek`, `gemini`, `grok`

Features: sentence embeddings

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Logistic regression | 47.50% | 47.30% |
| Linear SVM | 46.50% | 46.38% |
| MLP classifier | 46.50% | 46.18% |
| Random forest | 34.50% | 34.09% |

Interpretation: the embedding-based AI attribution model is more than double the 20% random baseline, so it captures some model-specific signal. However, modern AI models have overlapping semantic and stylistic behavior, so embeddings alone are limited for this task.

## v2 AI TF-IDF Dataset

Dataset: `data/processed/train_v2_combined.csv` + `data/processed/test_v2.csv`

Task: classify text into five AI model labels:

`chatgpt`, `claude`, `deepseek`, `gemini`, `grok`

Features: character n-gram TF-IDF

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Character TF-IDF + Linear SVM | 84.50% | 84.40% |

Interpretation: this is the strongest result so far. Character n-gram TF-IDF appears to capture local writing patterns, punctuation, formatting habits, phrase choices, and other small fingerprints that are useful for AI model attribution.

## Report-Friendly Summary

The v1 embedding classifier achieved 50.71% accuracy across seven classes, compared with a 14.29% random baseline. This shows that embeddings capture meaningful signal for distinguishing AI-generated, human-written, and AI-polished text, although the task remains challenging.

For AI-only model attribution, the embedding-based v2 classifier achieved 47.50% accuracy across five AI models, compared with a 20% random baseline. This suggests that sentence embeddings capture some model-specific writing patterns, but not enough for highly reliable attribution.

The best AI-only result came from a character n-gram TF-IDF model with Linear SVM, reaching 84.50% accuracy and 84.40% macro F1. This indicates that model attribution depends strongly on local stylistic and formatting patterns, not only on semantic meaning.

## Limitations

These results should not be interpreted as proof of authorship. The classifiers learn patterns from the collected dataset and may fail on heavily edited text, future model versions, different prompt styles, or text from models not included in training. The demo should present predictions as probabilistic signals rather than guaranteed truth.
