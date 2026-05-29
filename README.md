# 🎬 Sentiment Analysis — IMDB Movie Reviews

Classifying movie reviews as positive or negative using NLP techniques and machine learning, targeting F1 ≥ 0.85.

---

## 📌 Overview

Film Junky Union, a community for classic movie enthusiasts, needed an automated system to filter and categorize movie reviews. This project trains multiple NLP models on IMDB reviews labeled with sentiment polarity to detect negative reviews.

**Business requirement:** F1 Score ≥ 0.85 on the test set.

---

## 📊 Dataset

| Feature | Description |
|---|---|
| `review` | Raw review text |
| `pos` | Sentiment label (0 = negative, 1 = positive) |
| `ds_part` | Train/test split indicator |
| `rating` | Numeric rating (1–10) |
| `start_year` | Movie release year |

- **Total records:** 47,331 reviews
- **Train:** 23,796 | **Test:** 23,535
- **Class balance:** Near-perfect (~50% positive / 50% negative)

---

## 🔧 Methodology

### Text Preprocessing
- Lowercasing, HTML tag removal, punctuation and digit stripping
- Two preprocessing pipelines tested:
  - **NLTK stopwords** + TF-IDF
  - **spaCy lemmatization** (en_core_web_sm) + TF-IDF

### Models Trained

| # | Model | Vectorizer | Preprocessing |
|---|---|---|---|
| 0 | Dummy (baseline) | TF-IDF | Lowercased |
| 1 | Logistic Regression | TF-IDF (50k, 1–2 ngrams) | NLTK stopwords |
| 3 | Logistic Regression | TF-IDF (50k) | spaCy lemmatization |
| 4 | LGBMClassifier | TF-IDF | spaCy lemmatization |

---

## 📈 Results

| Model | F1 Score | Meets Threshold (≥ 0.85)? |
|---|---|---|
| Dummy (baseline) | 0.00 | ❌ |
| **Model 1 — LR + NLTK TF-IDF** | **0.888** | ✅ |
| Model 3 — LR + spaCy TF-IDF | 0.875 | ✅ |
| Model 4 — LGBM + spaCy TF-IDF | 0.853 | ✅ |

🏆 **Best model: Model 1** — Logistic Regression with NLTK stopwords and TF-IDF (1–2 ngrams), achieving F1 = 0.888.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green)
![spaCy](https://img.shields.io/badge/spaCy-NLP-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-Boosting-yellow)

- **NLP** — NLTK, spaCy (en_core_web_sm)
- **Vectorization** — TF-IDF (scikit-learn)
- **Models** — Logistic Regression, LGBMClassifier, DummyClassifier
- **Evaluation** — F1 Score, Classification Report, Confusion Matrix

---

## 📁 Project Structure

```
sentiment-analysis-imdb/
│
├── sprint15.ipynb       # Full analysis and modeling
├── README.md
└── .gitignore
```

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/laaurasaporski/sentiment-analysis-imdb.git

# Install dependencies
pip install pandas numpy scikit-learn nltk spacy lightgbm tqdm
python -m spacy download en_core_web_sm

# Open the notebook
jupyter notebook sprint15.ipynb
```

---

## 💡 Key Insights

- **Class balance was ideal** (~50/50 split), eliminating the need for resampling techniques
- **Model 1 (LR + NLTK + bigrams)** outperformed spaCy-based models, showing that simpler preprocessing can outperform heavier NLP pipelines on this task
- **spaCy lemmatization** (Model 3) achieved comparable results with a slightly cleaner vocabulary representation
- **LightGBM** (Model 4) met the threshold but underperformed Logistic Regression — tree-based models are generally less effective with high-dimensional sparse TF-IDF vectors
- All three non-dummy models exceeded the F1 ≥ 0.85 business requirement
