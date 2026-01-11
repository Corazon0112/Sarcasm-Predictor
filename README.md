# Sarcasm Detection ML Model

A machine learning system for detecting sarcasm in text using ensemble methods and multi-level feature engineering. This project achieves **86.23% accuracy** and **84.90% F1-score** on sarcasm detection tasks.

## Author
- Tushar Cora Suresh

## Overview

This project implements a sophisticated sarcasm detection system that combines:
- **Multi-level feature extraction** (word-level, character-level, and handcrafted features)
- **Ensemble learning** (Logistic Regression + Naive Bayes with soft voting)
- **Hyperparameter optimization** via GridSearchCV with 5-fold cross-validation
- **Conservative regularization** to prevent overfitting

## Performance

| Metric | Score |
|--------|-------|
| Accuracy | 86.23% |
| Precision | 84.81% |
| Recall | 85.00% |
| F1-Score | 84.90% |

## Features

### Feature Engineering
1. **Word-level TF-IDF**: Unigrams and bigrams to capture phrases like "oh great" and "yeah right"
2. **Character-level TF-IDF**: Character n-grams (2-4) for spelling variations and stylistic patterns
3. **Handcrafted Features**: 9 domain-specific features including:
   - Punctuation counts (!, ?, quotes)
   - Text statistics (word count, character count, average word length)
   - Repeated punctuation patterns (!!!, ???)

### Model Architecture
- **Logistic Regression** with L2 regularization
- **Multinomial Naive Bayes** with additive smoothing
- **Soft Voting Ensemble** combining both models

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sarcasm-detection.git
cd sarcasm-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Required Dependencies
```
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
```

## Usage

### Command Line Interface

Run predictions on a CSV file:

```bash
python predict_sarcasm.py --input test_data.csv --output predictions.csv
```

### Arguments
- `--input`: Path to input CSV file (must contain a 'text' column)
- `--output`: Path to save predictions CSV file

### Input Format
Your input CSV must have at least one column named `text`:

```csv
text
"drone places fresh kill on steps of white house"
"states slow to shut down weak teacher education programs"
```

### Output Format
The output CSV will contain the original text and predictions:

```csv
text,prediction
"drone places fresh kill on steps of white house",1
"states slow to shut down weak teacher education programs",0
```

Where:
- `0` = Not sarcastic
- `1` = Sarcastic

## Project Structure

```
sarcasm-detection/
├── README.md
├── requirements.txt
├── predict_sarcasm.py          # Main inference script
├── report.pdf                  # Detailed project report
├── models/
│   ├── ensemble_model.pkl      # Trained ensemble model
│   └── vectorizer.pkl          # Feature extraction pipeline
└── src/                        # Additional source code (if any)
    └── utils.py
```

## Model Training Details

### Preprocessing
- Lowercase conversion
- URL and mention removal
- Hashtag symbol removal (preserves word)
- **Punctuation preservation** (crucial for sarcasm detection)
- Whitespace normalization

### Hyperparameter Optimization
Optimized using GridSearchCV with 5-fold stratified cross-validation:

**Logistic Regression:**
- `C`: [0.1, 0.5, 1.0, 2.0]
- `ngram_range`: [(1,2), (1,3)]
- `min_df`: [3, 4]

**Naive Bayes:**
- `alpha`: [1.0, 2.0, 3.0]
- `ngram_range`: [(1,2)]
- `min_df`: [3, 4]

### Regularization Techniques
1. **L2 Regularization** in Logistic Regression
2. **Additive Smoothing** in Naive Bayes
3. **Feature Filtering** (min_df=3, max_df=0.9)
4. **Conservative hyperparameters** to prevent overfitting

## Key Findings

### What Works Well
- Multi-level feature engineering captures different aspects of sarcasm
- Character n-grams handle spelling variations and stylistic patterns
- Punctuation features (!!!, ???) are strong sarcasm indicators
- Ensemble methods improve robustness and accuracy
- Bigrams capture sarcastic phrases effectively

### Limitations
- Struggles with deadpan sarcasm lacking obvious markers
- Context-dependent sarcasm requires world knowledge
- Trained on news-style text; may need retraining for other domains

## Ablation Study Results

| Features | Accuracy | F1-Score | Improvement |
|----------|----------|----------|-------------|
| Word TF-IDF only | 84.0% | 0.840 | Baseline |
| + Character TF-IDF | 84.8% | 0.862 | +1.4% |
| + Handcrafted | 86.2% | 0.873 | +1.1% |

## Examples

### Sarcastic Examples
- "drone places fresh kill on steps of white house"
- "oh great, another meeting"
- "yeah right, that'll definitely work"

### Non-Sarcastic Examples
- "states slow to shut down weak teacher education programs"
- "new policy aims to reduce carbon emissions"

## References

1. Joshi, Aditya, et al. "Automatic Sarcasm Detection: A Survey." ArXiv.org, 2016.
2. Karmaker, Subrata. "Sarcasm Detection on Reddit Using Classical Machine Learning and Feature Engineering." ArXiv.org, 2025.

**Note**: This model does not use transformer architectures (BERT, GPT, RoBERTa) as per assignment constraints. It demonstrates that traditional machine learning with thoughtful feature engineering can achieve strong performance on NLP tasks.
