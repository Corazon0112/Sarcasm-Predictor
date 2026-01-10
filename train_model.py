import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import re

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

RANDOM_STATE = 42

# Text preprocessing
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    # Keep quotes which can indicate sarcasm
    text = re.sub(r"[^a-z0-9!?'\"]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Extract handcrafted features that help detect sarcasm
def extract_features(texts):
    """Extract hand-crafted features that might indicate sarcasm"""
    features = []
    for text in texts:
        if not isinstance(text, str):
            text = ""
        
        # Punctuation features
        exclamation_count = text.count('!')
        question_count = text.count('?')
        quote_count = text.count('"') + text.count("'")
        
        # Length features
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        avg_word_length = char_count / max(word_count, 1)
        
        # Punctuation ratio
        punct_ratio = (exclamation_count + question_count) / max(word_count, 1)
        
        # Repeated punctuation patterns
        repeated_excl = len(re.findall(r'!{2,}', text))
        repeated_quest = len(re.findall(r'\?{2,}', text))
        
        features.append([
            exclamation_count,
            question_count,
            quote_count,
            word_count,
            char_count,
            avg_word_length,
            punct_ratio,
            repeated_excl,
            repeated_quest
        ])
    
    return np.array(features)

# Load data
def load_data():
    train_df = pd.read_csv("./train.csv")
    valid_df = pd.read_csv("./valid.csv")

    # Expected columns: text, label (0/1)
    for df in (train_df, valid_df):
        df["text"] = df["text"].apply(clean_text)

    X_train = train_df["text"].values
    y_train = train_df["label"].values

    X_valid = valid_df["text"].values
    y_valid = valid_df["label"].values

    return X_train, y_train, X_valid, y_valid

# Model pipelines
def build_pipelines():
    # Word-level TF-IDF with more conservative settings to prevent overfitting
    tfidf_word = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,  # Increased to reduce feature space
        max_df=0.9,  # More conservative to filter common words
        sublinear_tf=True,
        analyzer='word'
    )
    
    # Character-level n-grams (captures spelling patterns)
    tfidf_char = TfidfVectorizer(
        ngram_range=(2, 4),
        min_df=3,  # Increased to reduce feature space
        max_df=0.9,
        sublinear_tf=True,
        analyzer='char'
    )
    
    # Handcrafted feature extractor
    feature_extractor = FunctionTransformer(extract_features, validate=False)
    
    # Combine all features
    combined_features = FeatureUnion([
        ('tfidf_word', tfidf_word),
        ('tfidf_char', tfidf_char),
        ('handcrafted', feature_extractor)
    ], n_jobs=1)

    pipelines = {
        "logreg": Pipeline([
            ("features", combined_features),
            ("clf", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                solver="liblinear",
                random_state=RANDOM_STATE,
                penalty='l2',  # L2 regularization
                C=1.0  # Default C, will be tuned
            ))
        ]),
        "svm": Pipeline([
            ("features", combined_features),
            ("clf", LinearSVC(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                max_iter=10000,  # Significantly increased for convergence
                tol=1e-2,  # Looser tolerance to allow convergence
                dual=False  # Use primal form for better convergence with large feature space
            ))
        ]),
        "nb": Pipeline([
            ("features", combined_features),
            ("clf", MultinomialNB(alpha=1.0))  # Explicit regularization
        ])
    }

    param_grids = {
        "logreg": {
            "features__tfidf_word__ngram_range": [(1, 2), (1, 3)],
            "features__tfidf_word__min_df": [3, 4],  # Higher min_df to reduce overfitting
            "clf__C": [0.1, 0.5, 1, 2]  # More conservative C values to prevent overfitting
        },
        "svm": {
            "features__tfidf_word__ngram_range": [(1, 2), (1, 3)],
            "features__tfidf_word__min_df": [3, 4],  # Higher min_df
            "clf__C": [0.5, 1, 2]  # More conservative C values
        },
        "nb": {
            "features__tfidf_word__ngram_range": [(1, 2)],
            "features__tfidf_word__min_df": [3, 4],  # Higher min_df
            "clf__alpha": [1.0, 2.0, 3.0]  # Higher alpha = more regularization
        }
    }

    return pipelines, param_grids

# Training & validation
def train_and_select(X_train, y_train, X_valid, y_valid):
    pipelines, param_grids = build_pipelines()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    models = {}
    results = []

    for name, pipe in pipelines.items():
        print(f"\n{'='*60}")
        print(f"Training {name}...")
        print(f"{'='*60}")
        grid = GridSearchCV(
            pipe,
            param_grids[name],
            scoring="f1",
            cv=skf,
            n_jobs=-1,
            verbose=1
        )
        grid.fit(X_train, y_train)

        print("Best CV params:", grid.best_params_)
        print("Best CV F1:", grid.best_score_)

        # Check for overfitting: compare train vs validation performance
        train_preds = grid.best_estimator_.predict(X_train)
        train_f1 = f1_score(y_train, train_preds)
        train_acc = accuracy_score(y_train, train_preds)
        
        # Validate on held-out validation set
        preds = grid.best_estimator_.predict(X_valid)
        f1 = f1_score(y_valid, preds)
        acc = accuracy_score(y_valid, preds)
        
        # Overfitting check
        train_val_gap_acc = train_acc - acc
        train_val_gap_f1 = train_f1 - f1
        
        print(f"\nTrain Accuracy: {train_acc:.4f}")
        print(f"Validation Accuracy: {acc:.4f}")
        print(f"Train-Validation Gap: {train_val_gap_acc:.4f}")
        print(f"\nTrain F1: {train_f1:.4f}")
        print(f"Validation F1: {f1:.4f}")
        print(f"Train-Validation Gap: {train_val_gap_f1:.4f}")
        
        if train_val_gap_acc > 0.10 or train_val_gap_f1 > 0.10:
            print("⚠️  WARNING: Large train-validation gap detected! Model may be overfitting.")
        elif train_val_gap_acc > 0.05 or train_val_gap_f1 > 0.05:
            print("⚠️  CAUTION: Moderate train-validation gap. Consider more regularization.")
        else:
            print("✓ Good generalization: Train-validation gap is small.")
        
        print("\nClassification Report:")
        print(classification_report(y_valid, preds))

        models[name] = grid.best_estimator_
        results.append({
            'model': name,
            'val_acc': acc,
            'val_f1': f1
        })

    # Find best individual model
    best_name = max(results, key=lambda x: x['val_f1'])['model']
    best_model = models[best_name]
    best_score = max(results, key=lambda x: x['val_f1'])['val_f1']
    
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print(f"\nSelected best model: {best_name} (F1={best_score:.4f})")
    
    return models, best_model, best_name

# Save model
def save_model(model, name):
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)

    with open(models_dir / f"{name}_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to ./models/{name}_model.pkl")

# Create voting ensemble from top models
def create_ensemble(models, X_train, y_train, X_valid, y_valid):
    """Create voting ensemble from available models"""
    print("\n" + "="*60)
    print("Creating Voting Ensemble...")
    print("="*60)
    
    # Select models that support predict_proba for soft voting
    ensemble_models = []
    for name in ["logreg", "nb"]:  # Only use models with predict_proba
        if name in models:
            model = models[name]
            # Check if model supports predict_proba
            if hasattr(model, "predict_proba") or (hasattr(model, "named_steps") and 
                hasattr(model.named_steps.get("clf", None), "predict_proba")):
                ensemble_models.append((name, model))
    
    if len(ensemble_models) < 2:
        print("Not enough models with predict_proba for ensemble, skipping...")
        return None
    
    # Create voting classifier with soft voting
    voting_clf = VotingClassifier(
        estimators=ensemble_models,
        voting='soft'
    )
    
    voting_clf.fit(X_train, y_train)
    
    # Check ensemble for overfitting
    train_preds = voting_clf.predict(X_train)
    train_f1 = f1_score(y_train, train_preds)
    train_acc = accuracy_score(y_train, train_preds)
    
    preds = voting_clf.predict(X_valid)
    f1 = f1_score(y_valid, preds)
    acc = accuracy_score(y_valid, preds)
    
    train_val_gap_acc = train_acc - acc
    train_val_gap_f1 = train_f1 - f1
    
    print(f"Ensemble Train Accuracy: {train_acc:.4f}")
    print(f"Ensemble Validation Accuracy: {acc:.4f}")
    print(f"Train-Validation Gap: {train_val_gap_acc:.4f}")
    print(f"\nEnsemble Train F1: {train_f1:.4f}")
    print(f"Ensemble Validation F1: {f1:.4f}")
    print(f"Train-Validation Gap: {train_val_gap_f1:.4f}")
    
    if train_val_gap_acc > 0.10 or train_val_gap_f1 > 0.10:
        print("⚠️  WARNING: Large train-validation gap detected! Ensemble may be overfitting.")
    elif train_val_gap_acc > 0.05 or train_val_gap_f1 > 0.05:
        print("⚠️  CAUTION: Moderate train-validation gap.")
    else:
        print("✓ Good generalization: Train-validation gap is small.")
    
    print("\nClassification Report:")
    print(classification_report(y_valid, preds))
    
    return voting_clf

# Main
def main():
    X_train, y_train, X_valid, y_valid = load_data()
    models, best_model, best_name = train_and_select(X_train, y_train, X_valid, y_valid)
    
    # Save best individual model
    save_model(best_model, best_name)
    
    # Create and save ensemble
    ensemble_model = create_ensemble(models, X_train, y_train, X_valid, y_valid)
    if ensemble_model:
        save_model(ensemble_model, "ensemble")
        
        # Compare ensemble vs best individual
        ensemble_preds = ensemble_model.predict(X_valid)
        ensemble_acc = accuracy_score(y_valid, ensemble_preds)
        ensemble_f1 = f1_score(y_valid, ensemble_preds)
        
        best_preds = best_model.predict(X_valid)
        best_acc = accuracy_score(y_valid, best_preds)
        best_f1 = f1_score(y_valid, best_preds)
        
        print(f"\n{'='*60}")
        print("FINAL COMPARISON")
        print(f"{'='*60}")
        print(f"Best Individual ({best_name}): Acc={best_acc:.4f}, F1={best_f1:.4f}")
        print(f"Ensemble: Acc={ensemble_acc:.4f}, F1={ensemble_f1:.4f}")
        print(f"\nRecommendation: Use ensemble_model.pkl for best performance")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()