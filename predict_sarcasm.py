import pickle
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import re

def extract_features(texts):
    features = []
    for text in texts:
        if not isinstance(text, str):
            text = ""

        exclamation_count = text.count("!")
        question_count = text.count("?")
        quote_count = text.count('"') + text.count("'")

        words = text.split()
        word_count = len(words)
        char_count = len(text)
        avg_word_length = char_count / max(word_count, 1)

        punct_ratio = (exclamation_count + question_count) / max(word_count, 1)
        repeated_excl = len(re.findall(r"!{2,}", text))
        repeated_quest = len(re.findall(r"\?{2,}", text))

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

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[^a-z0-9!?'\"]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_model():
    models_dir = Path("./models")
    model_files = list(models_dir.glob("*_model.pkl"))

    if not model_files:
        raise FileNotFoundError("No trained model found in ./models directory")

    # Prefer logreg model (best individual performer on test set)
    logreg_files = [f for f in model_files if "logreg" in f.name]
    if logreg_files:
        model_file = logreg_files[0]
    else:
        # Fallback to ensemble, then any available model
        ensemble_files = [f for f in model_files if "ensemble" in f.name]
        if ensemble_files:
            model_file = ensemble_files[0]
        else:
            model_file = model_files[0]

    with open(model_file, "rb") as f:
        model = pickle.load(f)

    print(f"Loaded model: {model_file.name}")
    return model

def main():
    parser = argparse.ArgumentParser(description="Predict sarcasm in text")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file path")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file path")
    
    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)

    if "text" not in df.columns:
        raise ValueError(f"{args.input} must contain a 'text' column")

    original_texts = df["text"].values
    cleaned_texts = df["text"].apply(clean_text)

    print("Loading model...")
    model = load_model()

    print(f"Generating predictions for {len(df)} samples...")
    labels = model.predict(cleaned_texts.values)

    output_df = pd.DataFrame({
        "text": original_texts,
        "prediction": labels
    })

    output_df.to_csv(args.output, index=False)

    print(f"Predictions saved to {args.output}")

if __name__ == "__main__":
    main()