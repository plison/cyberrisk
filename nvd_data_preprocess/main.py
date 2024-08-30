
import argparse

from pathlib import Path

import pandas as pd

from fetch_nvd_data import download_and_process_cve_data
from training_set_extraction import extract_training_set
from feature_extraction import process_cve_entries

NVD_FILEPATH = Path(__file__).parent.parent / "cve_corpus"
DATA_DIR = Path(__file__).parent.parent / "training_dir"
NVD_ENRICHED = Path(__file__).parent.parent / "nvd_data_with_predictions"
PREPROCESS_FILEPATH = Path(__file__).parent.parent / "preprocessed"

def train_cvss_model():
    pass

def train_cwe_model():
    pass

def predict():
    pass

def enrich_corpus(nvd_filepath: Path, output_path: Path):
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "fetch_corpus",
            "prepare_data",
            "train_cvss_model",
            "train_cwe_model",
            "enrich_corpus",
            "preprocess"
        ],
    )
    parser.add_argument("--input", help="Input text for prediction mode")
    return parser.parse_args()

def main():
    args = parse_args()

    try:
        if args.mode == "fetch_corpus":
            download_and_process_cve_data(nvd_filepath=NVD_FILEPATH)
        elif args.mode == "prepare_data":
            extract_training_set(data_dir=DATA_DIR, nvd_filepath=NVD_FILEPATH)
        elif args.mode == "train_cvss_model":
            train_cvss_model(filepath=args.filepath)
        elif args.mode == "train_cwe_model":
            train_cwe_model(filepath=args.filepath)
        elif args.mode == "predict":
            if not args.input:
                raise ValueError("Input is required for predict mode")
            predict(args.input, filepath=args.filepath)
        elif args.mode == "enrich_corpus":
            enrich_corpus(filepath=args.filepath)
        elif args.mode == "preprocess":
            cve_data = process_cve_entries(nvd_filepath=NVD_ENRICHED)
            df = pd.DataFrame(cve_data)
            PREPROCESS_FILEPATH.mkdir(parents=True, exist_ok=True)
            save_filepath = PREPROCESS_FILEPATH / "processed_nvd_data_with_predictions.csv"
            df.to_csv(save_filepath, index=False)
            print(f"Data has been processed and saved to '{save_filepath}'")
        else:
            raise ValueError(f"Invalid mode: {args.mode}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
