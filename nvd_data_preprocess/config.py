
from pathlib import Path

NVD_FILEPATH: Path = Path(__file__).parent.parent.resolve() / "data" / "cve_corpus"
DATA_DIR = Path(__file__).parent.parent / "data"
TRAINING_DIR = DATA_DIR / "training_dir"
NVD_ENRICHED = DATA_DIR / "nvd_data_with_predictions"
PREPROCESS_DIR = DATA_DIR / "preprocessed"

OBSERVATIONS_PATH = DATA_DIR / "cve_observations.csv"
PREPROCESSED_PATH = DATA_DIR / "preprocessed" / "processed_nvd_data_with_predictions.csv"
CVE_MENTIONS_PATH = DATA_DIR / "cve_mentions.json"

CORPUS_FILEPATH = DATA_DIR / "nvd_data_with_predictions"