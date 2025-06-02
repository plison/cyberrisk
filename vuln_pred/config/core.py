from pathlib import Path

from strictyaml import YAML, load, Map, Seq, Str, Int, Float, Bool

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "features.csv"
TRAINED_MODEL_DIR = Path(__file__).resolve().parent.parent / "trained_models"
CONFIG_FILE_PATH = PROJECT_ROOT / "vuln_pred" / "configs" / "config.yml"


schema = Map({
    "feature_columns": Map({
        "cwe_features": Seq(Str()),
        "log_features": Seq(Str()),
        "other_numeric_features": Seq(Str()),
        "categorical_features": Seq(Str())
    }),
    "hyperparameters": Map({
        "learning_rate": Float(),
        "n_estimators": Int(),
        "max_depth": Int(),
        "min_child_weight": Int(),
        "subsample": Float(),
        "colsample_bytree": Float(),
        "gamma": Float(),
        "reg_alpha": Float(),
        "reg_lambda": Float()
    }),
    "use_predictions": Map({
        "cwe": Bool(),
        "cvss": Bool()
    }),
    "training_data_file": Str(),
    "target": Str(),
    "pipeline_name": Str(),
    "pipeline_save_file": Str(),
    "cutoff_date": Str()
})

def find_config_file() -> Path:
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found in {CONFIG_FILE_PATH}")

def fetch_config_from_yaml() -> YAML:
    cfg_path = find_config_file()
    
    with open(cfg_path, "r") as conf_file:
        parsed_config = load(conf_file.read(), schema)
        return parsed_config.data