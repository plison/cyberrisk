from preprocessing.data_manager import load_dataset, save_pipeline, split_data_by_date
from pipeline import create_full_pipeline
from config.core import fetch_config_from_yaml
from sklearn.metrics import classification_report
import xgboost as xgb
import pandas as pd



def run_training() -> None:
    data = load_dataset()
    config = fetch_config_from_yaml()
    cutoff_date = pd.Timestamp(config["cutoff_date"]).tz_localize('UTC')
    X_train, X_test = split_data_by_date(data, cutoff_date)
    print(data["window_end"].min(), data["window_end"].max())
    print(X_train.shape, X_test.shape)
    y_train = X_train["target"]
    y_test = X_test["target"]

    classifier = create_full_pipeline(
        xgb.XGBClassifier(**config["hyperparameters"]), config["feature_columns"]
    )
    classifier.fit(X_train, y_train)
    print(classification_report(y_test, classifier.predict(X_test)))
    save_pipeline(pipeline=classifier)


if __name__ == "__main__":
    run_training()
