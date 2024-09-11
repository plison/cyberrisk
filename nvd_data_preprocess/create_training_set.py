from pathlib import Path
from datetime import timedelta
from typing import Tuple

import pandas as pd
import numpy as np


def load_and_preprocess_data(
    observations_path: Path, preprocessed_path: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess the observation and preprocessed data.

    Args:
        observations_path (Path): Path to the observations CSV file.
        preprocessed_path (Path): Path to the preprocessed CSV file.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Preprocessed observations and NVD data.
    """
    try:
        observations = pd.read_csv(observations_path)
        preprocessed = pd.read_csv(preprocessed_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Unable to load data files: {e}")

    observations["date"] = pd.to_datetime(observations["date"], utc=True)
    preprocessed["published_date"] = pd.to_datetime(
        preprocessed["published_date"], utc=True
    )
    preprocessed["last_modified_date"] = pd.to_datetime(
        preprocessed["last_modified_date"], utc=True
    )

    df = pd.merge(
        observations, preprocessed, left_on="cve", right_on="cve_id", how="left"
    )
    df = df.sort_values("date")
    preprocessed = preprocessed.sort_values("published_date")

    return df, preprocessed


def generate_time_windows(
    df: pd.DataFrame,
    preprocessed: pd.DataFrame,
    window_size: int,
    prediction_horizon: int,
    stride: int,
) -> pd.DataFrame:
    """
    Generate time windows for feature extraction.

    Args:
        df (pd.DataFrame): Merged observations and preprocessed data.
        preprocessed (pd.DataFrame): Preprocessed NVD data.
        window_size (int): Size of each window in days.
        prediction_horizon (int): Days ahead to predict.
        stride (int): Number of days to move the window forward.

    Returns:
        pd.DataFrame: Feature DataFrame with time windows.
    """
    windows = []
    all_cves = set(preprocessed["cve_id"].unique())

    for start_date in pd.date_range(
        df["date"].min(),
        df["date"].max() - timedelta(days=window_size + prediction_horizon),
        freq=f"{stride}D",
    ):
        end_date = start_date + timedelta(days=window_size)
        target_date = end_date + timedelta(days=prediction_horizon)

        # Get window data
        window = df[(df["date"] >= start_date) & (df["date"] < end_date)]

        if len(window) == 0:
            continue

        # Get CVEs observed in the window
        window_cves = set(window["cve_id"].unique())

        # Get CVEs not observed in the window but published before the window end
        potential_negative_cves = all_cves - window_cves
        potential_negative_cves = [
            cve
            for cve in potential_negative_cves
            if preprocessed[preprocessed["cve_id"] == cve]["published_date"].iloc[0]
            < end_date
        ]

        # Sample an equal number of negative CVEs
        sampled_negative_cves = np.random.choice(
            potential_negative_cves,
            min(len(window_cves), len(potential_negative_cves)),
            replace=False,
        )

        # Prepare data for all samples
        for cve in list(window_cves) + list(sampled_negative_cves):
            cve_data = preprocessed[preprocessed["cve_id"] == cve].iloc[-1].to_dict()
            cve_window = window[window["cve_id"] == cve]

            # Add time window specific features
            cve_data.update(
                {
                    "window_start": start_date,
                    "window_end": end_date,
                    "observations_in_window": len(cve_window),
                    "total_count_in_window": cve_window["count"].sum()
                    if len(cve_window) > 0
                    else 0,
                    "days_since_published": (
                        end_date - cve_data["published_date"]
                    ).days,
                    "days_in_window": min(
                        (end_date - cve_data["published_date"]).days, window_size
                    ),
                    "target": int(
                        df[
                            (df["date"] >= end_date)
                            & (df["date"] < target_date)
                            & (df["cve_id"] == cve)
                        ]["count"].sum()
                        > 0
                    ),
                }
            )

            windows.append(cve_data)

    return pd.DataFrame(windows)


def feature_generation():
    current_dir = Path.cwd()
    grandparent_dir = current_dir.parent.parent
    observations_path = grandparent_dir / "data" / "cve_observations.csv"
    preprocessed_path = (
        grandparent_dir / "preprocessed" / "processed_nvd_data_with_predictions.csv"
    )

    try:
        df, preprocessed = load_and_preprocess_data(
            observations_path, preprocessed_path
        )

        window_size = 30  # Size of each window in days
        prediction_horizon = 30  # Days ahead to predict
        stride = 30

        feature_df = generate_time_windows(
            df, preprocessed, window_size, prediction_horizon, stride
        )

        output_path = grandparent_dir / "data" / "features.csv"
        feature_df.to_csv(output_path, index=False)
        print(f"Features saved to {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    feature_generation()
