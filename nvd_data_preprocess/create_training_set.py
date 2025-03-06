import json
from pathlib import Path
from datetime import timedelta
from typing import Tuple, Dict

import pandas as pd
import numpy as np

from config import OBSERVATIONS_PATH, PREPROCESSED_PATH, CVE_MENTIONS_PATH, DATA_DIR

def load_cve_mentions(input_file):
    with open(input_file, 'r') as f:
        cve_mentions = json.load(f)
    
    # Convert string dates back to timezone-naive pandas Timestamp objects
    for cve in cve_mentions:
        cve_mentions[cve] = {pd.Timestamp(date): count 
                             for date, count in cve_mentions[cve].items()}
    
    return cve_mentions

def load_and_preprocess_data(
    observations_path: Path, preprocessed_path: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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


def stratified_temporal_sampling(
   df: pd.DataFrame,
   preprocessed: pd.DataFrame,
   n_negative_per_positive: int = 2,
   severity_weight: bool = False
) -> pd.DataFrame:
   all_cves = set(preprocessed["cve_id"].unique())
   windows = []
   
   df['time_bucket'] = pd.qcut(df['date'], q=10)
   
   for _, bucket_data in df.groupby('time_bucket', observed=True):
       window_cves = set(bucket_data["cve_id"].unique())
       potential_negatives = all_cves - window_cves
       
       if severity_weight:
           weights = preprocessed[preprocessed["cve_id"].isin(potential_negatives)]["base_score"]
       else:
           weights = None
           
       n_samples = len(window_cves) * n_negative_per_positive
       sampled_negatives = np.random.choice(
           list(potential_negatives),
           size=min(n_samples, len(potential_negatives)),
           replace=False,
           p=weights
       )
       
       windows.extend(list(window_cves))
       windows.extend(sampled_negatives)
       
   return preprocessed[preprocessed["cve_id"].isin(windows)]


def generate_time_windows(
    df: pd.DataFrame,
    preprocessed: pd.DataFrame,
    window_size: int,
    prediction_horizon: int,
    stride: int,
    cve_mentions: Dict
) -> pd.DataFrame:
    windows = []
    all_cves = set(preprocessed["cve_id"].unique())

    date_range = pd.date_range(
        df["date"].min(),
        df["date"].max() - pd.Timedelta(days=window_size + prediction_horizon),
        freq=f"{stride}D",
    )

    for start_date in date_range:
        end_date = start_date + pd.Timedelta(days=window_size)
        target_date = end_date + pd.Timedelta(days=prediction_horizon)

        window = df[(df["date"] >= start_date) & (df["date"] < end_date)]

        if len(window) == 0:
            continue

        window_cves = set(window["cve_id"].unique())

        potential_negative_cves = all_cves - window_cves
        potential_negative_cves = preprocessed[
            (preprocessed["cve_id"].isin(potential_negative_cves))
            & (preprocessed["published_date"] < end_date)
        ]["cve_id"].unique()

        sampled_negative_cves = np.random.choice(
            potential_negative_cves,
            min(len(window_cves) * 2, len(potential_negative_cves)),
            replace=False,
        )

        cves_to_process = list(window_cves) + list(sampled_negative_cves)
        cve_data = preprocessed[preprocessed["cve_id"].isin(cves_to_process)].set_index(
            "cve_id"
        )
        for cve in cves_to_process:
            cve_window = window[window["cve_id"] == cve]
            try:
                cve_info = cve_data.loc[cve].to_dict()
                cve_info["cve_id"] = cve
            except KeyError:
                print(f"Warning: CVE {cve} not found in cve_data. Skipping.")
                continue
            observations_in_window = len(cve_window)

            cve_info.update(
                {
                    "window_start": start_date,
                    "window_end": end_date,
                    "observations_in_window": observations_in_window,
                    "obs_buckets": pd.cut(
                        [observations_in_window],
                        bins=[0, 1, 5, 10, np.inf],
                        labels=[0, 1, 2, 3],
                    )[0],
                    "has_observations": int(observations_in_window > 0),
                    "total_count_in_window": cve_window["count"].sum()
                    if len(cve_window) > 0
                    else 0,
                    "days_since_published": (
                        end_date - cve_info["published_date"]
                    ).days,
                    "days_in_window": min(
                        (end_date - cve_info["published_date"]).days, window_size
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
            if cve in cve_mentions:
                mentions_in_window = sum(count for date, count in cve_mentions[cve].items() 
                                         if start_date <= date < end_date)
                total_mentions = sum(cve_mentions[cve].values())
                mention_dates = [date for date in cve_mentions[cve].keys() if date < end_date]
                days_since_last_mention = (end_date - max(mention_dates)).days if mention_dates else None

                cve_info.update({
                    'mentions_in_window': mentions_in_window,
                    'total_mentions': total_mentions,
                    'days_since_last_mention': days_since_last_mention,
                    'mention_frequency': mentions_in_window / window_size  # mentions per day in the window
                })
            else:
                cve_info.update({
                    'mentions_in_window': 0,
                    'total_mentions': 0,
                    'days_since_last_mention': None,
                    'mention_frequency': 0
                })

            windows.append(cve_info)

    result_df = pd.DataFrame(windows)

    return result_df


def feature_generation():
    df, preprocessed = load_and_preprocess_data(OBSERVATIONS_PATH, PREPROCESSED_PATH)
    cve_mentions = load_cve_mentions(CVE_MENTIONS_PATH)

    window_size = 30  # Size of each window in days
    prediction_horizon = 30  # Days ahead to predict
    stride = 7

    sampled_preprocessed = stratified_temporal_sampling(
        df, 
        preprocessed,
        n_negative_per_positive=2,
        severity_weight=False
    )

    feature_df = generate_time_windows(
        df, 
        sampled_preprocessed,
        window_size, 
        prediction_horizon, 
        stride, 
        cve_mentions
    )

    output_path = DATA_DIR / "features.csv"
    print(f"Total samples: {len(feature_df)}")
    print(f"Positive samples: {feature_df['target'].sum()}")
    print(f"Negative samples: {len(feature_df) - feature_df['target'].sum()}")
    print(f"Positive ratio: {feature_df['target'].mean():.2%}")
    feature_df.to_csv(output_path)
    print(f"Features saved to {output_path}")


if __name__ == "__main__":
    feature_generation()
