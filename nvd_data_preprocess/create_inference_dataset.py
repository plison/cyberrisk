import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Tuple, Dict

import pandas as pd
import numpy as np

from config import OBSERVATIONS_PATH, PREPROCESSED_PATH, CVE_MENTIONS_PATH, DATA_DIR

def load_cve_mentions(input_file):
    with open(input_file, 'r') as f:
        cve_mentions = json.load(f)
    
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

    #df = pd.merge(
    #    observations, preprocessed, left_on="cve", right_on="cve_id", how="left"
    #)
    df = observations.sort_values("date")
    df = df.rename(columns={"cve": "cve_id"})
    preprocessed = preprocessed.sort_values("published_date")

    return df, preprocessed

def generate_inference_features(
    df: pd.DataFrame,
    preprocessed: pd.DataFrame,
    window_size: int,
    cve_mentions: Dict
) -> pd.DataFrame:
    windows = []
    
    end_date = pd.Timestamp.now(tz='UTC')
    start_date = end_date - pd.Timedelta(days=window_size)

    valid_cves = preprocessed[preprocessed["published_date"] < end_date]["cve_id"].unique()
    
    window = df[(df["date"] >= start_date) & (df["date"] < end_date)]
    
    cve_data = preprocessed[preprocessed["cve_id"].isin(valid_cves)].set_index("cve_id")
    
    for cve in valid_cves:
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
                'mention_frequency': mentions_in_window / window_size
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

def generate_inference_dataset():
    df, preprocessed = load_and_preprocess_data(OBSERVATIONS_PATH, PREPROCESSED_PATH)
    cve_mentions = load_cve_mentions(CVE_MENTIONS_PATH)

    window_size = 30  # Size of window in days

    feature_df = generate_inference_features(
        df, preprocessed, window_size, cve_mentions
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = DATA_DIR / f"inference_features_{timestamp}.csv"
    print(f"Total CVEs processed: {len(feature_df)}")
    feature_df.to_csv(output_path, index=False)
    print(f"Features saved to {output_path}")

if __name__ == "__main__":
    generate_inference_dataset()