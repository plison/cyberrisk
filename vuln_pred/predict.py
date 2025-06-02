from typing import Union, Dict
import pandas as pd
from preprocessing.data_manager import load_pipeline, load_dataset
from config.core import PROJECT_ROOT

DATA_PATH = PROJECT_ROOT / "data" / "inference_features.csv"
RESULTS_PATH = PROJECT_ROOT / "data" / "cve_predictions.csv"

def make_prediction(input_data: Union[pd.DataFrame, Dict]) -> pd.DataFrame:
    data = pd.DataFrame(input_data)
    classifier = load_pipeline(file_name="vuln_prediction.pkl")
    probabilities = classifier.predict_proba(data)[:, 1]
    data['exploitation_probability'] = probabilities
    data['percentile'] = data['exploitation_probability'].rank(pct=True)

    return data


def main():
    inference_data = load_dataset(DATA_PATH)
    results = make_prediction(inference_data)
    results_sorted = results.sort_values('exploitation_probability', ascending=False)
    results.to_csv(RESULTS_PATH, index=False)
    
    print(f"\nTotal CVEs analyzed: {len(results)}")
    print("\nTop 10 highest risk CVEs:")
    print(results_sorted[['cve_id', 'exploitation_probability', 'percentile']].head(10))
    
    high_risk = results[results['exploitation_probability'] > 0.9]
    print(f"\nFound {len(high_risk)} high-risk CVEs (p > 0.9)")

if __name__ == "__main__":
    main()
