
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from config.core import DATA_PATH, TRAINED_MODEL_DIR

def load_dataset(dataset_path = DATA_PATH):
    df = pd.read_csv(dataset_path)
    date_columns = [
        "window_start",
        "window_end",
        "published_date",
        "last_modified_date",
    ]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])
    return df

def split_data_by_date(df : pd.DataFrame, cutoff_date):
    """Split the data into train and test sets based on a cutoff date."""
    train = df[df["window_end"] < cutoff_date]
    test = df[df["window_end"] >= cutoff_date]
    print(cutoff_date)
    print(train.shape, test.shape)
    return train, test


def save_pipeline(pipeline: Pipeline):
    save_file_name = "vuln_prediction.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name
    joblib.dump(pipeline, save_path)


def load_pipeline(file_name: str):
    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def fill_with_predictions(df, use_predicted_cwe=False, use_predicted_cvss=False):
    df_updated = df.copy()
    
    if use_predicted_cwe:
        df_updated.loc[df_updated['cwe_list'].isin(['[]', '', None]), 'cwe_list'] = \
            df_updated.loc[df_updated['cwe_list'].isin(['[]', '', None]), 'predicted_cwe_list']
    
    if use_predicted_cvss:
        cvss_fields = [
            'vector_string',
            'attack_vector',
            'attack_complexity',
            'privileges_required',
            'user_interaction',
            'scope',
            'confidentiality_impact',
            'integrity_impact',
            'availability_impact',
            'base_score',
            'base_severity',
            'exploitability_score',
            'impact_score'
        ]
        
        for field in cvss_fields:
            mask = df_updated[field].isna() | (df_updated[field] == '')
            df_updated.loc[mask, field] = df_updated.loc[mask, f'predicted_{field}']
    
    return df_updated