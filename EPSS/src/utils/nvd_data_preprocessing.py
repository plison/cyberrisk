import pandas as pd
import json, os, datetime, numbers, csv
import numpy as np

def extract_features():
    nvd_features = extract_features_from_NVD()
    observation_features = extract_features_from_observations()
    features = nvd_features + observation_features


def get_nvd_feature_column_names(features):
    columns = []
    for f, v in features.items():
        if not v:
            continue
        if isinstance(v, list):
            for item in v:
                name = f + "_" + str(item)
                columns.append(name)
        else:
            columns.append(f)
    return columns

def read_json_file(filepath):
    with open(filepath, "r") as f:
        corpus = json.load(f)
    return corpus

def read_NVD_directory(config):
    df = pd.DataFrame()
    columns = ["cve"] + get_nvd_feature_column_names(config.nvd_features)
    column2list = {c: [] for c in columns}
    for f in os.listdir(config.nvd_filepath):
        filepath = config.nvd_filepath + os.sep + f
        print(filepath)
        nvd_dict = read_json_file(filepath)
        for entry in nvd_dict["CVE_Items"]:
            for c in columns:
                column2list[c].append(FEATURE2FUNCTION[c](entry, config))
    for c, l in column2list.items():
        df[c] = l
    return df

def get_missing_nvd_features(df, config):
    print("Checking all selected features are present in NVD feature dataframe.")
    columns = set(df.columns)
    features = set(get_nvd_feature_column_names(config.nvd_features))
    missing = list(features.difference(columns))
    if missing:
        return {m: True for m in missing}
    else:
        return False

    
def add_missing_nvd_features(df, features, config):
    cve2features = {}
    columns = get_nvd_feature_column_names(features)
    column2list = {c: [] for c in columns}
    for f in os.listdir(config.nvd_filepath):
        filepath = config.nvd_filepath + os.sep + f
        print(filepath)
        nvd_dict = read_json_file(filepath)
        for entry in nvd_dict["CVE_Items"]:
            cve = get_cve_id(entry, config)
            cve2features[cve] = {}
            for c in columns:
                cve2features[cve][c] = FEATURE2FUNCTION[c](entry, config)
    ## ensure correct values attributed to correct CVE entry
    for i, r in df.iterrows():
        for c in columns:
            try:
                column2list[c].append(cve2features[r["cve"]][c])
            except:
                data_type = type(column2list[c][-1])
                column2list[c].append(type(""))
    for c, l in column2list.items():
        df[c] = l
    return df

def is_cve_missing(df, config):
    pass

def save_dataframe(df,filepath):
    df.to_csv(filepath, index=False, quoting=csv.QUOTE_NONNUMERIC)
    
def get_age(entry, config):
    try:
        published_date_string = entry["publishedDate"]
    except:
        return -99
    published_date_components = [int(c) for c in published_date_string.split("T")[0].split("-")]
    published_date_time = [int(c) for c in published_date_string[:-1].split("T")[1].split(":")]
    published_date = datetime.datetime(published_date_components[0],
                                       published_date_components[1],
                                       published_date_components[2],
                                       published_date_time[0],
                                       published_date_time[1])
    return (config.date - published_date).days 

def get_cvss_exploitability_score(entry, config):
    if "baseMetricV3" in entry["impact"].keys():
        return entry["impact"]["baseMetricV3"]["exploitabilityScore"]
    if "baseMetricV2" in entry["impact"].keys():
        return entry["impact"]["baseMetricV2"]["exploitabilityScore"]
    else:
        return -99

def get_cvss_impact_score(entry, config):
    if "baseMetricV3" in entry["impact"].keys():
        return entry["impact"]["baseMetricV3"]["impactScore"]
    if "baseMetricV2" in entry["impact"].keys():
        return entry["impact"]["baseMetricV2"]["impactScore"]
    else:
        return -99
    

def get_cve_id(entry, config):
    try:
        return entry["cve"]["CVE_data_meta"]["ID"]
    except:
        return "NVD-CVE-noinfo"

def get_cwe_id(entry, config):
    try:
        return entry["cve"]["problemtype"]["problemtype_data"][0]["description"][0]["value"]
    except:
        return "NVD-CW-noinfo"
    
    
def get_n_references(entry, config):
    try:
        return len(entry["cve"]["references"]["reference_data"])
    except:
        return 0


def get_description(entry, config):
    try:
        return entry["cve"]["description"]["description_data"][0]["value"]
    except:
        return ""
    



FEATURE2FUNCTION = {"cve":  get_cve_id,
                    "age":  get_age,
                    "cwe_id":  get_cwe_id,
                    "cvss_elements_exploitability_score": get_cvss_exploitability_score,
                    "cvss_elements_impact_score": get_cvss_impact_score,
                    "n_references": get_n_references,
                    "description": get_description}
