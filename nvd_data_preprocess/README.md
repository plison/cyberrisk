
# Scripts for preparing the training set

### NVD Data Downloader
A Python script to download and extract CVE data from the National Vulnerability Database.
#### Overview
This script downloads historical CVE data from the NVD in JSON format. It downloads data for each year from 2002 to 2024, extracts the ZIP files, and saves the JSON data for further processing. The data is saved under data/ at the project root. 
Usage:
```python nvd_data_preprocess/fetch_nvd_data.py```

### CVE Data Extractor
This script processes NVD JSON data files to create structured training datasets for CVSS and CWE classifiers in the next step.
#### Purpose
The script serves two main purposes:
1. Extracts CVSS data and associated descriptions and places them under data
2. Extracts CWE classifications and descriptionsand places them under data
Usage:
```python nvd_data_preprocess/training_set_extraction.py```

### Train CVSS and CWE classifiers
This Python script implements a multi-output DistilBERT model for classifying CVSS metrics from vulnerability descriptions. The model uses PyTorch and Hugging Face's transformers library to train on labeled vulnerability data.
Usage:

```python nvd_data_preprocess/cwe_cvss_prediction/cvss_classifier.py```

and
```python nvd_data_preprocess/cwe_cvss_prediction/cwe_classifier.py```
These two commands trains the classifiers based on the description of the vulnerability. The scripts downloads the model from huggingface if not already available.

### Predict CVSS and CWE 
This script processes NVD data to add predicted CWE classifications and CVSS metrics to vulnerability entries. It uses the models trained in the previous step. The script reads JSON files containing NVD data downloaded in the first step, adds the predictions to each CVE entry, and saves the enriched data to a new output folder. Adds them with a predicted_ suffix for each entry.
Usage:
```python nvd_data_preprocess/predict_cvss_cwe_for_nvd.py```


### Preprocess for into tabular format
Usage:
```python nvd_data_preprocess/feature_extraction.py```

This Python script processes CVE  data from JSON files and converts it into a CSV format. It extracts various security-related information including:
- Basic CVE metadata (ID, publication dates)
- CWE listings
- CVSS metrics
- Reference information including tags and domains
- Both original and predicted security metrics

The script filters out rejected CVE entries and organizes all the extracted data into a tabular format, saving it as "processed_nvd_data_with_predictions.csv". The script expects the data from the previous step, with the enriched data under the data/ directory under the project root.


### Generate training set
Generated the features for training the model. The script processes historical CVE observations and metadata to create time-windowed (default 30 days) features for training a machine learning.
Usage:
The script requires the following input files in the data directory:
- cve_observations.csv: Historical CVE observations
- preprocessed/processed_nvd_data_with_predictions.csv: Preprocessed NVD data
- cve_mentions.json: CVE mention data, indicating if the CVE is mentioned in a given day.
```python nvd_data_preprocess/feature_extraction.py```
Output will be saved to data/features.csv with statistics about the generated samples printed to console.


# Create inference dataset
```python nvd_data_preprocess/create_inference_dataset.py```
A script that generates features for CVE vulnerability prediction by processing historical observations, NVD data, and social media mentions. Used for inference.
