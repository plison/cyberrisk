## General

Code for the detection of threat event scenario components with two approaches: 
- a simpler, keyword-based approach relying on [KeyBERT](https://maartengr.github.io/KeyBERT/index.html) and 
- a Named Entity Recognition (NER) based approach that produces categorized information relevant to threat scenarios (trained on the [CASIE](https://github.com/Ebiquity/CASIE) corpus.

## Modules

- get_keywords.py: extracts keywords including some post-processing (e.g. removing similar ones or duplicates), saves them  in a CSV file and counts their occurrence in the source texts.
- preprocess_casie.py: extracts entity annotations from CASIE and create an dataset usable to train a Spacy NER model
- test_detection.py: apply a NER model trained with CASIE annotations on any text

## Dependencies

- keybert 0.7.0
- spacy 3.5.3
- spacy_fastlang 1.0.1
- Levenshtein 0.20.9
- pandas 1.5.3
- Python 3.10.9

(You have to run `python -m spacy download en_core_web_sm` for downloading the spacy model required if you don't already have it.) 

These packages should be enough for the modules included here, but there is also a pip-generated requirements.txt file for replicating the exact environment used (contains some other packages too used for additional tests not included here).

## Keyword detection

- Example call for running keyword extraction:

`python get_keywords.py -data_dir sciodump.zip -start_date 2023-03-01 -end_date 2023-04-01`

- Get statistics about docs including language detection, without extracting keywords:

`python get_keywords.py -data_dir sciodump.zip -start_date 2023-03-01 -end_date 2023-04-01 -skip_extraction -detect_lg`

## NER-based threat scenario component detection

As entity labels we used both argument type labels and event type labels (the CASIE paper used separate classifiers for these).

### Dataset preparation

Create a a Spacy-compatible docbin dataset (with a train, development and test split) from a downloaded version of the CASIE corpus:

`python preprocess_casie.py -data_dir CASIE-master/CASIE-master/data/`

Alternatively, the already prepared dataset files under the casie_ner directory can also be used for training and testing.

### Training with Spacy (GPU)

Train a NER model on the CASIE entity annotations

`python -m spacy train -g 0 casie_ner/ner_config.cfg --output ./casie_ner/output`

The trained model can also be made available upon request outside GitHub. It has a performace of F1=63.24 (some CASIE annotations were excluded from the data used for training due to overlapping span offsets incompatible with Spacy's NER. The dataset used had a total of 831 documents across the 3 splits.)

### Inference (GPU)

Evaluate with Spacy on the CASIE test set (with the trained model assumed to be under casie_ner/output) and generate some example html visualizations with color-highlighting:

`python -m spacy benchmark accuracy casie_ner/output/model-best casie_ner/test.spacy --output casie_ner/test_meta.json --gpu-id 0 --displacy-path html_vis --displacy-limit 10`

Do inference on a plain text file (with SCIO texts):

`python test_detection.py -model_dir casie_ner/output/model-best -test_data scio_test_data.txt -from_file`

Do inference on a short text directly from the command line with html visualizations using color-highlighting:

`python test_detection.py -model_dir casie_ner/output/model-best -test_data "QuoIntelligence followed the massive ransomware attacks targeting VMware ESXi servers worldwide by exploiting the two-year-old vulnerability CVE-2021-21974. While such servers should not be publicly exposed and should be patched by now, we observed how the attackers continue to exploit old vulnerabilities resulting in successful attacks."`


