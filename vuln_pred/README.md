
# Script for training and inference


### Optimize parameters
Usage:
```python vuln_pred/parameter_tuning.py```
This script implements a machine learning pipeline that trains and optimizes an XGBoost classifier using  cross-validation on time windows and Optuna for hyperparameter optimization.


# Training model
Usage:
```python vuln_pred/train_pipeline.py```
A simple training script that loads data, trains an XGBoost classifier with predefined hyperparameters, and saves the model pipeline.


# Prediction
Usage:
```python vuln_pred/predict.py```
A prediction script that loads a trained model and predicts exploitation probabilities for CVEs. The highest scoring CVEs are shown, and the result is places under data/ at the project root, cve_predictions.csv.