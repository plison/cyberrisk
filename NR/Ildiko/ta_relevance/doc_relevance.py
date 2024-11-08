"Trains a neural (BERT) model and uses it for scoring documents for threat actor relevance."

import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from safetensors.torch import load_file
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_squared_error, r2_score, accuracy_score
import torch
from datasets import Dataset, load_dataset, DatasetDict, concatenate_datasets, Value
import pandas as pd
import math
import argparse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import json

BASE_MODEL = "bert-base-cased"
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
EPOCHS = 20

Dataset.cleanup_cache_files

def split_data(dataset, stratif_col, test_size_prop=0.3, val_prop=0.66):
    """ Split dataset into train, validation and test sets. The distribution of labels
    in each split are maintained (= stratified splitting) based on 'stratif_col'.
    Parameters:
        dataset: a Dataset instance
        stratif_col: str; column name to base the stratification on
        test_size_prop: proportion of instances to use for testing out of the full dataset
        val_prop: proportion of validation set with respect to the test set size.
    Returns:
        a DatasetDict with the three splits. 
    """
    # Divide all into 2 sets: train and test+validation
    train_testvalid = dataset['train'].train_test_split(test_size=test_size_prop, seed=123, stratify_by_column=stratif_col)
    
    if val_prop:
        # Redivide test+validation into separate test and validation sets
        test_valid_split = train_testvalid['test'].train_test_split(test_size=0.66, seed=123, stratify_by_column=stratif_col)
        
        # Gather all in a single DatasetDict
        split_ds = DatasetDict({
            'train': train_testvalid['train'],
            'test': test_valid_split['test'],
            'validation': test_valid_split['train']})

    else:
        # Return 2 splits and a placeholder for a test set to be filled in later 
        split_ds = DatasetDict({
            'train': train_testvalid['train'],
            'test': None,
            'validation': train_testvalid['test']})

    print("Dataset information (with original columns)")
    print(split_ds)
    print()

    return split_ds

def preprocess(examples):
    "Preprocess texts with tokenizer and truncate."
    label = examples["score"] 
    # Tokenize (will add new columns with input ids etc) 
    # Truncate based on model's limit + pad to max doc length in batch   
    examples = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512) 
    examples["label"] = float(label)
    return examples

def compute_metrics_for_regression(eval_pred):
    """ Compute regression metrics.
    Parameters:
        eval_pred (tuple): logits and scores lists
    Returns:
        A dictionary with metrics names and their values.
    """
    logits, labels = eval_pred
    labels = labels.reshape(-1, 1)
    rmse = root_mean_squared_error(labels, logits)
    mse = mean_squared_error(labels, logits)
    mae = mean_absolute_error(labels, logits)
    r2 = r2_score(labels, logits)
    single_squared_errors = ((logits - labels).flatten()**2).tolist()
    
    # Compute accuracy 
    # Based on the fact that the rounded score = true score only if |single_squared_errors| < 0.5
    accuracy = sum([1 for e in single_squared_errors if e < 0.25]) / len(single_squared_errors) 
    
    return {"rmse": rmse, "mse": mse, "mae": mae, "r2": r2, "accuracy": accuracy}

# not really neeeded here, but useful if one also wants to predict multiple continuous values at the same time (e.g. x,y coordinates). 
class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """ Overrides the default compute_loss method of the Trainer class. 
        Computes the Mean Squared Error (MSE) loss for regression tasks.
        Parameters:
            model: The model being trained.
            inputs: The input data, containing both features and labels.
            return_outputs (bool): If True, returns both the loss and model outputs. 
                                   Otherwise, returns only the loss.
        Returns:
            The MSE loss between predicted logits and actual labels. 
            Optionally returns the model outputs if return_outputs is True.
        """
        # Extract the labels (target values) from the inputs
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        # Take the first column of logits from the outputs 
        logits = outputs[0][:, 0] 
        loss = torch.nn.functional.mse_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss

def train_model(model_dir, model, ds):
    """ Trains the given model on the provided dataset.
    Parameters:
        model_dir (str): Directory where the trained model will be saved.
        model: The model to be trained.
        ds (DatasetDict): The dataset dictionary containing training and validation data.
    Returns:
        trainer (Trainer): The trained model's trainer object.
    """
    # Define the training arguments for the Trainer class
    # by default all layers are updated & a random seed is set 
    training_args = TrainingArguments(
        output_dir=model_dir, 
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        eval_strategy="epoch",    # Evaluate at the end of each epoch
        save_strategy="epoch",          # Save checkpoints at each epoch
        save_total_limit=2,             # Save only the best and last models
        metric_for_best_model="accuracy",
        weight_decay=0.01,
        load_best_model_at_end=True,    # Load the best model after training
    )

    # Initialize the custom RegressionTrainer with the model and datasets
    trainer = RegressionTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        compute_metrics=compute_metrics_for_regression,
    )
    
    print("Starting training...")
    trainer.train()
    print("Finished training...")

    return trainer

def load_model(model_dir, my_device):
    """ Loads a pre-trained model from the specified directory and moves it to the given device.
    Parameters:
        model_dir (str): Path to the directory containing the model checkpoint (.safetensors).
        my_device (str): The device ('cpu' or 'cuda') where the model should be loaded.
    Returns:
        model: The loaded model, set to evaluation mode.
    """
    # Load the model's state dictionary from the saved file
    print(f"Loading model from {model_dir}...")
    state_dict = load_file(model_dir)
    # Load the state dict into the model
    model.load_state_dict(state_dict)
    model.to(torch.device(my_device))
    # Set the model to evaluation mode (disables dropout, etc.)
    model.eval()  
    return model

def map_scores(dataset, score_col="manual_score"):
    # Cast manual labels to be between 0-2 (as predictions) instead of 1-3
    label = dataset[score_col] 
    dataset[score_col] = float(label-1)
    return dataset

def predict_w_model(model, test_dataset, score_col, text_col='text'):    
    """ Collect model predictions in batches for a given test dataset. The input data  should be raw text as
    it is tokenized within the function.
    Parameters:
        model: The pre-trained model used for making predictions.
        test_dataset (Dataset): The dataset containing raw text data and corresponding scores.
        score_col (str): The column in the dataset containing the actual target scores.
        text_col (str): The column containing the text data to be fed into the model. Defaults to 'text'.
    Returns:
        y_preds_float (list): The model's predicted values as unrounded floats.
        metrics (dict): Dictionary of results per evaluation metric (accuracy, RMSE, etc.).
    """
    # Calculate the number of batches needed based on the test dataset size and batch size
    nb_batches = math.ceil(len(test_dataset)/BATCH_SIZE)
    y_preds = []
    num_labels = len(test_dataset[score_col])

    # Initialize a tensor to store all logits (predicted values)
    all_logits = torch.empty((num_labels, 1))
    start_ix = 0
    with torch.no_grad():  # No need to track gradients during evaluation, so we disable it
        for i in range(nb_batches):

            # Get the current batch of input texts
            input_texts = test_dataset[i * BATCH_SIZE: (i+1) * BATCH_SIZE][text_col]
            
            # Encode the data to predict 
            encoded = tokenizer(input_texts, 
                                truncation=True, 
                                padding="max_length", 
                                max_length=512, 
                                return_tensors="pt").to("cuda")
            
            # Get predictions (logits) for the current batch
            logits = model(**encoded).logits
            all_logits[start_ix:start_ix + logits.shape[0], :] = logits
            start_ix += logits.shape[0]

    # Get list of predicted (unrounded) float values (need to be rounded to match against original ordinal scores)
    y_preds_float = all_logits.reshape(-1).tolist()

    # Compute evaluation scores
    eval_pred = (all_logits, torch.FloatTensor(test_dataset[score_col]))
    metrics = compute_metrics_for_regression(eval_pred)
    print("Results on test set") 
    print('Acc:', metrics["accuracy"], 'RMSE:', metrics["rmse"])

    # TODO: finish save predictions (y_preds)
    df = pd.DataFrame([test_dataset[text_col], test_dataset[score_col], y_preds], ["Text", "Score", "Prediction"])

    return y_preds_float, metrics

def plot_confusion_matrix(true_labels, predictions, labels):
    """ Plots a confusion matrix using the true labels and predicted labels.
    Parameters:
        true_labels (list or array): The ground truth labels.
        predictions (list or array): The predicted labels from the model.
        labels (list): The list of label names to be displayed on the matrix.
    """
    conf_matrix = confusion_matrix(true_labels, predictions)

    # Change figure size and increase dpi for better resolution
    fig, ax = plt.subplots(figsize=(8,6), dpi=100)
    
    display = ConfusionMatrixDisplay(conf_matrix, display_labels=labels)
     
    # Pass the parameter ax to show customizations (ex. title) 
    display.plot(ax=ax)
    
    # Show the plot
    plt.show()

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("-d", "--data_path", type=str,
                           help="Path to dataset with a selected subset of SCIO docs.", default="scio_ta_ds_ann_20241104.json") #scio_ta_ds_llm_ann_v2
    argparser.add_argument("-m", "--model_dir", type=str,
                           help="Directory of model to train or path to 'model.safetensors' file to load.", required=True)
    argparser.add_argument("-a", "--action", type=str, 
                           help="Action to take: 't' for train, 'e' for evaluate, 'p' for predict or any combination of these concatenated in a string with no space.", 
                           default="e")
    argparser.add_argument("-g", "--gpu", type=str, 
                           help="Which device (GPU) to use, a string with a colon fomatting. Default: 'cuda:0' (first visible device).", 
                           default="cuda:0")
    argparser.add_argument("-s", "--score_col", type=str, 
                           help="The name of the data column containing the annotated score.", 
                           default="score")
    argparser.add_argument("-t", "--test_data", type=str, 
                           help="For self-training: The type of test data to use, 'manual' for manually labeled or 'llm' for data without manual labels.", 
                           default="manual")
    argparser.add_argument("-ad", "--additional_data", type=str, 
                           help="For self-training: Path to JSON with additional (training) data to merge with the other data file provided.", 
                           default=None) 
    argparser.add_argument("-pf", "--pred_file", type=str, 
                           help="JSON filename where to save the predictions.", 
                           default="preds.json")
    args = argparser.parse_args()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)

    # Load data, standardize score column and encode it
    unsplit_ds = load_dataset("json", data_files=args.data_path)
    try:
        unsplit_ds = unsplit_ds.rename_column("llm_score", "score")
    except ValueError:
        pass
    unsplit_ds = unsplit_ds.class_encode_column(args.score_col)

    # Re-train llm model excluding the manual test set used for evaluating all other models
    if args.action in ['rtep', 'rp']:
        # Extract filenames from the test set of the manual data
        manual_data = load_dataset("json", data_files=args.additional_data) 
        manual_data = manual_data.filter(lambda x: x["labeler"] != "llm")
        split_ds = split_data(manual_data, args.score_col)
        test_filenames = set(split_ds["test"]["filename"])
        val_filenames = set(split_ds["validation"]["filename"])
        test_val_filenames = test_filenames.union(val_filenames)
        test_dataset = split_ds["test"]

        # Filter the llm training set to keep rows where "filename" is in not test_filenames
        unsplit_ds["train"] = unsplit_ds["train"].filter(lambda example: example["filename"] not in test_val_filenames) 

        # Use the LLM annotated data for training and the manual data for validation and testing        
        split_ds['train'] = unsplit_ds["train"]
        
    else:
        # Filter instances based on label source
        if args.test_data == 'llm':
            # Create a test set with all the llm labeled data in one split
            # For auto annotating training data with BERT (to be used for self-training later) 
            unsplit_ds = unsplit_ds.filter(lambda x: x["labeler"] == "llm")
            test_dataset = unsplit_ds["train"]
        else:
            # Keep only human-labeled 
            unsplit_ds = unsplit_ds.filter(lambda x: x["labeler"] != "llm")

            # For using only one data source (LLM annotated data only or manual data only)
            split_ds = split_data(unsplit_ds, args.score_col)

            # Concatenate additional data (if any) to current training data for the self-training setup 
            if args.additional_data:
                split_ds["train"] = split_ds["train"].remove_columns(["terms"])
                bert_ds = load_dataset("json", data_files=args.additional_data)
                bert_ds = bert_ds.class_encode_column("score")
                bert_ds["train"] = bert_ds["train"].remove_columns(["terms"])
                split_ds["train"] = concatenate_datasets([split_ds["train"], bert_ds["train"]])
            
            test_dataset = split_ds['test']

    # Train
    if 't' in args.action:

        # Preprocess data (SCIO dataset)
        for split in split_ds:
            try: 
                split_ds[split] = split_ds[split].map(preprocess, remove_columns=["filename", "title", "date", "lang", "terms", "score", "text", "split"])
                                                                                    
            except:
                split_ds[split] = split_ds[split].map(preprocess, remove_columns=["filename", "title", "date", "lang", "score", "text", "labeler", "split"])

        print("Dataset information (with updated columns after preprocessing)")
        print(split_ds)
        print()

        # Train
        trainer = train_model(args.model_dir, model, split_ds)

        # Evaluate a trained model
        if 'e' in args.action:
            # Evaluate on the test set with a trainer
            print("Evaluating on test set...")
            trainer.eval_dataset=split_ds["test"]
            print(trainer.evaluate())
        if 'p' in args.action:
            predictions = trainer.predict(test_dataset) # predicted labels in 'label_ids' attrib, results in 'metrics'

    # Predict
    if 'p' in args.action:
    
        # Load model from disk
        if 't' not in args.action:
            model = load_model(args.model_dir, args.gpu)

        # Get metrics    
        y_preds, metrics = predict_w_model(model, test_dataset, score_col=args.score_col)
        for metric, result in metrics.items():
            print(metric, "\t", result)

        # Get predicted scores               
        pd.set_option('display.max_rows', 20)
        df = pd.DataFrame([test_dataset["text"], test_dataset[args.score_col], y_preds], ["Text", "Score", "Prediction"]).T
        df["Rounded Prediction"] = df["Prediction"].apply(round)
        
        # Increment predicted score with 1 to match human annotation labels when saving data (encoding casts 1,2,3 to 0,1,2)
        pred_scores = [v+1 for v in df["Rounded Prediction"].to_list()]
         
        assert sorted(set(df["Score"].to_list())) == sorted(set(df["Rounded Prediction"].to_list()))

        # Analyze misclassifications 
        incorrect_cases = df[df["Score"] != df["Rounded Prediction"]]
        print(incorrect_cases)
        print("Acc trad", accuracy_score(df["Score"].to_list(), df["Rounded Prediction"].to_list()))
        print(classification_report(df["Score"].to_list(), df["Rounded Prediction"].to_list()))

        # Save predictions
        if args.test_data == 'llm':    
            test_dataset = test_dataset.map(lambda x: {"labeler": "bert_ft"})
            # Remove old score column and replace with the newly predicted scores
            test_dataset = test_dataset.remove_columns(["score"])
            test_dataset = test_dataset.add_column("score", pred_scores)
        else:
            # Save the test split with predictions and remove some metadata (misleading or incorrectly tranformed during loading)
            test_dataset = test_dataset.remove_columns(["terms"])
            test_dataset = test_dataset.remove_columns(["date"])
            test_dataset = test_dataset.remove_columns(["split"])
            test_dataset = test_dataset.remove_columns(["labeler"])
            test_dataset = test_dataset.add_column("predictions", pred_scores)
        test_dataset.to_json(args.pred_file, lines=False, indent=4)

        # Plot confusion matrix
        plot_confusion_matrix(df["Score"].tolist(), df["Rounded Prediction"].to_list(), ["Not relevant", "Somewhat relevant", "Very relevant"])


