# -*- coding: utf-8 -*-

import os
import pickle
import random
from typing import List, Optional

import numpy as np
import pandas
import sklearn.metrics
import tqdm

from . import osint_base, osint_data

"""Utility functions to evaluate OSINT forecasters on the VCDB test set and tune
their hyperparameters."""


def run_evaluation(
    forecasters,
    output_to_file: str = "results.pkl",
    target_cols: Optional[List[str]] = None,
    append=True,
):
    """Runs evaluation for the provided list of forecasters on the VCDB test set.
    The results are saved to the specified output file."""

    _, test_df = osint_data.VCDB().split_train_test()

    # We load all forecasters (if they were provided as paths)
    loaded_forecasters = []
    for forecaster in forecasters:
        if isinstance(forecaster, osint_base.BaseForecaster):
            loaded_forecasters.append(forecaster)
        elif os.path.isdir(forecaster):
            from . import osint_gluon

            loaded_forecasters.append(osint_gluon.GluonForecaster.load(forecaster))
        elif forecaster.endswith(".pkl"):
            forecaster = osint_base.BaseForecaster.load(forecaster)
        else:
            raise RuntimeError(f"Cannot load forecaster from {forecaster}")

    if append and output_to_file is not None and os.path.isfile(output_to_file):
        with open(output_to_file, "rb") as f:
            all_results = pickle.load(f)
            del all_results["aggregate"]
    else:
        all_results = {}

    for forecaster in loaded_forecasters:
        # We evaluate each forecaster individually
        all_results[forecaster.name] = eval_forecaster(forecaster, test_df, target_cols)

        # We compute aggregate results (for all forecasters so far)
        agg_results = {
            forecaster_name: results_for_forecaster.mean(axis=0, skipna=True)  # type: ignore
            for forecaster_name, results_for_forecaster in all_results.items()
        }
        agg_results = pandas.DataFrame.from_dict(agg_results, orient="index")

        # We save the results so far to file
        if output_to_file is not None:
            results_to_write = {"aggregate": agg_results, **all_results}
            with open(output_to_file, "wb") as f:
                pickle.dump(results_to_write, f)


def run_kernel_tuning(output_to_file: str = "results.pkl", only_top_level: bool = True):
    """Runs a grid search over various kernel forecaster hyperparameters,
    evaluating each configuration on the VCDB test set.
    The results are saved to the specified output file."""

    from . import osint_gluon, osint_kernel

    train_df, _ = osint_data.VCDB().split_train_test()
    _, input_cols, target_cols = osint_data.classify_columns(train_df)

    # To speed up tuning, we only consider top-level targets
    if only_top_level:
        dummy_forecaster = osint_gluon.GluonForecaster("")
        dummy_forecaster.target_cols = target_cols
        target_cols_to_test = dummy_forecaster.get_top_level_targets()
        print(
            "Tuning only %i top-level targets: %s"
            % (len(target_cols_to_test), target_cols_to_test)
        )

    # We create all forecaster configurations to evaluate
    forecasters = []
    for distance_metric in ["euclidean", "cosine"]:
        for bandwidth in [0.1, 0.5, 1.0, 10.0]:
            for k in [100, None]:
                for rareness_bonus in [0, 5, 10, 20]:
                    for rareness_steepness in [1, 5, 10]:
                        name = f"kernel_k={k}_rb={rareness_bonus}_rs={rareness_steepness}_bw={bandwidth}_dist={distance_metric}"
                        forecaster = osint_kernel.KernelForecaster(
                            name=name,
                            bandwidth=bandwidth,
                            distance_metric=distance_metric,  # type: ignore
                            k=k,
                            rareness_bonus=rareness_bonus,
                            rareness_steepness=rareness_steepness,
                        )
                        forecaster.train(
                            train_df, relevant_cols=(input_cols + target_cols)
                        )
                        forecasters.append(forecaster)
    random.shuffle(forecasters)

    # We run evaluation for all forecasters
    run_evaluation(forecasters, output_to_file, target_cols_to_test)


def eval_forecaster(
    forecaster: osint_base.BaseForecaster,
    test_df: pandas.DataFrame,
    target_cols: Optional[List[str]] = None,
    verbose: bool = True,
) -> pandas.DataFrame:
    """Evaluates the provided forecaster on the provided test set."""

    print("=== Evaluating forecaster:", forecaster.name)

    target_cols = target_cols or forecaster.target_vars

    all_results = {}

    # We evaluate each target column individually
    target_cols_iter = target_cols if verbose else tqdm.tqdm(target_cols)
    for target_col in target_cols_iter:
        full_target_var = str(target_col)
        if forecaster.has_precondition(target_col):
            full_target_var += "|" + forecaster.get_precondition(target_col)  # type: ignore
        all_results[full_target_var] = eval_predictor(
            forecaster, test_df, target_col, verbose=verbose
        )

    # We convert the results to a DataFrame
    all_results = pandas.DataFrame.from_dict(all_results, orient="index")  # type: ignore

    print("=== Finished evaluation for forecaster:", forecaster.name)
    return all_results


def eval_predictor(
    forecaster: osint_base.BaseForecaster,
    test_df: pandas.DataFrame,
    target_col: str,
    verbose: bool = True,
) -> dict:
    """Evaluates the predictor for the specified target column."""

    if hasattr(forecaster, "predictors") and target_col not in forecaster.predictors:  # type: ignore
        raise RuntimeError(f"Predictor for {target_col} not trained yet")

    if verbose:
        print("Evaluating predictor for", target_col)
    if forecaster.has_precondition(target_col):
        if verbose:
            print(
                "Conditioned on:", forecaster.get_precondition(target_col), "being True"
            )
        test_df = test_df[test_df[forecaster.get_precondition(target_col)]]

    # We drop instances with unknown target value
    test_df = test_df.dropna(subset=[target_col])

    ground_truth = test_df[target_col]
    if verbose:
        print("Number of test instances:", len(test_df))

    _, input_cols, _ = osint_data.classify_columns(test_df)
    # Important for e.g. kernel forecasters: we only keep input columns
    test_df = test_df[input_cols]

    # We get predictions from the forecaster
    full_predictions = forecaster.predict(test_df, target_col)
    point_predictions = full_predictions["preds"]

    # Classification evaluation
    if ground_truth.dtype in ["bool", "category"]:
        # We print the relative frequencies
        if verbose:
            print("==> Relative frequencies of each value for", target_col)
            print(get_relative_frequencies(ground_truth, point_predictions))

        # We compute standard classification metrics
        acc = sklearn.metrics.accuracy_score(ground_truth, point_predictions)

        average = "binary" if ground_truth.dtype == "bool" else "weighted"
        p, r, f1, _ = sklearn.metrics.precision_recall_fscore_support(
            ground_truth,
            point_predictions,
            average=average,
            zero_division=np.nan,  # type: ignore
        )
        results = {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

        # We extract probability predictions, and postprocess it whenever necessary
        # (e.g. if some classes found in the test set were not in the training set)
        prob_predictions = full_predictions["probs"]
        for val in ground_truth.unique():
            if val not in prob_predictions.columns:
                prob_predictions[val] = 0.0
        prob_predictions = prob_predictions.reindex(
            sorted(prob_predictions.columns), axis=1
        )
        prob_predictions = prob_predictions.div(prob_predictions.sum(axis=1), axis=0)

        # We compute log-loss and ROC-AUC
        if len(prob_predictions.columns) >= 2:
            results["log_loss"] = sklearn.metrics.log_loss(
                ground_truth, prob_predictions, labels=prob_predictions.columns
            )
        if ground_truth.nunique() == 2:
            results["roc_auc"] = sklearn.metrics.roc_auc_score(
                ground_truth,
                prob_predictions.iloc[:, 1],
                labels=prob_predictions.columns,
            )
        elif ground_truth.nunique() > 2:
            results["roc_auc"] = sklearn.metrics.roc_auc_score(
                ground_truth,
                prob_predictions,
                multi_class="ovr",
                average="weighted",
                labels=prob_predictions.columns,
            )
        if verbose:
            print("Results: ", {k: round(v, 4) for k, v in results.items()})
        return results

    # Regression evaluation using MSE, MAE, MAPE
    elif ground_truth.dtype == "float":
        if verbose:
            print(
                "  Average values: %.3f (system output) vs. %.3f (true values)"
                % (np.mean(point_predictions), ground_truth.mean())
            )
        results = {
            "mse": sklearn.metrics.mean_squared_error(ground_truth, point_predictions),
            "mae": sklearn.metrics.mean_absolute_error(ground_truth, point_predictions),
            "mape": sklearn.metrics.mean_absolute_percentage_error(
                ground_truth, point_predictions
            ),
        }
        if verbose:
            print("Results: ", results)
        return results

    else:
        print(
            "WARNING:",
            f"Cannot evaluate predictor for {target_col} of type {ground_truth.dtype}",
        )
        return {}


def get_relative_frequencies(
    ground_truth: pandas.Series,
    point_predictions: pandas.Series,
    include_percentages: bool = True,
) -> pandas.DataFrame:
    """Computes the relative frequencies of each class in the ground truth
    and the model's predictions."""

    true_frequencies = ground_truth.value_counts().to_dict()
    output_frequencies = point_predictions.value_counts().to_dict()
    output_proportions = (
        point_predictions.value_counts() * 100 / len(ground_truth)
    ).to_dict()
    true_proportions = (ground_truth.value_counts() * 100 / len(ground_truth)).to_dict()

    all_possible_values = sorted(
        set(true_frequencies.keys()).union(set(output_frequencies.keys()))
    )
    conf_records = {}
    for value in all_possible_values:
        conf_records[value] = {
            "predicted": output_frequencies.get(value, 0),
            "true": true_frequencies.get(value, 0),
        }
        if include_percentages:
            conf_records[value]["predicted (%)"] = output_proportions.get(value, 0.0)
            conf_records[value]["true (%)"] = true_proportions.get(value, 0.0)
    return pandas.DataFrame.from_dict(conf_records, orient="index")
