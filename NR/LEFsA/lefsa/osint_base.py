# -*- coding: utf-8 -*-

import pickle
from abc import abstractmethod
from typing import Any, Dict, List, Optional

import pandas
import tqdm

from lefsa.osint_data import InputFeatures

CONF_INTERVAL_BOUNDS = [0.025, 0.975]


class BaseForecaster:
    """Defines the interface for all forecasters.

    It should implement the three following methods:
    - train(train_df): trains the forecaster on the provided training DataFrame
    - predict(test_df, target_var): predicts the target variable for the provided
    test DataFrame.
    - target_vars: property returning the list of target variables that the forecaster
       can predict.
    """

    def __init__(self, name: str):
        """Initialises the forecaster"""

        self.name = name

    @abstractmethod
    def train(self, train_df):
        """Trains the forecaster on the provided training DataFrame."""

        raise NotImplementedError("train() not implemented yet")

    def predict_from_feats(
        self, feats: InputFeatures, target_vars: Optional[List[str]] = None
    ):
        """Runs the predictions on multiple target variables and returns the results
        in a DataFrame. feats is a dictionary of feature-value pairs.
        If target_vars is None, predicts for all target variables."""

        test_df = pandas.DataFrame.from_records([feats.to_dict()])
        pred_results, prob_results, interval_results = self.predict_all(
            test_df, target_vars
        )

        pred = pred_results.round(4).iloc[0].to_dict()
        prob_distribs = {
            k: v.round(4).iloc[0].to_dict() for k, v in prob_results.items()
        }
        intervals = {
            k: v.round(4).iloc[0].to_dict() for k, v in interval_results.items()
        }

        return pred, prob_distribs, intervals

    def predict_all(
        self, test_df: pandas.DataFrame, target_vars: Optional[List[str]] = None
    ):
        """Runs the forecaster on multiple target variables (based on the input
        features in test_df) and returns the results.
        If target_vars is None, predicts for all variables in self.target_vars
        that are not in test_df.columns.

        Arguments:
        - test_df: DataFrame containing the input features for prediction
        - target_vars: List of target variable names to predict. If None, predicts for all
          variables in self.target_vars that are not in test_df.columns.

        Returns:
        - pred_results: DataFrame containing the predictions for each target variable
        - prob_results: Dictionary mapping target variable names to DataFrames containing
          class probabilities (for discrete variables)
        - interval_results: Dictionary mapping target variable names to DataFrames containing
          prediction intervals (for continuous variables)
        """

        if target_vars is None:
            target_vars = [c for c in self.target_vars if c not in test_df.columns]

        pred_results = {}
        prob_results = {}
        interval_results = {}
        for target_var in target_vars:
            results = self.predict(test_df, target_var)

            # If the target variable has a precondition, we change the name of the
            # variable to reflect that conditional dependency
            full_target_var = str(target_var)
            if "precondition" in results:
                full_target_var += "|" + results["precondition"]

            # We store both the predictions themselves as well as the class
            # probabilities (for classification tasks) or prediction intervals
            # (for regression tasks)
            pred_results[full_target_var] = results["preds"]
            if "probs" in results:
                prob_results[full_target_var] = results["probs"]
            if "intervals" in results:
                interval_results[full_target_var] = results["intervals"]

        pred_results = pandas.DataFrame.from_dict(pred_results, orient="columns")
        return pred_results, prob_results, interval_results

    @abstractmethod
    def predict(self, test_df: pandas.DataFrame, target_var: str) -> dict:
        raise NotImplementedError("train() not implemented yet")

    @property
    @abstractmethod
    def target_vars(self) -> List[str]:
        """Returns the list of target variable names that the forecaster can predict."""

        raise NotImplementedError("target_vars() not implemented yet")

    def has_precondition(self, var: str) -> bool:
        """Returns True if the variable name depends on another variable being true
        (i.e. it has a precondition), False otherwise."""
        if not hasattr(self, "preconditions"):
            self.preconditions = {}
            for c in self.target_vars:
                splits = c.split(".")
                prefixes = [".".join(splits[:i]) for i in range(2, len(splits))]
                prefixes = [p for p in prefixes if p in self.target_vars]
                if prefixes:
                    self.preconditions[c] = prefixes[-1]
        return var in self.preconditions

    def get_precondition(self, var: str) -> str:
        """If the variable name depends on another variable being true (i.e. it has
        a precondition), returns the name of that precondition variable. Otherwise,
        raise an error"""

        # The search for predictions is done only once and cached. The system
        # is quite simple and based on string prefixes: if a variable is named
        # "a.b.c", and there exists another variable "a.b", then "a.b" is
        # considered a precondition for "a.b.c".

        if not self.has_precondition(var):
            raise KeyError(f"Variable {var} has no precondition")
        return self.preconditions[var]

    def get_top_level_targets(
        self, to_exclude: Optional[List[str]] = None
    ) -> List[str]:
        """Returns the list of target variables that do not have any preconditions."""
        top_level_vars = []
        for var in self.target_vars:
            if not self.has_precondition(var) and (
                to_exclude is None or var not in to_exclude
            ):
                top_level_vars.append(var)
        return top_level_vars

    def get_conditional_targets(self, precondition: str) -> List[str]:
        """Returns the list of target variables that are relevant to query if
        the given precondition is met."""
        conditional_vars = []
        for var in self.target_vars:
            if self.has_precondition(var):
                if self.get_precondition(var) == precondition:
                    conditional_vars.append(var)
        return conditional_vars

    def evaluate(
        self,
        test_df: pandas.DataFrame,
        target_vars: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> pandas.DataFrame:
        """Evaluates the forecaster on the provided test DataFrame
        and returns the evaluation metrics as a DataFrame."""

        from .osint_evaluate import eval_forecaster

        return eval_forecaster(self, test_df, target_vars, verbose=verbose)

    def save(self, path: Optional[str] = None):
        """Saves the forecaster to a pickle file"""

        if path is None:
            path = self.name + ".pkl"
        with open(path, "wb") as fd:
            pickle.dump(self, fd)
        return self

    @staticmethod
    def load(path: str) -> "DummyForecaster":
        """Loads the forecaster from disk."""

        with open(path, "rb") as fd:
            forecaster = pickle.load(fd)
        return forecaster


class DummyForecaster(BaseForecaster):
    """A simple forecaster that uses a DummyPredictor for each target variable.
    The DummyPredictor always predicts the most frequent class for classification
    tasks, or the median value for regression tasks."""

    def __init__(self, name="dummy_forecaster"):
        super().__init__(name=name)

        # The dictionary of trained predictors, one per target column
        self.predictors: Dict[str, Any] = {}

    # =========================
    # TRAINING
    # =========================

    def train(self, train_df, target_cols: List[str]):
        self.target_cols = target_cols

        print("==> Fitting %i predictors" % (len(self.target_cols)))
        for target_col in tqdm.tqdm(self.target_cols):
            self.fit_predictor(train_df, target_col)

        return self

    def fit_predictor(self, train_df: pandas.DataFrame, target_col: str):
        """Fits a predictor for a specific target column, and stores
        it in self.predictors."""

        # If the predictor has parent conditions, we filter the training data accordingly
        if self.has_precondition(target_col):
            train_df = train_df[train_df[self.get_precondition(target_col)]]

        # We only select data with a known output value
        train_df = train_df.dropna(subset=[target_col])

        if len(train_df) == 0:
            print("No data with", target_col, "=True, skipping")
            return self

        self.predictors[target_col] = DummyPredictor(train_df[target_col])

        return self

    def predict(self, test_df: pandas.DataFrame, target_col):
        """Predicts the target column for the given test data.
        Returns a dictionary with the following keys:
        - "preds": pandas.Series with point predictions
        - "probs": pandas.DataFrame with class probabilities (for classification)
        - "intervals": pandas.DataFrame with prediction intervals (for regression)
        - "precondition": str with the precondition column name (if any)
        """

        if target_col not in self.predictors:
            raise RuntimeError(f"Predictor for {target_col} not trained yet")

        # If the predictor has parent conditions, we filter the training data accordingly
        results = {}
        if self.has_precondition(target_col):
            results["precondition"] = self.get_precondition(target_col)

        predictor = self.predictors[target_col]
        results["preds"] = predictor.predict(test_df)

        if predictor.problem_type == "regression":
            results["intervals"] = predictor.predict_interval(test_df)
        else:
            results["probs"] = predictor.predict_proba(test_df)

        return results

    @property
    def target_vars(self) -> List[str]:
        """Returns the list of target variable names that the forecaster can predict."""

        return self.target_cols


class DummyPredictor:
    """A simple predictor that always predicts the most frequent class
    for classification tasks, or the median value for regression tasks."""

    def __init__(self, target_values: pandas.Series):
        if target_values.dtype in ["bool", "category"]:
            self.value = target_values.mode(dropna=True).iloc[0]
            self.value_distrib = target_values.value_counts(normalize=True)
        else:
            self.lower_bound = target_values.dropna().quantile(
                (CONF_INTERVAL_BOUNDS[0])
            )
            self.value = target_values.dropna().median()
            self.upper_bound = target_values.dropna().quantile(CONF_INTERVAL_BOUNDS[1])

    @property
    def problem_type(self):
        if hasattr(self, "value_distrib"):
            return "classification"
        else:
            return "regression"

    def predict(self, input_data):
        """Predicts the median or target value found during training."""

        return pandas.Series(
            [self.value for _ in range(len(input_data))], index=input_data.index
        )

    def predict_proba(self, input_data):
        """Returns a probability distribution with a probability of 1
        for the most frequent class."""

        if self.problem_type == "regression":
            raise RuntimeError("Probability predictions not available for regression")

        repeated_probs = [self.value_distrib.values] * len(input_data)
        proba = pandas.DataFrame(
            repeated_probs, columns=self.value_distrib.index, index=input_data.index
        )
        return proba

    def predict_interval(self, input_data):
        """Returns prediction intervals based on the quantiles found during training."""

        if self.problem_type == "classification":
            raise RuntimeError("Prediction intervals not available for classification")

        low_bound_percent = "%.1f %%" % (100 * CONF_INTERVAL_BOUNDS[0])
        high_bound_percent = "%.1f %%" % (100 * CONF_INTERVAL_BOUNDS[1])
        return pandas.DataFrame(
            {
                low_bound_percent: [self.lower_bound] * len(input_data),
                high_bound_percent: [self.upper_bound] * len(input_data),
            },
            index=input_data.index,
        )
