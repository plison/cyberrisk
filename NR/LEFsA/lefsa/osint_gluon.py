# -*- coding: utf-8 -*-

import os
import pickle
import shutil
from typing import Any, Dict, List, Literal, Optional

import autogluon.tabular
import pandas

from .osint_base import CONF_INTERVAL_BOUNDS, BaseForecaster, DummyPredictor


class GluonForecaster(BaseForecaster):
    def __init__(
        self,
        name="gluon_forecaster",
        base_path: Optional[str] = None,
        model_type: Literal["full", "gbm"] = "gbm",
    ):
        super().__init__(name=name)

        self.base_path = base_path if base_path is not None else name
        self.model_type = model_type

        # The dictionary of trained predictors, one per target column
        self.predictors: Dict[str, Any] = {}

    # =========================
    # TRAINING
    # =========================

    def train(self, train_df, input_cols: List[str], target_cols: List[str]):
        self.input_cols = input_cols
        self.target_cols = target_cols

        print(
            "==> Training %i predictors, each given %i input features"
            % (len(self.target_cols), len(self.input_cols))
        )
        for target_col in self.target_cols:
            self.fit_predictor(train_df, target_col)

        return self

    def fit_predictor(self, train_df: pandas.DataFrame, target_col: str):
        """Fits a predictor for a specific target column, and stores
        it in self.predictors."""

        print("Fitting predictor for", target_col)

        # If the predictor has parent conditions, we filter the training data accordingly
        if self.has_precondition(target_col):
            print("Conditioned on:", self.get_precondition(target_col), "being True")
            train_df = train_df[train_df[self.get_precondition(target_col)]]

        # We only select data with a known output value
        train_df = train_df.dropna(subset=[target_col])

        if len(train_df) == 0:
            print("No data with", target_col, "=True, skipping")
            return self

        if train_df[target_col].dtype in ["bool", "category"]:
            print(
                "Output value distribution:",
                train_df[target_col].value_counts().to_dict(),
            )
        else:
            print("Output value statistics:", train_df[target_col].describe().to_dict())

        # We only keep the relevant columns
        train_df = train_df[self.input_cols + [target_col]]

        # If there is any variance in the output column, we use a dummy predictor
        if len(train_df[target_col].unique()) == 1:
            print("Only one value present, using a dummy predictor")
            self.predictors[target_col] = DummyPredictor(train_df[target_col])

        else:
            path = os.path.join(self.base_path, target_col)
            self.predictors[target_col] = GluonPredictor().fit(
                train_df, target_col, path=path, model_type=self.model_type
            )
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

        #    print("Predicting target column:", target_col)

        # If the predictor has parent conditions, we filter the training data accordingly
        results = {}
        if self.has_precondition(target_col):
            #        print("Conditioned on:", self.get_precondition(target_col), "being True")
            results["precondition"] = self.get_precondition(target_col)

        predictor = self.predictors[target_col]
        if isinstance(predictor, str):  # lazy loading
            self.predictors[target_col] = GluonPredictor.load(predictor)
            predictor = self.predictors[target_col]

        results["preds"] = predictor.predict(test_df)

        if predictor.problem_type == "regression":
            results["intervals"] = predictor.predict_interval(test_df)
        else:
            results["probs"] = predictor.predict_proba(test_df)

        return results

    def save(self):
        """Saves the forecaster to disk, excluding non-dummy predictors to save space.
        (as those are already on disk, we just reload them given their path)"""

        tmp_predictors = {k: v for k, v in self.predictors.items()}
        self.predictors = {
            k: v for k, v in self.predictors.items() if isinstance(v, DummyPredictor)
        }
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
        with open(os.path.join(self.base_path, "forecaster.pkl"), "wb") as fd:
            pickle.dump(self, fd)
        self.predictors = tmp_predictors
        return self

    @staticmethod
    def load(dir_name="osint_forecaster"):
        """Loads the forecaster from disk."""

        with open(os.path.join(dir_name, "forecaster.pkl"), "rb") as fd:
            forecaster = pickle.load(fd)
        for predictor_name in os.listdir(dir_name):
            full_path = os.path.join(dir_name, predictor_name)
            if os.path.isdir(full_path):
                try:
                    forecaster.predictors[predictor_name] = full_path  # lazy loading
                except Exception as e:
                    print(f"Failed to load predictor {predictor_name}: {e}")
        return forecaster

    @property
    def target_vars(self) -> List[str]:
        """Returns the list of target variable names that the forecaster can predict."""

        return self.target_cols


class GluonPredictor:
    """A predictor using AutoGluon TabularPredictor."""

    @property
    def problem_type(self):
        if self.predictor.problem_type == "quantile":
            return "regression"
        else:
            return "classification"

    def fit(
        self,
        train_df,
        target_col,
        path,
        model_type,
        time_limit: int = 120,
        skip_if_exists: bool = False,
    ):
        """Fits an AutoGluon TabularPredictor for the given target column."""

        if os.path.exists(path):
            if skip_if_exists:
                print("Model already trained, skipping")
                try:
                    return autogluon.tabular.TabularPredictor.load(path)
                except Exception as e:
                    print("Failed to load existing model, retraining:", e)
                    shutil.rmtree(path)
            else:
                shutil.rmtree(path)

        print(
            "Fitting %s models with a time limit of %i secs and storing it in %s"
            % (model_type, time_limit, path)
        )

        train_data = autogluon.tabular.TabularDataset(train_df)

        # Setting training arguments (to avoid training too many or too large models)
        args = {
            "train_data": train_data,
            "presets": "medium",
            "time_limit": time_limit,
            "keep_only_best": True,
            "hyperparameters": "light",
            "excluded_model_types": ["KNN", "XT", "RF"],
        }

        # Specific settings for gradient-boosted models
        if model_type == "gbm":
            args["hyperparameters"] = {"GBM": {}}
            args["fit_weighted_ensemble"] = False
            args["fit_full_last_level_weighted_ensemble"] = False

        # Binary classification
        if train_data[target_col].dtype in ["bool"]:
            predictor = autogluon.tabular.TabularPredictor(
                label=target_col, problem_type="binary", verbosity=1, path=path
            )
        # Multiclass classification
        elif train_data[target_col].dtype in ["category"]:
            predictor = autogluon.tabular.TabularPredictor(
                label=target_col, problem_type="multiclass", verbosity=1, path=path
            )

        # For regression, we estimate quantiles to be able to produce prediction intervals
        else:
            quantile_levels = [CONF_INTERVAL_BOUNDS[0], 0.5, CONF_INTERVAL_BOUNDS[1]]
            predictor = autogluon.tabular.TabularPredictor(
                label=target_col,
                problem_type="quantile",
                verbosity=1,
                path=path,
                quantile_levels=quantile_levels,
            )
        predictor.fit(**args)

        # To save space, we delete all models except the best one
        predictor.delete_models(models_to_keep="best")
        predictor.save_space()

        self.predictor = predictor
        return self

    def predict(self, test_df: pandas.DataFrame) -> pandas.Series:
        """Predicts point estimates for the given test data."""

        # We only keep the relevant input columns
        test_df = test_df[self.predictor.features()]
        test_data = autogluon.tabular.TabularDataset(test_df)

        # In case of regression, predict the median
        if self.predictor.problem_type == "quantile":
            return self.predictor.predict(test_data)[0.5]  # type:ignore
        else:
            return self.predictor.predict(test_data)  # type:ignore

    def predict_proba(self, test_df: pandas.DataFrame) -> pandas.DataFrame:
        """Predicts class probabilities for the given test data."""

        if self.predictor.problem_type == "quantile":
            raise RuntimeError("Probability predictions not available for regression")

        # We only keep the relevant input columns
        test_df = test_df[self.predictor.features()]
        test_data = autogluon.tabular.TabularDataset(test_df)
        return self.predictor.predict_proba(test_data)  # type:ignore

    def predict_interval(self, test_df: pandas.DataFrame) -> pandas.DataFrame:
        """Predicts prediction intervals for the given test data."""

        if self.predictor.problem_type != "quantile":
            raise RuntimeError(
                "Prediction intervals not available for non-quantile models"
            )

        # We only keep the relevant input columns
        test_df = test_df[self.predictor.features()]
        test_data = autogluon.tabular.TabularDataset(test_df)
        quantiles = self.predictor.predict(test_data).to_numpy()  # type:ignore

        low_bound_percent = "%.1f %%" % (100 * CONF_INTERVAL_BOUNDS[0])
        high_bound_percent = "%.1f %%" % (100 * CONF_INTERVAL_BOUNDS[1])
        return pandas.DataFrame(
            {low_bound_percent: quantiles[:, 0], high_bound_percent: quantiles[:, 2]},
            index=test_df.index,
        )

    @staticmethod
    def load(path):
        """Reloads a GluonPredictor from disk."""

        predictor = GluonPredictor()
        predictor.predictor = autogluon.tabular.TabularPredictor.load(path)
        return predictor

        return predictor
