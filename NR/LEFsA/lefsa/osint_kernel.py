# -*- coding: utf-8 -*-

import math
from typing import Any, Dict, List, Literal, Optional, Tuple

import numba
import numpy as np
import pandas
import sklearn.preprocessing
from numpy.typing import NDArray

from . import osint_data
from .osint_base import CONF_INTERVAL_BOUNDS, BaseForecaster


class KernelForecaster(BaseForecaster):
    """A non-parametric kernel-based forecaster. It uses a kernel function to
    compute similarities between instances based on their input features,
    and makes predictions based on weighted averages (for regression) or
    weighted voting (for classification) of the target values of similar instances.

    One key advantage of this forecaster is that it becomes possible to predict
    target columns based on any set of input columns (i.e. not only the features about
    the victim organisation, but also any covariate), without having to retrain the
    model. For instance, one may ask: "given a victim organisation with these features,
    and given that the attack use a phishing vector, what is the likely impact? Or
    which asset is most likely to be affected?".
    This is because the model simply stores the training instances and uses them
    to make predictions at inference time."""

    def __init__(
        self,
        name="kernel_forecaster",
        distance_metric: Literal["euclidean", "cosine"] = "euclidean",
        bandwidth: float = 1.0,
        k: Optional[int] = 100,
        rareness_bonus=10,
        rareness_steepness=10,
    ):
        """Initialises the kernel-based forecaster with the specified parameters.

        Arguments:
        - name: the name of the forecaster
        - distance_metric: the distance metric to use ('euclidean' or 'cosine')
        - bandwidth: the bandwidth parameter for the Gaussian kernel
        - k: the number of nearest neighbors to consider (if None, all training
        instances are used)
        - rareness_bonus: the bonus weight to give to rare features
        - rareness_steepness: the steepness of the rareness bonus function
        """

        super().__init__(name=name)
        self.distance_metric = distance_metric
        self.bandwidth = bandwidth
        self.k = k
        self.rareness_bonus = rareness_bonus
        self.rareness_steepness = rareness_steepness

    # =========================
    # TRAINING
    # =========================

    def train(self, train_df, relevant_cols: Optional[List[str]] = None):
        """Load the training data into memory, encode it as a sparse matrix, making it
        possible to compute distances between instances efficiently, and thus
        make predictions.

        Arguments:
        - train_df: the training DataFrame
        - relevant_cols: optional list of columns to use from the training DataFrame.
          If None, all columns are used.
        """

        # We filter out purely descriptive columns
        if relevant_cols is not None:
            train_df = train_df[[c for c in train_df.columns if c in relevant_cols]]
        else:
            train_df = train_df

        print("Loading training vectors in memory", end="...", flush=True)
        self.train_df = train_df

        # We encode the training data as a sparse matrix (with one-hot encoding for
        # categorical variables and min-max scaling for numerical variables)
        self.encoder = SparseEncoder()
        sparse_df = self.encoder.fit_transform(train_df)

        self.sparse_cols = sparse_df.columns.tolist()
        self.train_vectors = sparse_df.to_numpy()
        self.train_norms = np.sqrt(np.nansum((self.train_vectors**2), axis=1))
        self.feat_weights = self._get_feature_weights()

        print("Done")

        return self

    def predict_all(
        self, test_df: pandas.DataFrame, target_vars: Optional[List[str]] = None
    ):
        """Runs the forecaster on multiple target variables (based on the input
        features in test_df) and returns the results.
        If target_vars is None, predicts for all variables in self.target_vars
        that are not in test_df.columns.

        Arguments:
        - test_df: DataFrame containing the input features for prediction
        - target_vars: List of target variable names to predict. If None, predicts for
          all variables in self.target_vars that are not in test_df.columns.

        Returns:
        - pred_results: DataFrame containing the predictions for each target variable
        - prob_results: Dictionary mapping target variable names to DataFrames
          containing class probabilities (for discrete variables)
        - interval_results: Dictionary mapping target variable names to DataFrames
          containing prediction intervals (for continuous variables)
        """

        if not hasattr(self, "train_vectors"):
            raise RuntimeError("Forecaster not loaded yet")

        if target_vars is None:
            target_vars = [c for c in self.target_vars if c not in test_df.columns]

        pred_results = {}
        prob_results = {}
        interval_results = {}

        # We compute the pairwise distances between points
        distances = self.get_distances(test_df)

        for target_var in target_vars:
            # If the target variable has a precondition, we change the name of the
            # variable to reflect that conditional dependency
            full_target_var = str(target_var)
            if self.has_precondition(target_var):
                full_target_var += "|" + self.get_precondition(target_var)

            # Get the weights of training instances for each test instance
            weights = self.get_instance_weights(
                distances, target_col=target_var
            )  # shape: (num_test, num_train)

            # Classification via weighted voting
            if self.train_df[target_var].dtype in ["bool", "category"]:
                prob_results[full_target_var] = self._weighted_voting(
                    weights, target_var
                )
                prob_results[full_target_var].index = test_df.index
                pred_results[full_target_var] = prob_results[full_target_var].idxmax(
                    axis=1
                )

            # Regression via weighted average
            elif self.train_df[target_var].dtype in ["float64"]:
                preds, intervals = self._weighted_average(weights, target_var)
                pred_results[full_target_var], interval_results[full_target_var] = (
                    preds,
                    intervals,
                )
                pred_results[full_target_var].index = test_df.index
                interval_results[full_target_var].index = test_df.index

        pred_results = pandas.DataFrame.from_dict(pred_results, orient="columns")
        return pred_results, prob_results, interval_results

    def predict(self, test_df: pandas.DataFrame, target_col) -> Dict[str, Any]:
        """Predict the target column for the provided test DataFrame."""

        #      print("Predicting target column:", target_col)
        if not hasattr(self, "train_vectors"):
            raise RuntimeError("Forecaster not trained yet")

        # Checking if the target column has a precondition
        results: Dict[str, Any] = {}
        if self.has_precondition(target_col):
            results = {"precondition": self.get_precondition(target_col)}

        # We compute the pairwise distances between points
        distances = self.get_distances(test_df)

        # Get the weights of training instances for each test instance
        weights = self.get_instance_weights(
            distances, target_col=target_col
        )  # shape: (num_test, num_train)

        # Classification via weighted voting
        if self.train_df[target_col].dtype in ["bool", "category"]:
            results["probs"] = self._weighted_voting(weights, target_col)
            results["probs"].index = test_df.index
            results["preds"] = results["probs"].idxmax(axis=1)

        # Regression via weighted average
        elif self.train_df[target_col].dtype in ["float64"]:
            results["preds"], results["intervals"] = self._weighted_average(
                weights, target_col
            )
            results["preds"].index = test_df.index
            results["intervals"].index = test_df.index

        return results

    def get_distances(self, test_df: pandas.DataFrame) -> NDArray[np.float64]:
        """Compute the distance matrix between the test DataFrame and the training data.
        The computation depends on the specified distance metric (euclidean or cosine).

        Arguments:
        - test_df: the test DataFrame

        Returns:
        - A 2D numpy array of shape (num_test_instances, num_train_instances)
          containing the distances.
        """

        # We first encode the test data as a sparse matrix
        test_arr = self.encoder.transform(test_df).to_numpy()

        # Option 1: weighted euclidean distance
        if self.distance_metric == "euclidean":
            distances = get_euclidean_distances_batch(
                test_arr, self.train_vectors, self.feat_weights
            )
            return distances

        # Option 2: weighted cosine distance
        elif self.distance_metric == "cosine":
            weighted_dot_products = get_weighted_dot_products_batch(
                test_arr, self.train_vectors, self.feat_weights
            )

            # We normalise the weighted dot products to get cosine similarities
            test_norms = np.sqrt(np.nansum((test_arr**2), axis=1))
            norms_product = test_norms[:, None] * self.train_norms[None, :]

            # We convert cosine similarities to cosine distances
            cosines = weighted_dot_products / (norms_product + 1e-5)
            return 1 - cosines
        else:
            raise ValueError("Unknown distance metric: %s" % self.distance_metric)

    def get_instance_weights(
        self, distances: NDArray[np.float64], target_col: Optional[str] = None
    ) -> NDArray[np.float64]:
        """Returns a (num_test_instances x num_train_instances) matrix of weights
        for each test instance and each training instance. The weights are computed
        using a Gaussian kernel based on the distances between instances. If target_col
        is provided, training instances with missing target values are set to zero
        weight (as they cannot be used for prediction).

        Arguments:
        - distances:  2D numpy array of shape (num_test_instances, num_train_instances)
          containing the distances.
        - target_col: the target column being predicted (if any), to adjust weights
        accordingly

        Returns:
        - A 2D numpy array of shape (num_test_instances, num_train_instances)
          containing the normalised weights.
        """

        # We convert distances to Gaussian kernel weights
        weights = np.exp(-(distances**2) / (2 * (self.bandwidth**2)))

        # We penalize training instances with missing target values
        # (as they cannot be used for prediction)
        if target_col is not None:
            to_avoid = self.train_df[target_col].isna().values
            weights[:, to_avoid] *= 0

        # If the target column has a precondition, we boost training instances
        # that satisfy the precondition
        if target_col is not None and self.has_precondition(target_col):
            precondition_col = self.get_precondition(target_col)
            to_boost = self.train_df[precondition_col].astype(bool).values
            weights[:, to_boost] *= 1e5

        # If k is specified, we zero out all but the k largest weights
        if self.k is not None and self.k < weights.shape[1]:
            weight_indices_to_skip = np.argsort(weights, axis=1)[:, : -self.k]
            weights[np.arange(weights.shape[0])[:, None], weight_indices_to_skip] = 1e-8

        # We normalise the weights to sum to 1 for each instance
        weights = weights / weights.sum(axis=1, keepdims=True)
        return weights

    def get_sorted_neighbors(
        self,
        feats: osint_data.InputFeatures,
        k: Optional[int] = None,
        return_distances: bool = False,
    ) -> pandas.DataFrame:
        """Returns the k nearest neighbors for a given test instance. If k is None,
        returns all neighbors, sorted by distance. If return_distances is True,
        the returned DataFrame includes a "distance" column."""

        # We compute the distance matrix
        feats_df = feats.to_pandas_series().to_frame().T
        distances = self.get_distances(feats_df)

        # We get the k nearest neighbors
        nearest_indices = np.argsort(distances[0])[:k]

        closest_rows = self.train_df.iloc[nearest_indices]

        if return_distances:
            closest_rows = closest_rows.copy()
            closest_rows["distance"] = distances[0][nearest_indices]

        return closest_rows

    def _get_feature_weights(self):
        """Compute the weight vector for the distance metric. Since rare variables (i.e.
        those that are non-missing in only a few training instances) are more
        informative, we add a bonus weight inversely proportional to their frequency.

        This returns a 1D numpy array of shape (num_features,). The method is usually
        called only once when loading the training data."""

        weights = np.ones(len(self.sparse_cols), dtype=np.float64)

        # boosting rare variables
        for i, c in enumerate(self.sparse_cols):
            arr = self.train_vectors[:, i]
            filled_values = (~np.isnan(arr)) & (arr != 0)
            filled_frequency = filled_values.sum() / len(arr)
            weights[i] += self.rareness_bonus * np.exp(
                -self.rareness_steepness * filled_frequency
            )

        return weights

    def _weighted_voting(
        self, all_weights: NDArray[np.float64], target_col: str
    ) -> pandas.DataFrame:
        """Perform weighted voting for classification. The list of all possible
        values must be provided (to account for classes not present in the
        nearest neighbors). The method returns class probabilities.

        Arguments:
        - all_weights: a 2D numpy array of shape
                       (num_test_instances, num_train_instances)
                       containing the weights of each training instance
                       for a given test instance
        - target_col: the target column being predicted.

        Returns: a DataFrame containing the class probabilities for each test instance.
        """

        # One-hot encode the target values
        target_values = self.train_df[target_col].values
        one_hot_targets = pandas.get_dummies(target_values)

        # Compute the weighted counts for each class
        weighted_counts = all_weights @ one_hot_targets.values

        # Normalize to get probabilities
        prob_predictions = weighted_counts / weighted_counts.sum(axis=1, keepdims=True)

        # Convert to DataFrame with correct column names
        class_labels = one_hot_targets.columns.tolist()
        prob_predictions = pandas.DataFrame(
            prob_predictions, columns=class_labels, dtype=np.float64
        )

        return prob_predictions

    def _weighted_average(
        self, all_weights: NDArray[np.float64], target_col: str
    ) -> Tuple[pandas.Series, pandas.DataFrame]:
        """Perform weighted averaging for regression. The method returns both
        the predicted values as well as the prediction intervals.

        Arguments:
        - all_weights: a 2D numpy array of shape
                        (num_test_instances, num_train_instances)
                       containing the weights of each training instance
                       for a given instance.
        - target_col: the target column being predicted.

        Returns: a tuple (preds, intervals) where preds is a Series containing the
                 predicted values for each test instance, and intervals is a DataFrame
                 containing the prediction intervals."""

        target_values = self.train_df[target_col].values
        repeated_targets = np.tile(target_values[:, None], (1, len(all_weights))).T

        # Computing the median (using weighted quantiles, and ignoring NaNs)
        pred_values = np.nanquantile(
            repeated_targets, 0.5, weights=all_weights, axis=1, method="inverted_cdf"
        )

        # Computing the confidence intervals
        conf_values = (
            np.nanquantile(
                repeated_targets,
                CONF_INTERVAL_BOUNDS,
                weights=all_weights,
                axis=1,
                method="inverted_cdf",
            )
            .reshape(2, -1)
            .T
        )

        columns = [
            "%.1f %%" % (CONF_INTERVAL_BOUNDS[0] * 100),
            "%.1f %%" % (CONF_INTERVAL_BOUNDS[1] * 100),
        ]
        preds = pandas.Series(pred_values, dtype=np.float64)
        intervals = pandas.DataFrame(conf_values, columns=columns)

        return preds, intervals

    @property
    def target_vars(self) -> List[str]:
        """Returns the list of target variable names that the forecaster can predict."""

        return self.train_df.columns.tolist()


@numba.njit
def get_euclidean_distances(
    query_arr: np.ndarray,
    data_matrix: np.ndarray,
    weights: np.ndarray,
    vagueness_penalty: int = 1,
) -> np.ndarray:
    """Compute the weighted Euclidean distances between a query array and
    a data matrix, but with three twists:
    1) Missing values (NaNs) are ignored in both the query and data matrix
    2) Each dimension has a weight
    3) A vagueness penalty is added for each missing value in the data matrix
       (to avoid always picking the most incomplete records)
    """

    squarediff_sum = np.zeros(data_matrix.shape[0], dtype=np.float64)
    weights_sum = np.zeros(data_matrix.shape[0], dtype=np.float64)
    for i in range(len(query_arr)):
        query_val = query_arr[i]
        if math.isnan(query_val):
            continue
        for j in range(data_matrix.shape[0]):
            data_val = data_matrix[j, i]
            if math.isnan(data_val):
                continue
            diff = query_val - data_val
            squarediff_sum[j] += diff * diff * weights[i]
            weights_sum[j] += weights[i]

    euclidean_norm = np.sqrt(squarediff_sum / (weights_sum + 1e-5))
    distances_with_penalties = euclidean_norm + vagueness_penalty / (weights_sum + 1e-5)

    return distances_with_penalties


@numba.njit(parallel=True)
def get_euclidean_distances_batch(
    query_matrix: np.ndarray,
    data_matrix: np.ndarray,
    weights: np.ndarray,
    vagueness_penalty: int = 1,
) -> np.ndarray:
    """Compute the weighted Euclidean distances between a query matrix and
    a data matrix. Runs in parallel over the query matrix rows."""

    all_distances = np.zeros(
        (query_matrix.shape[0], len(data_matrix)), dtype=np.float64
    )

    for i in numba.prange(query_matrix.shape[0]):
        distances = get_euclidean_distances(
            query_matrix[i], data_matrix, weights, vagueness_penalty
        )
        all_distances[i] = distances

    return all_distances


@numba.njit
def get_weighted_dot_products(
    query_arr: np.ndarray, data_matrix: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Compute the weighted dot products between a query array and
    a data matrix, ignoring missing values (NaNs)."""

    weighted_dot_products = np.zeros(data_matrix.shape[0], dtype=np.float64)
    for i in range(len(query_arr)):
        query_val = query_arr[i]
        if math.isnan(query_val):
            continue
        for j in range(data_matrix.shape[0]):
            data_val = data_matrix[j, i]
            if math.isnan(data_val):
                continue
            weighted_dot_products[j] += query_val * data_val * weights[i]

    return weighted_dot_products


@numba.njit(parallel=True)
def get_weighted_dot_products_batch(
    query_matrix: np.ndarray, data_matrix: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Compute the weighted dot products between a query matrix and
    a data matrix. Runs in parallel over the query matrix rows."""

    weighted_dot_products = np.zeros(
        (query_matrix.shape[0], len(data_matrix)), dtype=np.float64
    )

    for i in numba.prange(query_matrix.shape[0]):
        weighted_dot_products[i] = get_weighted_dot_products(
            query_matrix[i], data_matrix, weights
        )
    return weighted_dot_products


class SparseEncoder:
    """Encodes a DataFrame as a sparse matrix, using one-hot encoding for
    categorical variables and min-max scaling for numerical variables.
    Ordinal variables are transformed to their ordinal values first."""

    def fit_transform(self, df: pandas.DataFrame):
        """Fits the encoder on the provided DataFrame and transforms it."""

        df = df.copy()

        # We add ordinal transformations for date, rating, overall_rating
        df = osint_data.add_ordinals(df)

        # we fit min-max scalers for numerical variables
        self.min_max_scalers = {}
        for col in df.columns:
            if df[col].dtype in ["float64"]:
                self.min_max_scalers[col] = sklearn.preprocessing.MinMaxScaler()
                transformed_numbers = self.min_max_scalers[col].fit_transform(df[[col]])
                df[col] = transformed_numbers
            if pandas.api.types.is_datetime64_any_dtype(df[col]):
                del df[col]

        # We one-hot encode categorical variables
        self.cat_values = {}
        new_columns = {}
        for c in df.columns:
            if df[c].dtype in ["str", "object", "category"]:
                values = sorted(df[c].dropna().unique().tolist())
                for v in values:
                    new_columns[f"{c}_{v}"] = (df[c] == v).astype(np.float64)
                self.cat_values[c] = values
                del df[c]
                if "ordinal" in c:
                    print("uh??", c)

        df = pandas.concat((df, pandas.DataFrame(new_columns)), axis=1)
        self.sparse_cols = df.columns.tolist()
        df = df.astype(np.float64)

        return df

    def transform(self, df: pandas.DataFrame):
        """Transforms the provided DataFrame using the fitted encoder."""

        df = df.copy()

        # We add ordinal transformations for date, rating, overall_rating
        df = osint_data.add_ordinals(df)

        # We one-hot encode categorical variables
        cat_columns_to_expand = [c for c in self.cat_values.keys() if c in df.columns]
        expanded_df = pandas.get_dummies(
            df, columns=cat_columns_to_expand, dtype=np.float64
        )
        expanded_df = expanded_df[
            [
                c
                for c in self.sparse_cols
                if c in expanded_df.columns and c not in self.min_max_scalers
            ]
        ]
        expanded_df = expanded_df.astype(np.float64)

        # All variables not present in the expanded DataFrame are filled with NaNs
        new_data = np.full((len(df), len(self.sparse_cols)), np.nan, dtype=np.float64)
        new_df = pandas.DataFrame(
            new_data, columns=self.sparse_cols, index=df.index, dtype=np.float64
        )
        new_df.update(expanded_df)

        # We transform numerical variables using the fitted min-max scalers
        for col in self.min_max_scalers:
            if col in df.columns:
                s = self.min_max_scalers[col].transform(df[col].to_frame())
                new_df[col] = s.flatten()
        return new_df
