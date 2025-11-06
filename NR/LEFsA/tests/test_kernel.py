# -*- coding: utf-8 -*-

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from lefsa.osint_kernel import KernelForecaster, SparseEncoder


@pytest.fixture
def sample_training_data():
    """Create sample training data for testing."""
    np.random.seed(42)
    n_samples = 50

    data = {
        "feature1": np.random.randn(n_samples),
        "feature2": np.random.randint(0, 5, n_samples),
        "feature3": pd.Categorical(np.random.choice(["A", "B", "C"], n_samples)),
        "target_binary": np.random.choice([True, False], n_samples),
        "target_multiclass": pd.Categorical(
            np.random.choice(["cat1", "cat2", "cat3"], n_samples)
        ),
        "target_regression": np.random.randn(n_samples) * 10 + 50,
        "condition_var": np.random.choice([True, False], n_samples, p=[0.3, 0.7]),
    }

    df = pd.DataFrame(data)
    # Ensure proper dtypes
    df["target_multiclass"] = df["target_multiclass"].astype("category")
    df["target_regression"] = df["target_regression"].astype("float64")

    return df


@pytest.fixture
def sample_test_data():
    """Create sample test data for predictions."""
    np.random.seed(123)
    n_samples = 10

    data = {
        "feature1": np.random.randn(n_samples),
        "feature2": np.random.randint(0, 5, n_samples),
        "feature3": pd.Categorical(np.random.choice(["A", "B", "C"], n_samples)),
    }

    return pd.DataFrame(data)


@pytest.fixture
def mock_input_features():
    """Mock InputFeatures object for testing."""
    mock_features = Mock()
    mock_features.to_pandas_series.return_value = pd.Series(
        {"feature1": 1.5, "feature2": 2, "feature3": "A"}
    )
    return mock_features


class TestKernelForecaster:
    """Test cases for the KernelForecaster class."""

    def test_init_default(self):
        """Test KernelForecaster initialization with default parameters."""
        forecaster = KernelForecaster()

        assert forecaster.name == "kernel_forecaster"
        assert forecaster.distance_metric == "euclidean"
        assert forecaster.bandwidth == 1.0
        assert forecaster.k == 100
        assert forecaster.rareness_bonus == 10
        assert forecaster.rareness_steepness == 10

    def test_init_custom(self):
        """Test KernelForecaster initialization with custom parameters."""
        forecaster = KernelForecaster(
            name="custom_kernel",
            distance_metric="cosine",
            bandwidth=2.5,
            k=50,
            rareness_bonus=5,
            rareness_steepness=20,
        )

        assert forecaster.name == "custom_kernel"
        assert forecaster.distance_metric == "cosine"
        assert forecaster.bandwidth == 2.5
        assert forecaster.k == 50
        assert forecaster.rareness_bonus == 5
        assert forecaster.rareness_steepness == 20

    @patch("lefsa.osint_kernel.osint_data.add_ordinals")
    def test_train_basic(self, mock_add_ordinals, sample_training_data):
        """Test basic training functionality."""
        mock_add_ordinals.return_value = sample_training_data

        forecaster = KernelForecaster()

        # Suppress print output for testing
        with patch("builtins.print"):
            result = forecaster.train(sample_training_data)

        assert result is forecaster  # Should return self
        assert hasattr(forecaster, "train_df")
        assert hasattr(forecaster, "encoder")
        assert hasattr(forecaster, "sparse_cols")
        assert hasattr(forecaster, "train_vectors")
        assert hasattr(forecaster, "train_norms")
        assert hasattr(forecaster, "feat_weights")

        # Check that training data is stored
        assert len(forecaster.train_df) == len(sample_training_data)

    @patch("lefsa.osint_kernel.osint_data.add_ordinals")
    def test_train_with_relevant_cols(self, mock_add_ordinals, sample_training_data):
        """Test training with subset of columns."""
        mock_add_ordinals.return_value = sample_training_data[
            ["feature1", "feature2", "target_binary"]
        ]

        forecaster = KernelForecaster()
        relevant_cols = ["feature1", "feature2", "target_binary"]

        with patch("builtins.print"):
            forecaster.train(sample_training_data, relevant_cols=relevant_cols)

        # Should only have the relevant columns
        assert set(forecaster.train_df.columns) == set(relevant_cols)

    def test_predict_untrained_error(self, sample_test_data):
        """Test that predicting without training raises an error."""
        forecaster = KernelForecaster()

        with pytest.raises(RuntimeError, match="Forecaster not trained yet"):
            forecaster.predict(sample_test_data, "target_binary")

    @patch("lefsa.osint_kernel.osint_data.add_ordinals")
    def test_predict_binary_classification(
        self, mock_add_ordinals, sample_training_data, sample_test_data
    ):
        """Test prediction for binary classification."""
        mock_add_ordinals.side_effect = lambda x: x  # Return input unchanged

        forecaster = KernelForecaster(bandwidth=1.0, k=10)

        with patch("builtins.print"):
            forecaster.train(sample_training_data)

        results = forecaster.predict(sample_test_data, "target_binary")

        assert "preds" in results
        assert "probs" in results
        assert len(results["preds"]) == len(sample_test_data)
        assert results["probs"].shape[0] == len(sample_test_data)
        assert results["probs"].shape[1] == 2  # Binary classification

    @patch("lefsa.osint_kernel.osint_data.add_ordinals")
    def test_predict_multiclass_classification(
        self, mock_add_ordinals, sample_training_data, sample_test_data
    ):
        """Test prediction for multiclass classification."""
        mock_add_ordinals.side_effect = lambda x: x

        forecaster = KernelForecaster(bandwidth=1.0, k=10)

        with patch("builtins.print"):
            forecaster.train(sample_training_data)

        results = forecaster.predict(sample_test_data, "target_multiclass")

        assert "preds" in results
        assert "probs" in results
        assert len(results["preds"]) == len(sample_test_data)
        assert results["probs"].shape[0] == len(sample_test_data)
        assert results["probs"].shape[1] == 3  # Three classes

    @patch("lefsa.osint_kernel.osint_data.add_ordinals")
    def test_predict_regression(
        self, mock_add_ordinals, sample_training_data, sample_test_data
    ):
        """Test prediction for regression."""
        mock_add_ordinals.side_effect = lambda x: x

        forecaster = KernelForecaster(bandwidth=1.0, k=10)

        with patch("builtins.print"):
            forecaster.train(sample_training_data)

        results = forecaster.predict(sample_test_data, "target_regression")

        assert "preds" in results
        assert "intervals" in results
        assert len(results["preds"]) == len(sample_test_data)
        assert results["intervals"].shape[0] == len(sample_test_data)
        assert results["intervals"].shape[1] == 2  # Lower and upper bounds

    @patch("lefsa.osint_kernel.osint_data.add_ordinals")
    def test_predict_with_precondition(
        self, mock_add_ordinals, sample_training_data, sample_test_data
    ):
        """Test prediction with precondition."""
        mock_add_ordinals.side_effect = lambda x: x

        forecaster = KernelForecaster(bandwidth=1.0, k=10)
        forecaster.has_precondition = Mock(return_value=True)
        forecaster.get_precondition = Mock(return_value="condition_var")

        with patch("builtins.print"):
            forecaster.train(sample_training_data)

        results = forecaster.predict(sample_test_data, "target_binary")

        assert "precondition" in results
        assert results["precondition"] == "condition_var"
        assert "preds" in results
        assert "probs" in results

    @patch("lefsa.osint_kernel.osint_data.add_ordinals")
    def test_predict_all(
        self, mock_add_ordinals, sample_training_data, sample_test_data
    ):
        """Test predict_all method."""
        mock_add_ordinals.side_effect = lambda x: x

        forecaster = KernelForecaster(bandwidth=1.0, k=10)

        with patch("builtins.print"):
            forecaster.train(sample_training_data)

        # Test with specific target vars
        target_vars = ["target_binary", "target_regression"]
        pred_results, prob_results, interval_results = forecaster.predict_all(
            sample_test_data, target_vars=target_vars
        )

        assert len(pred_results.columns) == 2
        assert "target_binary" in pred_results.columns
        assert "target_regression" in pred_results.columns
        assert "target_binary" in prob_results
        assert "target_regression" in interval_results

    @patch("lefsa.osint_kernel.osint_data.add_ordinals")
    def test_predict_all_untrained_error(self, mock_add_ordinals, sample_test_data):
        """Test predict_all raises error when not trained."""
        forecaster = KernelForecaster()

        with pytest.raises(RuntimeError, match="Forecaster not loaded yet"):
            forecaster.predict_all(sample_test_data)

    @patch("lefsa.osint_kernel.osint_data.add_ordinals")
    def test_get_distances_euclidean(
        self, mock_add_ordinals, sample_training_data, sample_test_data
    ):
        """Test distance computation with euclidean metric."""
        mock_add_ordinals.side_effect = lambda x: x

        forecaster = KernelForecaster(distance_metric="euclidean", bandwidth=1.0)

        with patch("builtins.print"):
            forecaster.train(sample_training_data)

        distances = forecaster.get_distances(sample_test_data)

        assert distances.shape == (len(sample_test_data), len(sample_training_data))
        assert distances.dtype == np.float64
        assert np.all(distances >= 0)  # Distances should be non-negative

    @patch("lefsa.osint_kernel.osint_data.add_ordinals")
    def test_get_distances_cosine(
        self, mock_add_ordinals, sample_training_data, sample_test_data
    ):
        """Test distance computation with cosine metric."""
        mock_add_ordinals.side_effect = lambda x: x

        forecaster = KernelForecaster(distance_metric="cosine", bandwidth=1.0)

        with patch("builtins.print"):
            forecaster.train(sample_training_data)

        distances = forecaster.get_distances(sample_test_data)

        assert distances.shape == (len(sample_test_data), len(sample_training_data))
        assert distances.dtype == np.float64

    def test_get_distances_invalid_metric(self):
        """Test that invalid distance metric raises error."""
        forecaster = KernelForecaster()
        forecaster.distance_metric = "invalid"

        # We need to simulate a trained forecaster first
        with patch(
            "lefsa.osint_kernel.osint_data.add_ordinals", side_effect=lambda x: x
        ):
            # Create minimal training data
            train_data = pd.DataFrame(
                {"feature1": [1.0, 2.0, 3.0], "target": [True, False, True]}
            )

            with patch("builtins.print"):
                forecaster.train(train_data)

            # Now test with invalid metric
            forecaster.distance_metric = "invalid"
            test_data = pd.DataFrame({"feature1": [1.5]})

            with pytest.raises(ValueError, match="Unknown distance metric"):
                forecaster.get_distances(test_data)

    @patch("lefsa.osint_kernel.osint_data.add_ordinals")
    def test_get_instance_weights(self, mock_add_ordinals, sample_training_data):
        """Test instance weight computation."""
        mock_add_ordinals.side_effect = lambda x: x

        forecaster = KernelForecaster(bandwidth=2.0, k=5)

        with patch("builtins.print"):
            forecaster.train(sample_training_data)

        # Create mock distances
        distances = np.random.rand(3, len(sample_training_data))

        weights = forecaster.get_instance_weights(distances, target_col="target_binary")

        assert weights.shape == distances.shape
        # Check that weights sum to 1 for each test instance
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, rtol=1e-10)

    @patch("lefsa.osint_kernel.osint_data.add_ordinals")
    def test_get_nearest_neighbors(
        self, mock_add_ordinals, sample_training_data, mock_input_features
    ):
        """Test get_nearest_neighbors method."""
        mock_add_ordinals.side_effect = lambda x: x

        forecaster = KernelForecaster(bandwidth=1.0)

        with patch("builtins.print"):
            forecaster.train(sample_training_data)

        # Test without distances
        neighbors = forecaster.get_sorted_neighbors(mock_input_features, k=5)
        assert len(neighbors) == 5
        assert isinstance(neighbors, pd.DataFrame)

        # Test with distances
        neighbors = forecaster.get_sorted_neighbors(
            mock_input_features, k=5, return_distances=True
        )
        assert len(neighbors) == 5
        assert "distance" in neighbors.columns

    @patch("lefsa.osint_kernel.osint_data.add_ordinals")
    def test_target_vars_property(self, mock_add_ordinals, sample_training_data):
        """Test target_vars property."""
        mock_add_ordinals.side_effect = lambda x: x

        forecaster = KernelForecaster()

        with patch("builtins.print"):
            forecaster.train(sample_training_data)

        target_vars = forecaster.target_vars
        assert isinstance(target_vars, list)
        assert set(target_vars) == set(sample_training_data.columns)

    @patch("lefsa.osint_kernel.osint_data.add_ordinals")
    def test_k_parameter_limiting(
        self, mock_add_ordinals, sample_training_data, sample_test_data
    ):
        """Test that k parameter correctly limits neighbors."""
        mock_add_ordinals.side_effect = lambda x: x

        # Use small k to ensure limiting behavior
        forecaster = KernelForecaster(k=3, bandwidth=1.0)

        with patch("builtins.print"):
            forecaster.train(sample_training_data)

        distances = forecaster.get_distances(sample_test_data)
        weights = forecaster.get_instance_weights(distances, target_col="target_binary")

        # Each test instance should have at most k non-zero weights
        for i in range(weights.shape[0]):
            non_zero_weights = np.sum(
                weights[i] > 1e-7
            )  # Account for small epsilon values
            assert non_zero_weights <= 3  # k=3

    @patch("lefsa.osint_kernel.osint_data.add_ordinals")
    def test_predict_all_with_condition(
        self, mock_add_ordinals, sample_training_data, sample_test_data
    ):
        """Test predict_all with preconditioned targets."""
        mock_add_ordinals.side_effect = lambda x: x

        forecaster = KernelForecaster(bandwidth=1.0, k=10)
        forecaster.has_precondition = Mock(side_effect=lambda x: x == "target_binary")
        forecaster.get_precondition = Mock(return_value="condition_var")

        with patch("builtins.print"):
            forecaster.train(sample_training_data)

        target_vars = ["target_binary", "target_regression"]
        pred_results, prob_results, interval_results = forecaster.predict_all(
            sample_test_data, target_vars=target_vars
        )

        # Check that conditioned target has modified name
        assert "target_binary|condition_var" in pred_results.columns
        assert "target_regression" in pred_results.columns

    @patch("lefsa.osint_kernel.osint_data.add_ordinals")
    def test_predict_with_missing_values(
        self, mock_add_ordinals, sample_training_data, sample_test_data
    ):
        """Test prediction with missing values in target column."""
        mock_add_ordinals.side_effect = lambda x: x

        # Create training data with missing values in the regression target
        train_data_with_na = sample_training_data.copy()
        train_data_with_na.loc[:5, "target_regression"] = np.nan

        forecaster = KernelForecaster(bandwidth=1.0, k=10)

        with patch("builtins.print"):
            forecaster.train(train_data_with_na)

        results = forecaster.predict(sample_test_data, "target_regression")

        # Should still work despite missing values
        assert "preds" in results
        assert "intervals" in results
        assert len(results["preds"]) == len(sample_test_data)

    @patch("lefsa.osint_kernel.osint_data.add_ordinals")
    def test_bandwidth_effect_on_predictions(
        self, mock_add_ordinals, sample_training_data, sample_test_data
    ):
        """Test that different bandwidth values affect predictions."""
        mock_add_ordinals.side_effect = lambda x: x

        # Train with small bandwidth (more local)
        forecaster_small = KernelForecaster(bandwidth=0.1, k=10)
        with patch("builtins.print"):
            forecaster_small.train(sample_training_data)

        # Train with large bandwidth (more global)
        forecaster_large = KernelForecaster(bandwidth=10.0, k=10)
        with patch("builtins.print"):
            forecaster_large.train(sample_training_data)

        # Get distances for same test data
        distances = forecaster_small.get_distances(sample_test_data)
        weights_small = forecaster_small.get_instance_weights(
            distances, target_col="target_binary"
        )
        weights_large = forecaster_large.get_instance_weights(
            distances, target_col="target_binary"
        )

        # Small bandwidth should have more concentrated weights
        # (higher variance in weights)
        assert np.var(weights_small) > np.var(weights_large)


class TestSparseEncoder:
    """Test cases for the SparseEncoder class."""

    def test_fit_transform_basic(self, sample_training_data):
        """Test basic fit_transform functionality."""
        encoder = SparseEncoder()

        with patch(
            "lefsa.osint_kernel.osint_data.add_ordinals",
            return_value=sample_training_data,
        ):
            result = encoder.fit_transform(sample_training_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_training_data)
        assert hasattr(encoder, "sparse_cols")
        assert hasattr(encoder, "min_max_scalers")
        assert hasattr(encoder, "cat_values")

    def test_transform_after_fit(self, sample_training_data, sample_test_data):
        """Test transform method after fitting."""
        encoder = SparseEncoder()

        with patch(
            "lefsa.osint_kernel.osint_data.add_ordinals", side_effect=lambda x: x
        ):
            encoder.fit_transform(sample_training_data)
            result = encoder.transform(sample_test_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_test_data)
        assert result.columns.tolist() == encoder.sparse_cols


class TestDistanceFunctions:
    """Test cases for the distance computation functions."""

    def test_euclidean_distances_batch_shape(self):
        """Test that euclidean distance batch function returns correct shape."""
        from lefsa.osint_kernel import get_euclidean_distances_batch

        query_matrix = np.random.randn(5, 3)
        data_matrix = np.random.randn(10, 3)
        weights = np.ones(3)

        distances = get_euclidean_distances_batch(query_matrix, data_matrix, weights)

        assert distances.shape == (5, 10)
        assert distances.dtype == np.float64

    def test_weighted_dot_products_batch_shape(self):
        """Test that weighted dot products batch function returns correct shape."""
        from lefsa.osint_kernel import get_weighted_dot_products_batch

        query_matrix = np.random.randn(5, 3)
        data_matrix = np.random.randn(10, 3)
        weights = np.ones(3)

        dot_products = get_weighted_dot_products_batch(
            query_matrix, data_matrix, weights
        )

        assert dot_products.shape == (5, 10)
        assert dot_products.dtype == np.float64
        assert dot_products.dtype == np.float64
