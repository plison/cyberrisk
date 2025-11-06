import os
import shutil
import sys
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.append("/nr/samba/shared/CyberRisk/code/LEF/lefsa")

from lefsa.osint_base import CONF_INTERVAL_BOUNDS, DummyForecaster, DummyPredictor
from lefsa.osint_gluon import GluonForecaster, GluonPredictor


@pytest.fixture
def sample_training_data():
    """Create sample training data for testing."""
    np.random.seed(42)
    n_samples = 100

    data = {
        "feature1": np.random.randn(n_samples),
        "feature2": np.random.randint(0, 5, n_samples),
        "feature3": np.random.choice(["A", "B", "C"], n_samples),
        "target_binary": np.random.choice([True, False], n_samples),
        "target_multiclass": pd.Categorical(
            np.random.choice(["cat1", "cat2", "cat3"], n_samples)
        ),
        "target_regression": np.random.randn(n_samples) * 10 + 50,
        "condition_var": np.random.choice([True, False], n_samples, p=[0.3, 0.7]),
    }

    df = pd.DataFrame(data)
    # Ensure target_multiclass is categorical
    df["target_multiclass"] = df["target_multiclass"].astype("category")
    return df


@pytest.fixture
def sample_test_data():
    """Create sample test data for predictions."""
    np.random.seed(123)
    n_samples = 20

    data = {
        "feature1": np.random.randn(n_samples),
        "feature2": np.random.randint(0, 5, n_samples),
        "feature3": np.random.choice(["A", "B", "C"], n_samples),
        "condition_var": np.random.choice([True, False], n_samples),
    }

    return pd.DataFrame(data)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


class TestDummyPredictor:
    """Test cases for DummyPredictor class."""

    def test_dummy_predictor_classification(self, sample_training_data):
        """Test DummyPredictor with classification data."""
        target_series = sample_training_data["target_binary"]
        predictor = DummyPredictor(target_series)

        assert predictor.problem_type == "classification"
        assert predictor.value in [True, False]
        assert hasattr(predictor, "value_distrib")
        assert set(predictor.value_distrib.index) == {True, False}

        # Test prediction
        test_data = pd.DataFrame({"dummy": [1, 2, 3]})
        predictions = predictor.predict(test_data)

        assert len(predictions) == 3
        assert all(predictions == predictor.value)

    def test_dummy_predictor_regression(self, sample_training_data):
        """Test DummyPredictor with regression data."""
        target_series = sample_training_data["target_regression"]
        predictor = DummyPredictor(target_series)

        assert predictor.problem_type == "regression"
        assert hasattr(predictor, "value")
        assert hasattr(predictor, "lower_bound")
        assert hasattr(predictor, "upper_bound")

        # Test prediction
        test_data = pd.DataFrame({"dummy": [1, 2, 3]})
        predictions = predictor.predict(test_data)

        assert len(predictions) == 3
        assert all(predictions == predictor.value)

    def test_dummy_predictor_proba(self, sample_training_data):
        """Test probability predictions for classification."""
        target_series = sample_training_data["target_multiclass"]
        predictor = DummyPredictor(target_series)

        test_data = pd.DataFrame({"dummy": [1, 2]})
        proba = predictor.predict_proba(test_data)

        assert proba.shape == (2, 3)  # 2 samples, 3 classes
        assert set(proba.columns) == {"cat1", "cat2", "cat3"}

        # Should sum to 1 for each row
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_dummy_predictor_intervals(self, sample_training_data):
        """Test prediction intervals for regression."""
        target_series = sample_training_data["target_regression"]
        predictor = DummyPredictor(target_series)

        test_data = pd.DataFrame({"dummy": [1, 2]})
        intervals = predictor.predict_interval(test_data)

        assert intervals.shape == (2, 2)  # 2 samples, 2 bounds
        assert len(intervals.columns) == 2

        # Lower bound should be less than upper bound
        low_col = intervals.columns[0]
        high_col = intervals.columns[1]
        assert all(intervals[low_col] <= intervals[high_col])

    def test_dummy_predictor_wrong_method_calls(self, sample_training_data):
        """Test error handling for wrong method calls."""
        # Classification predictor
        target_series = sample_training_data["target_binary"]
        class_predictor = DummyPredictor(target_series)

        test_data = pd.DataFrame({"dummy": [1]})

        with pytest.raises(RuntimeError, match="Prediction intervals not available"):
            class_predictor.predict_interval(test_data)

        # Regression predictor
        target_series = sample_training_data["target_regression"]
        reg_predictor = DummyPredictor(target_series)

        with pytest.raises(RuntimeError, match="Probability predictions not available"):
            reg_predictor.predict_proba(test_data)


class TestDummyForecaster:
    """Test cases for DummyForecaster class."""

    def test_init_default(self):
        """Test DummyForecaster initialization with defaults."""
        forecaster = DummyForecaster()

        assert forecaster.name == "dummy_forecaster"
        assert forecaster.predictors == {}

    def test_init_custom(self):
        """Test DummyForecaster initialization with custom name."""
        forecaster = DummyForecaster(name="custom_dummy")

        assert forecaster.name == "custom_dummy"

    def test_train_dummy_model(self, sample_training_data, temp_dir):
        """Test training with DummyForecaster."""
        forecaster = DummyForecaster(name="test_forecaster")

        target_cols = ["target_binary", "target_regression"]

        forecaster.train(sample_training_data, target_cols)

        assert forecaster.target_cols == target_cols
        assert len(forecaster.predictors) == 2

        for target_col in target_cols:
            assert target_col in forecaster.predictors
            assert isinstance(forecaster.predictors[target_col], DummyPredictor)

    def test_predict_dummy_predictor(
        self, sample_training_data, sample_test_data, temp_dir
    ):
        """Test prediction with DummyForecaster."""
        forecaster = DummyForecaster(name="test_forecaster")

        target_cols = ["target_binary", "target_regression"]

        forecaster.train(sample_training_data, target_cols)

        # Test binary classification prediction
        binary_results = forecaster.predict(sample_test_data, "target_binary")

        assert "preds" in binary_results
        assert "probs" in binary_results
        assert len(binary_results["preds"]) == len(sample_test_data)

        # Test regression prediction
        reg_results = forecaster.predict(sample_test_data, "target_regression")

        assert "preds" in reg_results
        assert "intervals" in reg_results
        assert len(reg_results["preds"]) == len(sample_test_data)

    def test_save_and_load(self, sample_training_data, temp_dir):
        """Test saving and loading DummyForecaster."""
        forecaster = DummyForecaster(name="test_forecaster")

        target_cols = ["target_binary"]
        forecaster.train(sample_training_data, target_cols)

        # Save forecaster
        save_path = os.path.join(temp_dir, "dummy_forecaster.pkl")
        forecaster.save(save_path)

        # Check that forecaster file exists
        assert os.path.exists(save_path)

        # Test loading
        loaded_forecaster = DummyForecaster.load(save_path)

        assert loaded_forecaster.name == forecaster.name
        assert loaded_forecaster.target_cols == forecaster.target_cols


class TestGluonForecaster:
    """Test cases for GluonForecaster class."""

    def test_init_default(self):
        """Test GluonForecaster initialization with defaults."""
        forecaster = GluonForecaster()

        assert forecaster.name == "gluon_forecaster"
        assert forecaster.base_path == "gluon_forecaster"
        assert forecaster.model_type == "gbm"
        assert forecaster.predictors == {}

    def test_init_custom(self):
        """Test GluonForecaster initialization with custom parameters."""
        forecaster = GluonForecaster(
            name="custom_forecaster", base_path="/tmp/custom_path", model_type="gbm"
        )

        assert forecaster.name == "custom_forecaster"
        assert forecaster.base_path == "/tmp/custom_path"
        assert forecaster.model_type == "gbm"

    def test_train_single_value_target(self, temp_dir):
        """Test training when target has only one unique value."""
        # Create data with single-value target
        data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "target": [True, True, True, True, True],  # Only one value
            }
        )

        forecaster = GluonForecaster(base_path=temp_dir, model_type="gbm")
        forecaster.train(data, ["feature1"], ["target"])

        assert isinstance(forecaster.predictors["target"], DummyPredictor)

    def test_train_with_precondition(self, sample_training_data, temp_dir):
        """Test training with conditional targets."""
        forecaster = GluonForecaster(base_path=temp_dir, model_type="gbm")

        # Mock precondition methods
        forecaster.has_precondition = Mock(return_value=True)
        forecaster.get_precondition = Mock(return_value="condition_var")

        input_cols = ["feature1", "feature2"]
        target_cols = ["target_binary"]

        forecaster.train(sample_training_data, input_cols, target_cols)

        # Should have filtered data based on condition
        assert "target_binary" in forecaster.predictors

    def test_train_empty_filtered_data(self, temp_dir):
        """Test training when filtering results in empty data."""
        data = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "target": [True, False, True],
                "condition": [False, False, False],  # All False
            }
        )

        forecaster = GluonForecaster(base_path=temp_dir, model_type="gbm")
        forecaster.has_precondition = Mock(return_value=True)
        forecaster.get_precondition = Mock(return_value="condition")

        forecaster.train(data, ["feature1"], ["target"])

        # Should not create predictor for empty filtered data
        assert "target" not in forecaster.predictors

    def test_predict_with_single_value_target(self, sample_test_data, temp_dir):
        """Test prediction when target has single value (uses DummyPredictor)."""
        # Create data with single-value targets to trigger DummyPredictor usage
        train_data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [1, 2, 3, 4, 5],
                "feature3": ["A", "B", "A", "B", "A"],
                "target_binary": [True, True, True, True, True],  # Single value
                "target_regression": [5.0, 5.0, 5.0, 5.0, 5.0],  # Single value
            }
        )

        forecaster = GluonForecaster(base_path=temp_dir, model_type="gbm")

        input_cols = ["feature1", "feature2", "feature3"]
        target_cols = ["target_binary", "target_regression"]

        forecaster.train(train_data, input_cols, target_cols)

        # Both should use DummyPredictor due to single values
        assert isinstance(forecaster.predictors["target_binary"], DummyPredictor)
        assert isinstance(forecaster.predictors["target_regression"], DummyPredictor)

        # Test binary classification prediction
        binary_results = forecaster.predict(sample_test_data, "target_binary")

        assert "preds" in binary_results
        assert "probs" in binary_results
        assert len(binary_results["preds"]) == len(sample_test_data)

        # Test regression prediction
        reg_results = forecaster.predict(sample_test_data, "target_regression")

        assert "preds" in reg_results
        assert "intervals" in reg_results
        assert len(reg_results["preds"]) == len(sample_test_data)

    def test_predict_with_precondition(
        self, sample_training_data, sample_test_data, temp_dir
    ):
        """Test prediction with precondition."""
        forecaster = GluonForecaster(base_path=temp_dir, model_type="gbm")
        forecaster.has_precondition = Mock(return_value=True)
        forecaster.get_precondition = Mock(return_value="condition_var")

        forecaster.train(sample_training_data, ["feature1"], ["target_binary"])

        results = forecaster.predict(sample_test_data, "target_binary")

        assert "precondition" in results
        assert results["precondition"] == "condition_var"

    def test_predict_untrained_target(self, sample_test_data, temp_dir):
        """Test prediction with untrained target raises error."""
        forecaster = GluonForecaster(base_path=temp_dir)

        with pytest.raises(
            RuntimeError, match="Predictor for untrained_target not trained yet"
        ):
            forecaster.predict(sample_test_data, "untrained_target")

    def test_save_and_load(self, temp_dir):
        """Test saving and loading forecaster."""
        # Create data with single-value target to avoid AutoGluon training
        train_data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [1, 2, 3, 4, 5],
                "target_binary": [True, True, True, True, True],  # Single value
            }
        )

        forecaster = GluonForecaster(
            name="test_forecaster", base_path=temp_dir, model_type="gbm"
        )

        input_cols = ["feature1", "feature2"]
        target_cols = ["target_binary"]

        forecaster.train(train_data, input_cols, target_cols)
        forecaster.save()

        # Check that forecaster file exists
        forecaster_file = os.path.join(temp_dir, "forecaster.pkl")
        assert os.path.exists(forecaster_file)

        # Test loading
        loaded_forecaster = GluonForecaster.load(temp_dir)

        assert loaded_forecaster.name == forecaster.name
        assert loaded_forecaster.base_path == forecaster.base_path
        assert loaded_forecaster.model_type == forecaster.model_type
        assert loaded_forecaster.input_cols == forecaster.input_cols
        assert loaded_forecaster.target_cols == forecaster.target_cols

    def test_target_vars_property(self, temp_dir):
        """Test target_vars property."""
        # Create simple data to avoid AutoGluon training
        train_data = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "target_binary": [True, True, True],
                "target_regression": [1.0, 1.0, 1.0],
            }
        )

        forecaster = GluonForecaster(base_path=temp_dir, model_type="gbm")

        target_cols = ["target_binary", "target_regression"]
        forecaster.train(train_data, ["feature1"], target_cols)

        assert forecaster.target_vars == target_cols

    def test_predict_with_lazy_loading(
        self, sample_training_data, sample_test_data, temp_dir
    ):
        """Test prediction with lazy loading of predictors."""
        forecaster = GluonForecaster(base_path=temp_dir, model_type="gbm")

        # Create simple data to get a DummyPredictor
        train_data = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "target": [True, True, True],  # All same value -> DummyPredictor
            }
        )

        forecaster.train(train_data, ["feature1"], ["target"])

        # Simulate lazy loading by replacing the DummyPredictor with a path string
        original_predictor = forecaster.predictors["target"]
        forecaster.predictors["target"] = "/fake/path/to/predictor"

        # Mock the GluonPredictor.load method to return the original predictor
        with patch(
            "lefsa.osint_gluon.GluonPredictor.load", return_value=original_predictor
        ):
            test_data = pd.DataFrame({"feature1": [1, 2]})
            results = forecaster.predict(test_data, "target")

            # Check that the predictor was loaded and the path replaced
            assert forecaster.predictors["target"] == original_predictor
            assert "preds" in results
            assert len(results["preds"]) == 2


class TestGluonPredictor:
    """Test cases for GluonPredictor class."""

    @patch("autogluon.tabular.TabularPredictor")
    def test_gluon_predictor_fit_binary(self, mock_predictor_class):
        """Test GluonPredictor fit for binary classification."""
        mock_predictor = Mock()
        mock_predictor_class.return_value = mock_predictor

        # Create binary classification data
        train_df = pd.DataFrame(
            {"feature1": [1, 2, 3, 4], "target": [True, False, True, False]}
        )

        gluon_pred = GluonPredictor()
        result = gluon_pred.fit(train_df, "target", "/tmp/test_path", "gbm")

        # Check that TabularPredictor was created with correct parameters
        mock_predictor_class.assert_called_once()
        call_kwargs = mock_predictor_class.call_args[1]
        assert call_kwargs["label"] == "target"
        assert call_kwargs["problem_type"] == "binary"

        # Check that fit was called
        mock_predictor.fit.assert_called_once()
        mock_predictor.delete_models.assert_called_once()
        mock_predictor.save_space.assert_called_once()

        assert result is gluon_pred
        assert gluon_pred.predictor is mock_predictor

    @patch("autogluon.tabular.TabularPredictor")
    def test_gluon_predictor_fit_multiclass(self, mock_predictor_class):
        """Test GluonPredictor fit for multiclass classification."""
        mock_predictor = Mock()
        mock_predictor_class.return_value = mock_predictor

        # Create multiclass classification data with categorical dtype
        train_df = pd.DataFrame(
            {"feature1": [1, 2, 3, 4], "target": pd.Categorical(["A", "B", "A", "C"])}
        )

        gluon_pred = GluonPredictor()
        gluon_pred.fit(train_df, "target", "/tmp/test_path", "gbm")

        call_kwargs = mock_predictor_class.call_args[1]
        assert call_kwargs["problem_type"] == "multiclass"

    @patch("autogluon.tabular.TabularPredictor")
    def test_gluon_predictor_fit_regression(self, mock_predictor_class):
        """Test GluonPredictor fit for regression."""
        mock_predictor = Mock()
        mock_predictor_class.return_value = mock_predictor

        # Create regression data
        train_df = pd.DataFrame(
            {"feature1": [1, 2, 3, 4], "target": [1.5, 2.3, 3.1, 4.7]}
        )

        gluon_pred = GluonPredictor()
        gluon_pred.fit(train_df, "target", "/tmp/test_path", "gbm")

        call_kwargs = mock_predictor_class.call_args[1]
        assert call_kwargs["problem_type"] == "quantile"
        assert call_kwargs["quantile_levels"] == [
            CONF_INTERVAL_BOUNDS[0],
            0.5,
            CONF_INTERVAL_BOUNDS[1],
        ]

    @patch("autogluon.tabular.TabularPredictor")
    def test_gluon_predictor_fit_gbm(self, mock_predictor_class):
        """Test GluonPredictor fit with GBM model type."""
        mock_predictor = Mock()
        mock_predictor_class.return_value = mock_predictor

        train_df = pd.DataFrame(
            {"feature1": [1, 2, 3, 4], "target": [True, False, True, False]}
        )

        gluon_pred = GluonPredictor()
        gluon_pred.fit(train_df, "target", "/tmp/test_path", "gbm")

        # Check that fit was called with GBM-specific arguments
        fit_kwargs = mock_predictor.fit.call_args[1]
        assert fit_kwargs["hyperparameters"] == {"GBM": {}}
        assert not fit_kwargs["fit_weighted_ensemble"]
        assert not fit_kwargs["fit_full_last_level_weighted_ensemble"]

    @patch("os.path.exists")
    @patch("shutil.rmtree")
    @patch("autogluon.tabular.TabularPredictor")
    def test_gluon_predictor_fit_existing_path(
        self, mock_predictor_class, mock_rmtree, mock_exists
    ):
        """Test GluonPredictor fit when path already exists."""
        mock_exists.return_value = True
        mock_predictor = Mock()
        mock_predictor_class.return_value = mock_predictor

        train_df = pd.DataFrame({"feature1": [1, 2, 3], "target": [True, False, True]})

        gluon_pred = GluonPredictor()
        gluon_pred.fit(train_df, "target", "/existing/path", "default")

        # Should remove existing directory
        mock_rmtree.assert_called_once_with("/existing/path")

    def test_gluon_predictor_problem_type_property(self):
        """Test problem_type property."""
        gluon_pred = GluonPredictor()

        # Mock predictor with quantile problem type
        mock_predictor = Mock()
        mock_predictor.problem_type = "quantile"
        gluon_pred.predictor = mock_predictor

        assert gluon_pred.problem_type == "regression"

        # Mock predictor with classification problem type
        mock_predictor.problem_type = "binary"
        assert gluon_pred.problem_type == "classification"

    def test_gluon_predictor_predict(self):
        """Test GluonPredictor predict method."""
        gluon_pred = GluonPredictor()

        # Mock predictor
        mock_predictor = Mock()
        mock_predictor.problem_type = "binary"
        mock_predictor.features.return_value = ["feature1", "feature2"]
        mock_predictor.predict.return_value = pd.Series([True, False])
        gluon_pred.predictor = mock_predictor

        test_df = pd.DataFrame(
            {
                "feature1": [1, 2],
                "feature2": [3, 4],
                "extra_feature": [5, 6],  # Should be filtered out
            }
        )

        result = gluon_pred.predict(test_df)

        # Check that only relevant features were used
        call_args = mock_predictor.predict.call_args[0][0]
        assert list(call_args.columns) == ["feature1", "feature2"]
        assert len(result) == 2

    def test_gluon_predictor_predict_quantile(self):
        """Test GluonPredictor predict method for quantile regression."""
        gluon_pred = GluonPredictor()

        # Mock predictor for quantile regression
        mock_predictor = Mock()
        mock_predictor.problem_type = "quantile"
        mock_predictor.features.return_value = ["feature1"]

        # Mock return value with quantile predictions
        mock_result = pd.DataFrame(
            {0.025: [1.0, 2.0], 0.5: [1.5, 2.5], 0.975: [2.0, 3.0]}
        )
        mock_predictor.predict.return_value = mock_result
        gluon_pred.predictor = mock_predictor

        test_df = pd.DataFrame({"feature1": [1, 2]})
        result = gluon_pred.predict(test_df)

        # Should return median (0.5 quantile)
        assert list(result) == [1.5, 2.5]

    def test_gluon_predictor_predict_proba(self):
        """Test GluonPredictor predict_proba method."""
        gluon_pred = GluonPredictor()

        # Mock predictor
        mock_predictor = Mock()
        mock_predictor.problem_type = "binary"
        mock_predictor.features.return_value = ["feature1"]
        mock_predictor.predict_proba.return_value = pd.DataFrame(
            {True: [0.8, 0.3], False: [0.2, 0.7]}
        )
        gluon_pred.predictor = mock_predictor

        test_df = pd.DataFrame({"feature1": [1, 2]})
        result = gluon_pred.predict_proba(test_df)

        assert result.shape == (2, 2)
        assert list(result.columns) == [True, False]

    def test_gluon_predictor_predict_proba_regression_error(self):
        """Test that predict_proba raises error for regression."""
        gluon_pred = GluonPredictor()

        mock_predictor = Mock()
        mock_predictor.problem_type = "quantile"
        gluon_pred.predictor = mock_predictor

        test_df = pd.DataFrame({"feature1": [1, 2]})

        with pytest.raises(RuntimeError, match="Probability predictions not available"):
            gluon_pred.predict_proba(test_df)

    def test_gluon_predictor_predict_interval(self):
        """Test GluonPredictor predict_interval method."""
        gluon_pred = GluonPredictor()

        # Mock predictor
        mock_predictor = Mock()
        mock_predictor.problem_type = "quantile"
        mock_predictor.features.return_value = ["feature1"]

        # Mock quantile predictions
        mock_result = pd.DataFrame(
            {0.025: [1.0, 2.0], 0.5: [1.5, 2.5], 0.975: [2.0, 3.0]}
        )
        mock_predictor.predict.return_value = mock_result
        gluon_pred.predictor = mock_predictor

        test_df = pd.DataFrame({"feature1": [1, 2]})
        result = gluon_pred.predict_interval(test_df)

        assert result.shape == (2, 2)
        expected_cols = ["2.5 %", "97.5 %"]
        assert list(result.columns) == expected_cols

        # Check that lower bounds are less than upper bounds
        assert all(result["2.5 %"] <= result["97.5 %"])

    def test_gluon_predictor_predict_interval_classification_error(self):
        """Test that predict_interval raises error for classification."""
        gluon_pred = GluonPredictor()

        mock_predictor = Mock()
        mock_predictor.problem_type = "binary"
        gluon_pred.predictor = mock_predictor

        test_df = pd.DataFrame({"feature1": [1, 2]})

        with pytest.raises(RuntimeError, match="Prediction intervals not available"):
            gluon_pred.predict_interval(test_df)

    @patch("autogluon.tabular.TabularPredictor.load")
    def test_gluon_predictor_load(self, mock_load):
        """Test GluonPredictor static load method."""
        mock_predictor = Mock()
        mock_load.return_value = mock_predictor

        result = GluonPredictor.load("/test/path")

        assert isinstance(result, GluonPredictor)
        assert result.predictor is mock_predictor
        mock_load.assert_called_once_with("/test/path")


class TestConstants:
    """Test module constants."""

    def test_conf_interval_bounds(self):
        """Test that confidence interval bounds are properly defined."""
        assert len(CONF_INTERVAL_BOUNDS) == 2
        assert CONF_INTERVAL_BOUNDS[0] < CONF_INTERVAL_BOUNDS[1]
        assert 0 < CONF_INTERVAL_BOUNDS[0] < 0.5
        assert 0.5 < CONF_INTERVAL_BOUNDS[1] < 1
