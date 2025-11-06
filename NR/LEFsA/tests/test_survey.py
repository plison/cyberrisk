import numpy as np
import pytest

from lefsa import survey


def test_create_org_features():
    """Create a basic OrgFeatures instance for testing, and checks
    the extraction of conditional assignments."""

    feats = survey.OrgFeatures(
        antall_ansatte="20 til 99 ansatte",
        bransje="Tjenesteytende næringer",
        sektor="Privat",
        samfunnskritiske_tjenester="Nei",
        rammeverk_for_informasjonssikkerhet="Ja",
        outsourcet="Delvis outsourcet",
        landsdel="Østlandet",
    )

    assignments = feats.get_cond_assignments()
    assert assignments == [
        ("Antall ansatte", "20 til 99 ansatte"),
        ("Bransje", "Tjenesteytende næringer"),
        ("Sektor", "Privat"),
        ("Samfunnskritiske tjenester", "Nei"),
        ("Rammeverk for informasjonssikkerhet", "Ja"),
        ("Outsourcet", "Delvis outsourcet"),
        ("Landsdel", "Østlandet"),
    ]


def test_load_survey_forecaster():
    """Loads the survey-based forecaster and checks that the
    covariate weights are correctly set up."""

    forecaster = survey.SurveyBasedForecaster()

    # Check that the forecaster was initialized properly
    assert forecaster.survey_data is not None
    assert hasattr(forecaster, "covariate_weights")
    assert hasattr(forecaster, "year_weights")
    assert len(forecaster.covariate_weights) > 0
    assert forecaster.bootstrap_samples == 10000
    assert forecaster.conf_level == 0.95


def test_basic_forecast():
    """Test basic forecasting functionality with a simple organization."""

    forecaster = survey.SurveyBasedForecaster()

    feats = survey.OrgFeatures(
        antall_ansatte="20 til 99 ansatte",
        bransje="Tjenesteytende næringer",
        sektor="Privat",
        samfunnskritiske_tjenester="Nei",
    )

    # Test basic forecast with confidence intervals
    forecast = forecaster.forecast(feats, return_type="ci")

    # Check that we get a proper forecast structure
    assert isinstance(forecast, dict)
    assert len(forecast) > 0

    # Check that all values are properly formatted (as percentages or tuples)
    for key, value in forecast.items():
        assert isinstance(value, (str, dict))
        if isinstance(value, str):
            assert "%" in value and "-" in value


def test_forecast_by_type():
    """Test forecasting by incident type."""

    forecaster = survey.SurveyBasedForecaster()

    feats = survey.OrgFeatures(
        antall_ansatte="100 ansatte eller flere",
        bransje="Offentlig administrasjon",
        sektor="Offentlig",
    )

    # Test forecast by type with short list
    by_type_forecast = forecaster.forecast_by_type(
        feats, use_short_list=True, return_type="ci"
    )

    assert isinstance(by_type_forecast, dict)
    assert len(by_type_forecast) > 0

    # Check that we have incident types from 2024 survey
    expected_types = [
        "Bedrageri",
        "Datainnbrudd",
        "Datatyveri",
        "Digitalt skadeverk",
        "Hacktivisme",
        "Tjenestenektangrep",
    ]
    for incident_type in expected_types:
        if incident_type in by_type_forecast:
            assert isinstance(by_type_forecast[incident_type], dict)


def test_predict_next_incident():
    """Test next incident prediction functionality."""

    forecaster = survey.SurveyBasedForecaster()

    feats = survey.OrgFeatures(
        antall_ansatte="20 til 99 ansatte",
        bransje="Helse og sosial",
        sektor="Offentlig",
        samfunnskritiske_tjenester="Ja",
    )

    prediction = forecaster.predict_next_incident(feats, return_type="ci")

    assert isinstance(prediction, dict)

    # Check for expected prediction categories
    expected_categories = [
        "time-to-event",
        "incident_types",
        "losses",
        "consequences",
        "factors",
        "discovery",
    ]
    for category in expected_categories:
        if category in prediction:
            if category == "time-to-event":
                assert "days" in str(prediction[category])
            else:
                assert isinstance(prediction[category], dict)


def test_different_return_types():
    """Test different return types for forecasting methods."""

    forecaster = survey.SurveyBasedForecaster()

    feats = survey.OrgFeatures(
        antall_ansatte="5 til 19 ansatte", bransje="Varehandel etc.", sektor="Privat"
    )

    # Test mean return type
    mean_forecast = forecaster.forecast(feats, return_type="mean")
    assert isinstance(mean_forecast, dict)
    for value in mean_forecast.values():
        if isinstance(value, str):
            assert "%" in value and "-" not in value

    # Test mean return type without percentages
    forecaster.use_percentages = False
    mean_forecast = forecaster.forecast(feats, return_type="mean")
    assert isinstance(mean_forecast, dict)
    for value in mean_forecast.values():
        if isinstance(value, str):
            assert "%" not in value and "-" not in value
    forecaster.use_percentages = True

    # Test confidence interval return type
    ci_forecast = forecaster.forecast(feats, return_type="ci")
    assert isinstance(ci_forecast, dict)
    for value in ci_forecast.values():
        if isinstance(value, str):
            assert "%" in value and "-" in value

    # Test samples return type
    samples_forecast = forecaster.forecast(feats, return_type="samples")
    assert isinstance(samples_forecast, dict)
    for value in samples_forecast.values():
        assert isinstance(value, np.ndarray)


def test_org_features_edge_cases():
    """Test OrgFeatures with minimal and maximal configurations."""

    # Test with minimal features
    minimal_feats = survey.OrgFeatures()
    assignments = minimal_feats.get_cond_assignments()
    assert assignments == []

    # Test with all features
    complete_feats = survey.OrgFeatures(
        antall_ansatte="100 ansatte eller flere",
        bransje="Offentlig administrasjon",
        sektor="Offentlig",
        samfunnskritiske_tjenester="Ja",
        rammeverk_for_informasjonssikkerhet="Ja",
        outsourcet="Helt outsourcet",
        landsdel="Oslo",
    )
    assignments = complete_feats.get_cond_assignments()
    assert len(assignments) == 7

    # Test that forecaster can handle both cases
    forecaster = survey.SurveyBasedForecaster()

    minimal_forecast = forecaster.forecast(minimal_feats)
    assert isinstance(minimal_forecast, dict)

    complete_forecast = forecaster.forecast(complete_feats)
    assert isinstance(complete_forecast, dict)


def test_covariate_weighting():
    """Test that covariate weighting is correctly applied in the forecaster."""

    forecaster = survey.SurveyBasedForecaster()

    # Check that covariate weights sum to 1
    total_weight = sum(forecaster.covariate_weights.values())
    assert np.isclose(total_weight, 1.0), (
        f"Total covariate weight should be 1, got {total_weight}"
    )

    # Check that each covariate has a non-negative weight
    for covariate, weight in forecaster.covariate_weights.items():
        assert weight >= 0, f"Covariate {covariate} has negative weight {weight}"

    # Check that antall_ansatte and bransje have the highest weights
    sorted_weights = sorted(
        forecaster.covariate_weights.items(), key=lambda x: x[1], reverse=True
    )
    top_covariates = [cov[0] for cov in sorted_weights[:2]]
    assert "Antall ansatte" in top_covariates
    assert "Bransje" in top_covariates


def test_different_organization_profiles():
    """Test forecasting for different types of organizations."""

    forecaster = survey.SurveyBasedForecaster()

    # Small private company
    small_private = survey.OrgFeatures(
        antall_ansatte="5 til 19 ansatte",
        bransje="Varehandel etc.",
        sektor="Privat",
        samfunnskritiske_tjenester="Nei",
        rammeverk_for_informasjonssikkerhet="Nei",
    )

    # Large public organization
    large_public = survey.OrgFeatures(
        antall_ansatte="100 ansatte eller flere",
        bransje="Offentlig administrasjon",
        sektor="Offentlig",
        samfunnskritiske_tjenester="Ja",
        rammeverk_for_informasjonssikkerhet="Ja",
    )

    # Healthcare organization
    healthcare = survey.OrgFeatures(
        antall_ansatte="100 ansatte eller flere",
        bransje="Helse og sosial",
        sektor="Offentlig",
        samfunnskritiske_tjenester="Ja",
    )

    profiles = [small_private, large_public, healthcare]

    for profile in profiles:
        forecast = forecaster.forecast(profile)
        assert isinstance(forecast, dict)
        assert len(forecast) > 0


def test_forecaster_time_units():
    """Test that the forecaster can handle different time units."""

    default_forecaster = survey.SurveyBasedForecaster()
    default_forecaster.use_percentages = False

    sample_proportions = default_forecaster._bootstrap_mixed_proportions(
        "incident_nb", ">= 1", []
    )

    default_forecaster.time_unit = "year"
    sample_rates_year = default_forecaster._get_rate(sample_proportions)
    avg_prob_atleast_1_year = np.mean(1 - np.exp(-sample_rates_year))

    yearly_forecast = default_forecaster.forecast(
        survey.OrgFeatures(), return_type="mean"
    )[">=1"]
    assert yearly_forecast == pytest.approx(avg_prob_atleast_1_year, rel=1e-3)

    default_forecaster.time_unit = "month"
    sample_rates_month = default_forecaster._get_rate(sample_proportions)
    assert np.mean(sample_rates_month) == pytest.approx(
        np.mean(sample_rates_year) / 12, rel=1e-3
    )

    avg_prob_atleast_1_month = np.mean(1 - np.exp(-sample_rates_month))
    monthly_forecast = default_forecaster.forecast(
        survey.OrgFeatures(), return_type="mean"
    )[">=1"]
    assert monthly_forecast == pytest.approx(avg_prob_atleast_1_month, rel=1e-2)

    assert (
        yearly_forecast > monthly_forecast * 12 - 0.01
    )  # Allow small numerical differences


def test_survey_data_access():
    """Test that survey data is properly accessible."""

    forecaster = survey.SurveyBasedForecaster()
    survey_data = forecaster.survey_data

    # Test basic survey data methods
    years = survey_data.get_years()
    assert isinstance(years, list)
    assert len(years) > 0

    categories = survey_data.get_categories()
    assert isinstance(categories, list)
    assert len(categories) > 0

    covariates = survey_data.get_covariates()
    assert isinstance(covariates, list)
    assert len(covariates) > 0

    # Test that we can get values for incident types
    incident_types = survey_data.get_values("incident_types")
    assert isinstance(incident_types, list)
    assert len(incident_types) > 0


def test_forecast_validation():
    """Test that forecasts produce reasonable values."""

    forecaster = survey.SurveyBasedForecaster()
    forecaster.use_percentages = False

    # High-risk organization
    high_risk = survey.OrgFeatures(
        antall_ansatte="100 ansatte eller flere",
        bransje="Offentlig administrasjon",
        sektor="Offentlig",
        samfunnskritiske_tjenester="Ja",
        rammeverk_for_informasjonssikkerhet="Nei",
    )

    # Low-risk organization
    low_risk = survey.OrgFeatures(
        antall_ansatte="5 til 19 ansatte",
        bransje="Kulturell virksomhet",
        sektor="Privat",
        samfunnskritiske_tjenester="Nei",
        rammeverk_for_informasjonssikkerhet="Ja",
    )

    high_risk_forecast = forecaster.forecast(high_risk, return_type="mean")[">=1"]
    low_risk_forecast = forecaster.forecast(low_risk, return_type="mean")[">=1"]

    assert high_risk_forecast > low_risk_forecast

    assert high_risk_forecast > low_risk_forecast
