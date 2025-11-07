# -*- coding: utf-8 -*-

from __future__ import annotations

import datetime
import itertools
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas

# Path to the JSON file containing the processed Mørketallsundersøkelse data
MORKETALL_JSON_FILE = os.path.join(os.path.dirname(__file__), "../data/morketall.json")

# Original excel files for the Mørketallsundersøkelse
EXCEL_FILES = {
    year: os.path.join(
        os.path.dirname(__file__),
        f"../mørketallsundersøkelser/Mørketallsundersøkelsen {year}.xlsx",
    )
    for year in [2016, 2018, 2020, 2022, 2024]
}

# mapping of questions to categories
QUESTION_MAPPING = {
    "Jeg vil nå lese opp noen mulige informasjonssikkerhetshendelser": "incident_types",
    "I forbindelse med datainnbruddet/datatyveriet, hva var deres tap?": "losses_breach_leakage",
    "Var det andre konsekvenser i forbindelse med datainnbruddet/datatyveriet?": "other_losses_breach_leakage",
    "I forbindelse med bedrageri, hva var deres tap?": "losses_fraud",
    "Hvilke metoder ble brukt av de som utførte bedrageriet?": "fraud_methods",
    "Førte denne spesifikke hendelsen til følgende?": "consequences",
    "Var noen av følgende faktorer medvirkende til at sikkerhetsbruddet oppsto": "factors",
    "Var noe av følgende årsak til at hendelsen ble oppdaget?": "discovery",
    "Hvor lang tid man det tok fra hendelsen skjedde til den ble oppdaget.": "discovery_time",
    "Ble hendelsen rapportert til noen av følgende?": "reporting",
    "Som et resultat av hendelsen, ble noen av følgende endringer": "subsequent_changes",
}


###############################################
# INPUT FEATURES TO FORECASTING MODEL
###############################################


@dataclass
class OrgFeatures:
    """Features extracted from the organisation's information, which can be used
    to forecast the number of incidents based on the survey"""

    # Number of employees of the organisation
    antall_ansatte: Optional[
        Literal["5 til 19 ansatte", "20 til 99 ansatte", "100 ansatte eller flere"]
    ] = None

    # Industry of the organisation
    bransje: Optional[
        Literal[
            "Industri etc.",
            "Bygg- og anleggsvirksomhet",
            "Varehandel etc.",
            "Transport og lagring",
            "Overnattings- og serveringsvirksomhet",
            "Tjenesteytende næringer",
            "Offentlig administrasjon",
            "Undervisning",
            "Helse og sosial",
            "Kulturell virksomhet",
        ]
    ] = None

    #
    sektor: Optional[Literal["Privat", "Offentlig"]] = None

    # Whether the organisation provides critical services to society
    samfunnskritiske_tjenester: Optional[Literal["Ja", "Nei"]] = None

    # Whether the organisation has a framework for information security
    rammeverk_for_informasjonssikkerhet: Optional[Literal["Ja", "Nei"]] = None

    # Whether the organisation outsources its information security
    outsourcet: Optional[
        Literal["Helt outsourcet", "Delvis outsourcet", "Organisert internt"]
    ] = None

    # The location of the organisation in Norway
    landsdel: Optional[
        Literal[
            "Nord-Norge",
            "Vestlandet",
            "Østlandet",
            "Sørlandet inkludert TeVe",
            "Midt-Norge",
            "Oslo",
        ]
    ] = None

    def get_cond_assignments(self) -> List[Tuple[str, str]]:
        """Returns a list of conditional assignments based on the organisation's features.
        Each assignment is a tuple of the form (feature_name, feature_value)."""

        conditional_assignments = []

        if self.antall_ansatte is not None:
            conditional_assignments.append(("Antall ansatte", self.antall_ansatte))
        if self.bransje is not None:
            conditional_assignments.append(("Bransje", self.bransje))
        if self.sektor is not None:
            conditional_assignments.append(("Sektor", self.sektor))
        if self.samfunnskritiske_tjenester is not None:
            conditional_assignments.append(
                ("Samfunnskritiske tjenester", self.samfunnskritiske_tjenester)
            )
        if self.rammeverk_for_informasjonssikkerhet is not None:
            conditional_assignments.append(
                (
                    "Rammeverk for informasjonssikkerhet",
                    self.rammeverk_for_informasjonssikkerhet,
                )
            )
        if self.outsourcet is not None:
            conditional_assignments.append(("Outsourcet", self.outsourcet))
        if self.landsdel is not None:
            conditional_assignments.append(("Landsdel", self.landsdel))

        return conditional_assignments

    @classmethod
    def from_description(cls, org_description: str) -> "OrgFeatures":
        """Creates an instance of OrgFeatures by querying a language model with the organisation's
        description in natural language.
        This method requires an OpenAI API key to be set in the environment variable OPENAI_KEY."""

        from . import llm

        return llm.create_survey_feats_from_description(org_description)


###############################################
# FORECASTING MODEL
###############################################


class SurveyBasedForecaster:
    """Forecaster based on the statistics from the Norwegian Mørketallsundersøkelse"""

    def __init__(self, survey_data: Union[SurveyData, str] = MORKETALL_JSON_FILE):
        """Initialises the forecaster based on the survey data"""

        print("Loading the survey-based forecaster", end="...", flush=True)

        # If the survey data is a string, we assume it is a path to a JSON file
        if isinstance(survey_data, str):
            survey_data = SurveyData.from_json(survey_data)

        # The data extracted from the Mørketallsundersøkelse
        self.survey_data = survey_data

        self.min_prob = (
            0.005  # Minimum proportion below which we do not report the probability
        )

        self.use_percentages = True  # Whether to use percentages or probabilities

        # "The time unit for the predictions, can be 'year', 'month' or 'day'"
        self.time_unit: Literal["year", "month", "day"] = "year"

        self.conf_level = 0.95  # Confidence level for the confidence intervals

        # Number of bootstrap samples for the confidence intervals
        self.bootstrap_samples = 10000

        # Whether to provide distinct probabilities for each number of incidents, or only give
        # probabilities for zero vs at least one incident
        self.split_probs_by_nb_events = False

        # Fit the temporal adjustments based on the survey data
        self.adjust_for_trends = False

        # Assuming a linear decay in the relevance of the predictions over the years
        linear_decay = 0.2
        self.year_weights = {
            "2016": max(0, 1 - 4 * linear_decay),
            "2018": max(0, 1 - 3 * linear_decay),
            "2020": max(0, 1 - 2 * linear_decay),
            "2022": max(0, 1 - 1 * linear_decay),
            "2024": 1.0,
        }

        # Setting the weights of each covariate by determining their explained variance
        self._set_covariate_weights()
        print("Done")

    def _set_covariate_weights(self):
        """Sets the weights of each covariate based on their explained variance in the survey data."""

        # We start by computing the variance of the proportions depending on each covariate.
        variances = {}
        for covariate in self.survey_data.get_covariates():
            # Skip the covariate "Opplevd hendelser" as it is not a feature known beforehand
            if covariate == "Opplevd hendelser":
                continue

            samples = []
            for year in self.survey_data.get_years():
                # We are mostly interested in the variances that impact the forecast of incidents
                for category in ["incident_nb", "incident_types"]:
                    for value in self.survey_data.get_values(category):
                        for covar_val in self.survey_data.get_covariate_values(
                            covariate
                        ):
                            if self.survey_data.has_condition(
                                category, value, year, covariate, covar_val
                            ):
                                sample = self._bootstrap_proportion(
                                    category, value, year, covariate, covar_val
                                )
                                samples.append(sample)
            full_sample = np.concatenate(samples)
            variances[covariate] = np.var(full_sample).round(5).item()
        #    print("Variances of covariates:", variances)

        # We also set a minimal variance for the case where no covariate is provided
        # (set to the minimal variance of the covariates)
        variances[None] = np.min(list(variances.values())).item()

        # The weight of each covariate is then proportional to its normalised variance.
        # (the idea being that a covariate that explains more variance in the data should have a higher weight)
        total_avg_variance = sum(variances.values())
        self.covariate_weights = {}
        for covariate, variance in variances.items():
            self.covariate_weights[covariate] = variance / total_avg_variance

    #        print("Covariance weight for", covariate, "is", self.covariate_weights[covariate])

    def forecast(
        self, feats: OrgFeatures, return_type: Literal["mean", "ci", "samples"] = "ci"
    ) -> Dict[str, Any]:
        """Forecast the number of incidents that a given organisation will experience in the time period.
        This forecast is limited to major incidents (that is, incidents that had a negative impact
        on the organisation).
        Args:
            feats (OrgFeatures): the features of the organisation
            return_type (str): the type of return value, can be 'mean', 'ci' or 'samples'
        """

        # We first extract the conditional assignments based on the organisation's features
        cond_assignments = feats.get_cond_assignments()

        sample_proportions = self._bootstrap_mixed_proportions(
            "incident_nb", ">= 1", cond_assignments
        )
        sample_poisson_probs = self._bootstrap_poisson_distrib(sample_proportions)

        if return_type == "mean":
            distrib = {k: np.mean(v) for k, v in sample_poisson_probs.items()}
        elif return_type == "ci":
            distrib = {
                k: self._get_confidence_interval(v)
                for k, v in sample_poisson_probs.items()
            }
        else:
            distrib = sample_poisson_probs

        return self._pretty_print(distrib)

    def forecast_by_type(
        self,
        feats: OrgFeatures,
        use_short_list=False,
        return_type: Literal["mean", "ci", "samples"] = "ci",
    ) -> Dict[str, Dict[str, Any]]:
        """Forecast the number of incidents of each type that a given organisation will experience in
        the time period. The incidents types are not mutually exclusive, and may also cover more minor
        incidents (e.g. phishing attemps, malware infections, etc.).
        Args:
            feats (OrgFeatures): the features of the organisation
            use_short_list (bool): whether to include all possible incident types, or only the shorter list of 6 types
              from the 2024 survey: Bedrageri, Datainnbrudd, Datatyveri, Digitalt skadeverk, Hacktivisme, Tjenestenektangrep.
            return_type (str): the type of return value, can be 'mean', 'ci' or 'samples'
        """

        # We first extract the conditional assignments based on the organisation's features
        cond_assignments = feats.get_cond_assignments()

        # Selecting the list of incident types to consider.
        if use_short_list:
            incident_types = self.survey_data.get_values("incident_types", ["2024"])
        else:
            incident_types = self.survey_data.get_values("incident_types")

        # Bootstrap samples for the proportions of each incident type
        by_type = {}
        for event_type in incident_types:
            sample_proportions = self._bootstrap_mixed_proportions(
                "incident_types", event_type, cond_assignments
            )
            sample_poisson_probs = self._bootstrap_poisson_distrib(sample_proportions)

            # We compute the mean, confidence interval, or return the samples themselves
            if return_type == "mean":
                by_type[event_type] = {
                    k: np.mean(v) for k, v in sample_poisson_probs.items()
                }
            elif return_type == "ci":
                by_type[event_type] = {
                    k: self._get_confidence_interval(v)
                    for k, v in sample_poisson_probs.items()
                }
            else:
                by_type[event_type] = sample_poisson_probs

        by_type = self._pretty_print(by_type)
        return by_type

    def predict_next_incident(
        self, feats: OrgFeatures, return_type: Literal["mean", "ci", "samples"] = "ci"
    ) -> Dict[str, Any]:
        # We first extract the conditional assignments based on the organisation's features
        cond_assignments = feats.get_cond_assignments()

        prediction = {}

        bootstrap_proportions = self._bootstrap_mixed_proportions(
            "incident_nb", ">= 1", cond_assignments
        )
        prediction["time-to-event"] = 1 / self._get_rate(bootstrap_proportions, "day")

        # We first draw bootstrap samples for the various incident types, given that a major incident occurs.
        bootstrap_incident_types = {}
        for value in self.survey_data.get_values("incident_types", ["2024"]):
            prob = self.survey_data.get_conditional_proportion(
                "incident_types",
                value,
                "2024",
                "Opplevd hendelser",
                "Opplevd en eller flere hendelser",
            )
            base_count = self.survey_data.get_counts(
                "incident_types",
                value,
                "2024",
                "Opplevd hendelser",
                "Opplevd en eller flere hendelser",
            )
            bootstrap_incident_types[value] = (
                np.random.binomial(base_count, prob, self.bootstrap_samples)
                / base_count
            )

        # We compute the confidence intervals for the probabilities of each incident type
        prediction["incident_types"] = {}
        for value, bootstrap_samples in bootstrap_incident_types.items():
            prediction["incident_types"][value] = bootstrap_samples

        # We draw samples for the probability of breach/leakage and fraud incidents
        bootstrap_prob_breach_leakage = (
            bootstrap_incident_types["Datainnbrudd"]
            + bootstrap_incident_types["Datatyveri"]
        )
        bootstrap_prob_fraud = bootstrap_incident_types["Bedrageri"]

        # We compute the CI of losses related to breaches, leakages and fraud
        # Full prob: P(loss) = P(loss |breach/leakage) * P(breach/leakage) + P(loss | fraud) * P(fraud)
        prediction["losses"] = {}
        for value in self.survey_data.get_values("losses_breach_leakage"):
            bootstrap_loss_breach_leakage = self._bootstrap_mixed_proportions(
                "losses_breach_leakage", value, cond_assignments
            )
            bootstrap_loss_fraud = self._bootstrap_mixed_proportions(
                "losses_fraud", value, cond_assignments
            )
            bootstrap_product = np.multiply(
                bootstrap_prob_breach_leakage, bootstrap_loss_breach_leakage
            ) + np.multiply(bootstrap_prob_fraud, bootstrap_loss_fraud)
            prediction["losses"][value] = bootstrap_product

        # We compute the CI of non-financial losses related to breaches and leakages
        prediction["other_losses"] = {}
        for value in self.survey_data.get_values("other_losses_breach_leakage"):
            bootstrap_other_loss_freq = self._bootstrap_mixed_proportions(
                "other_losses_breach_leakage", value, cond_assignments
            )
            bootstrap_product = np.multiply(
                bootstrap_prob_breach_leakage, bootstrap_other_loss_freq
            )
            prediction["other_losses"][value] = bootstrap_product

        # Finally, we compute the confidence intervals for the probabilities of consequences, factors and discovery methods
        for category in ["consequences", "factors", "discovery"]:
            prediction[category] = {}
            for value in self.survey_data.get_values(category):
                bootstrap_proportions = self._bootstrap_mixed_proportions(
                    category, value, cond_assignments
                )
                prediction[category][value] = bootstrap_proportions

        # We convert the predictions to the requested format
        for k, v in prediction.items():
            if k == "time-to-event":
                if return_type == "mean":
                    prediction[k] = f"{int(np.mean(v))} days"
                elif return_type == "ci":
                    low_bound, high_bound = self._get_confidence_interval(v)
                    prediction[k] = f"{int(low_bound)}-{int(high_bound)} days"
                continue

            for value, sample in v.items():
                if return_type == "mean":
                    prediction[k][value] = np.mean(sample)
                elif return_type == "ci":
                    prediction[k][value] = self._get_confidence_interval(sample)

        prediction = self._pretty_print(prediction)
        return prediction

    def _bootstrap_poisson_distrib(
        self, bootstrap_proportions, max_val=4
    ) -> Dict[str, np.ndarray]:
        """Based on a sample of proportions of organisations that experienced at least one incident,
        return a dictionary mapping each number of incidents to a list of sample probabilities"""

        # We derive the Poisson rates associated with the proportions
        rate = self._get_rate(bootstrap_proportions)

        # We compute the probabilities for the Poisson distribution.
        probs_samples = {}
        for i in range(max_val):
            probs_sample = (rate**i) * np.exp(-rate) / math.factorial(i)
            probs_samples[i] = probs_sample

        # We simplify the probabilities
        probs = {}
        for i, probs_sample in probs_samples.items():
            if i == 0:
                probs["None"] = probs_sample

            elif (
                self.split_probs_by_nb_events and np.mean(probs_sample) >= self.min_prob
            ):
                probs[f"{i}"] = probs_sample

            else:
                remaining_samples = [
                    probs_samples[j] for j in range(i, len(probs_samples))
                ]
                remainder = np.sum(remaining_samples, axis=0)
                if np.mean(remainder) >= self.min_prob / 10:
                    probs[f">={i}"] = remainder
                break

        return probs

    def _get_rate(
        self,
        proportion_of_orgs: Union[float, np.ndarray],
        time_unit: Optional[Literal["year", "month", "day"]] = None,
    ) -> Union[float, np.ndarray]:
        """Returns the rate of events per time unit, based on the proportion of organisations
        that experienced at least one event in the preceding year."""

        # We want to infer the Poisson rate λ expressing the expected number of events in the time period.
        # However, the survey data only provides the proportion p of organisations that experienced at least one
        # event in the preceding year. Fortunately, the rate λ can easily derived from the proportion p:
        # In a Poisson distribution, the probability of zero events is e^{−λ}. Therefore, the probability
        # of at least one event is 1−e^{−λ}.  If we know the proportion p, and solve 1−e^{−λ}=p for p,
        # we get λ=−ln(1−p).

        # If proportion_of_orgs is exactly 1, replace with 0.999 to avoid log(0)
        if isinstance(proportion_of_orgs, np.ndarray):
            proportion_of_orgs = np.where(
                proportion_of_orgs == 1, 0.999, proportion_of_orgs
            )
        elif proportion_of_orgs == 1:
            proportion_of_orgs = 0.999
        rate = -np.log(1 - proportion_of_orgs)

        # We set a small rate to avoid division by zero
        if isinstance(rate, np.ndarray):
            rate[rate == 0] = 1e-12
        elif rate == 0:
            rate = 1e-12

        time_unit = self.time_unit if time_unit is None else time_unit

        if time_unit == "year":
            return rate
        elif time_unit == "month":
            return rate / 12
        elif time_unit == "day":
            return rate / 365
        else:
            return rate

    def _get_confidence_interval(self, sample) -> Tuple[float, float]:
        """Returns the confidence interval for the given sample, based on the percentiles"""

        if not isinstance(sample, np.ndarray) and not isinstance(sample, list):
            raise TypeError("Sample must be a numpy array or a list.")

        conf_percentiles = [
            (1 - self.conf_level) / 2 * 100,
            (1 + self.conf_level) / 2 * 100,
        ]
        low_bound, high_bound = np.percentile(sample, conf_percentiles)

        return float(low_bound), float(high_bound)

    def _bootstrap_mixed_proportions(
        self, category: str, value: str, conditional_assignments: List[Tuple[str, str]]
    ) -> np.ndarray:
        """bootstrap a sample of proportions for the given category and value, based on a simple
        Mixture of Experts with a fixed gating function (the weights of the experts)
        """

        # STEP 1: get all relevant proportions for the given category and value
        relevant_proportions = []
        for year in self.survey_data.get_years():
            if self.survey_data.has_proportion(category, value, year):
                relevant_proportions.append((year, None, None))
            for cond_var, cond_value in conditional_assignments:
                if self.survey_data.has_condition(
                    category, value, year, cond_var, cond_value
                ):
                    relevant_proportions.append((year, cond_var, cond_value))

        # STEP 2: compute the weights for each proportion based on the year and covariate weight
        proportion_weights = {}
        for year, cond_var, _ in relevant_proportions:
            weight = self.year_weights[year] * self.covariate_weights[cond_var]
            proportion_weights[(year, cond_var)] = weight
        total_weight = sum(proportion_weights.values())

        if not relevant_proportions or total_weight == 0:
            return np.zeros(self.bootstrap_samples)

        # STEP 3: bootstrap samples for each relevant proportion
        samples = []
        for year, cond_var, cond_value in relevant_proportions:
            # The number of samples to bootstrap is proportional to the weight
            nb_to_bootstrap = int(
                proportion_weights[(year, cond_var)]
                * self.bootstrap_samples
                // total_weight
            )

            sample = self._bootstrap_proportion(
                category, value, year, cond_var, cond_value, nb_to_bootstrap
            )
            samples.append(sample)

        mixed_sample = np.concatenate(samples)

        # If we have not enough samples, we sample with replacement from the mixed sample
        if len(mixed_sample) < self.bootstrap_samples:
            remaining_nb = self.bootstrap_samples - len(mixed_sample)
            resampled = np.random.choice(mixed_sample, size=remaining_nb, replace=True)
            mixed_sample = np.concatenate([mixed_sample, resampled])

        return mixed_sample

    def _bootstrap_proportion(
        self,
        category: str,
        value: str,
        year: str,
        cond_var: Optional[str] = None,
        cond_value: Optional[str] = None,
        nb_to_bootstrap: Optional[int] = None,
    ) -> np.ndarray:
        """Bootstrap a sample of proportions for the given category and value, for the category, value
        ,year and condition (if provided). This is done by sampling from a binomial distribution.
        If adjusting for trends, we also apply a correction factor based on the elapsed time"""

        if nb_to_bootstrap is None:
            nb_to_bootstrap = self.bootstrap_samples

        base_count = self.survey_data.get_counts(
            category, value, year, cond_var, cond_value
        )
        if cond_var is None or cond_value is None:
            proportion = self.survey_data.get_proportion(category, value, year)
        else:
            proportion = self.survey_data.get_conditional_proportion(
                category, value, year, cond_var, cond_value
            )

        # We sample the proportion from a binomial distribution, where the number of trials is the base count
        # and the probability of success is the proportion
        bootstrap = (
            np.random.binomial(base_count, proportion, nb_to_bootstrap) / base_count
        )

        # If we are adjusting for trends, we bootstrap possible corrections (based on the elapsed time
        # since the survey was conducted), and add them to the bootstrap samples.
        if self.adjust_for_trends:
            correction_factor = self._bootstrap_correction_factor(
                category, value, nb_to_bootstrap, cond_var, cond_value
            )
            bootstrap += correction_factor * self._get_elapsed_days(year)

        bootstrap = np.clip(bootstrap, 1e-12, 0.9999)
        return bootstrap

    def _bootstrap_correction_factor(
        self,
        category: str,
        value: str,
        nb_to_bootstrap: int,
        cond_var: Optional[str] = None,
        cond_value: Optional[str] = None,
    ) -> np.ndarray:
        """To account for upwards or downwards trends in the number of incidents (or other
        security-related information) over the years, we adjust each prediction from a survey
        by a linear correction factor multiplied by the number of days elapsed since that survey.

        This correction factor is itself computed based on the differences between the proportions of
        consecutive surveys (for the given category, value, and condition is provided). The idea is
        that the mean of those differences indicates the slope of the trend from survey to survey.
        To account for uncertainty, we do not output a single factor, but rather an array of factors
        sampled via bootstrapping.

        Args:
            category (str): the category for which to compute the corrections (e.g. "incident_nb", "incident_types")
            value (str): the value for which to compute the corrections (e.g. ">= 1", "Datainnbrudd")
            nb_to_bootstrap (int): the number of bootstrap samples to draw
            cond_var (Optional[str]): the conditional variable for which to compute the corrections, if any
            cond_value (Optional[str]): the conditional value for which to compute the corrections, if any
        """

        # Retrieve the proportion for the given category and value (and possibly condition), for each year
        if cond_var and cond_value:
            prop_by_year = {
                int(year): self.survey_data.get_conditional_proportion(
                    category, value, year, cond_var, cond_value
                )
                for year in self.survey_data.get_years()
                if (
                    self.survey_data.has_condition(
                        category, value, year, cond_var, cond_value
                    )
                    and self.survey_data.has_proportion(category, value, year)
                )
            }
        else:
            prop_by_year = {
                int(year): self.survey_data.get_proportion(category, value, year)
                for year in self.survey_data.get_years()
                if self.survey_data.has_proportion(category, value, year)
            }

        # We compute the difference between two surveys divided by the number of days between them
        ndiffs = {}
        year_pairs = list(itertools.combinations(sorted(prop_by_year.keys()), 2))
        for year1, year2 in year_pairs:
            diff = prop_by_year[year2] - prop_by_year[year1]
            ndiffs[(year1, year2)] = diff / (year2 - year1) / 365

        if len(ndiffs) == 0:
            return np.zeros(nb_to_bootstrap)

        # We bootstrap differences by sampling with replacement
        bootstrap_diffs = np.random.choice(
            list(ndiffs.values()), size=(len(ndiffs), nb_to_bootstrap), replace=True
        )

        corrections_per_day = np.mean(bootstrap_diffs, axis=0)
        return corrections_per_day

    def _get_elapsed_days(
        self, survey_year: str, starting_day: Optional[datetime.datetime] = None
    ) -> int:
        """Returns the number of days elapsed since the survey was published."""
        if starting_day is None:
            starting_day = datetime.datetime.now()
        last_survey_day = datetime.datetime(int(survey_year), 12, 31)
        if starting_day <= last_survey_day:
            raise RuntimeError(
                f"Starting day {starting_day} is before the last survey day {last_survey_day}."
            )
        nb_elapsed_days = (starting_day - last_survey_day).days
        return nb_elapsed_days

    def _pretty_print(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Converts the results to a more human-readable format, rounding the values,
        and converting them to percentages if needed."""

        # For convenience, we drop the probability of zero events
        if "None" in results:
            del results["None"]

        # Using a natural order for the losses
        if "losses" in results:
            LOSS_ORDER = [
                "Ingen kostnader",
                "Under 50 000 NOK",
                "50 000 til 99 999 NOK",
                "100 000 til 499 999 NOK",
                "500 000 til 1 000 000 NOK",
                "Over 1 000 000 NOK",
            ]
            results["losses"] = {k: results["losses"][k] for k in LOSS_ORDER}

        if "other_losses" in results:
            OTHER_LOSSES_ORDER = [
                "Ingen",
                "Nedetid",
                "Omdømmetap",
                "Produksjonstopp",
                "Annet",
            ]
            results["other_losses"] = {
                k: results["other_losses"][k] for k in OTHER_LOSSES_ORDER
            }

        # For other types of predictions, we sort the keys such that the values with shortest names come first
        # (useful for plotting results)
        for k in [
            "incident_types",
            "consequences",
            "factors",
            "discovery",
            "reporting",
            "subsequent_changes",
        ]:
            if k in results:
                results[k] = {
                    key: results[k][key]
                    for key in sorted(results[k], key=lambda x: len(x))
                }

        for k, v in results.items():
            if isinstance(v, float):
                # we convert it to a percentage or round it to 4 decimal places
                if self.use_percentages:
                    results[k] = f"{round(float(v * 100), 1)}%"
                else:
                    results[k] = round(float(v), 4)

            elif isinstance(v, tuple):
                if self.use_percentages:
                    results[k] = (
                        f"{round(float(v[0] * 100), 1)}-{round(float(v[1] * 100), 1)}%"
                    )
                else:
                    results[k] = f"[{round(float(v[0]), 4)}, {round(float(v[1]), 4)}]"

            elif isinstance(v, dict):
                results[k] = self._pretty_print(v)

        return results


###############################################
# SURVEY DATA
###############################################


class SurveyData:
    """Class representing the survey data from the Mørketallsundersøkelse.
    It can be loaded from a JSON file or from the original Excel files."""

    def __init__(self):
        self._by_year = {}

    def get_years(self):
        """Returns the list of years for which the survey data is available."""
        return sorted(self._by_year.keys())

    def get_categories(self, years: Optional[List[str]] = None) -> List[str]:
        """Returns the sorted list of categories present in the survey data.
        If years is provided, only the categories for those years are returned."""

        if years is None:
            years = list(self._by_year.keys())
        return sorted(
            {category for year in years for category in self._by_year[year].keys()}
        )

    def get_values(self, category: str, years: Optional[List[str]] = None) -> List[str]:
        """Returns the sorted list of values for the given category across all years. If
        years is provided, only the values for those years are returned."""

        if years is None:
            years = list(self._by_year.keys())
        return sorted(
            {value for year in years for value in self._by_year[year].get(category, [])}
        )

    def has_proportion(self, category: str, value: str, year: Union[int, str]) -> bool:
        """Returns True if the given value for the given category in the given year has a proportion,
        otherwise returns False."""

        year = str(year) if isinstance(year, int) else year
        contains_val = value in self._by_year[year].get(category, {})

        # Nb: we also check whether the counts for the value are greater than 0,
        return (
            contains_val and self._by_year[year][category][value]["counts"]["total"] > 0
        )

    def has_condition(
        self,
        category: str,
        value: str,
        year: Union[int, str],
        cond_var: str,
        cond_value: str,
    ) -> bool:
        if not self.has_proportion(category, value, year):
            return False
        cat_freqs = self._by_year[year][category][value]
        condition = f"{cond_var}={cond_value}"
        return (
            condition in cat_freqs
            and self._by_year[year][category][value]["counts"][condition] > 0
        )

    def get_covariates(self) -> List[str]:
        """Returns a list of all covariates present in the survey data."""
        covariates = set()
        for year in self._by_year:
            for category in self._by_year[year]:
                for value in self._by_year[year][category]:
                    cat_freqs = self._by_year[year][category][value]
                    for condition in cat_freqs:
                        if condition != "default_prob" and condition != "counts":
                            cond_var, cond_value = condition.split("=")
                            covariates.add(cond_var)
        return sorted(covariates)

    def get_covariate_values(self, covariate: str) -> List[str]:
        """Returns a list of all values for a given covariate present in the survey data."""
        values = set()
        for year in self._by_year:
            for category in self._by_year[year]:
                for value in self._by_year[year][category]:
                    cat_freqs = self._by_year[year][category][value]
                    for condition in cat_freqs:
                        if condition != "default_prob" and condition != "counts":
                            cond_var, cond_value = condition.split("=")
                            if cond_var == covariate:
                                values.add(cond_value)
        return sorted(values)

    def get_proportion(self, category: str, value: str, year: Union[int, str]) -> float:
        """Returns the proportion of the given value for the given category in the given year."""

        year = str(year) if isinstance(year, int) else year

        if not self.has_proportion(category, value, year):
            raise KeyError(
                f"Category '{category}' and value '{value}' not found for year {year}."
            )

        default_prob = self._by_year[year][category][value]["default_prob"]

        # For 2024, we adjust (assuming an exponential model) the default probability to account
        # for the fact that the 2024 survey was conducted in september/october 2024, while the
        # respondents were asked about incidents in that same year (that is, before it was complete).
        if year == "2024":
            default_prob = 1 - (1 - default_prob) ** (12 / 10)
        return default_prob

    def get_conditional_proportion(
        self,
        category: str,
        value: str,
        year: Union[int, str],
        cond_var: str,
        cond_value: str,
    ) -> float:
        """Returns the conditional proportion of the given value for the given category in the given year,
        conditioned on the given covariate and its value. If the value is not present, returns None."""

        year = str(year) if isinstance(year, int) else year
        if not self.has_proportion(category, value, year):
            raise KeyError(
                f"Category '{category}' and value '{value}' not found for year {year}."
            )
        condition = f"{cond_var}={cond_value}"
        if not self.has_condition(category, value, year, cond_var, cond_value):
            raise KeyError(
                f"Condition '{condition}' not found for value '{value}' in category '{category}' for year {year}."
            )
        prob = self._by_year[year][category][value][condition]

        # For 2024, we adjust (assuming an exponential model) the default probability to account
        # for the fact that the 2024 survey was conducted in september/october 2024, while the
        # respondents were asked about incidents in that same year (that is, before it was complete).
        if year == "2024":
            prob = 1 - (1 - prob) ** (12 / 10)
        return prob

    def get_counts(
        self,
        category: str,
        value: str,
        year: Union[int, str],
        cond_var: Optional[str] = None,
        cond_value: Optional[str] = None,
    ) -> int:
        """Returns the counts of organisations from which the proportion (categy=value) was computed.
        If cond_var and cond_value are provided, returns the counts for that condition."""

        year = str(year) if isinstance(year, int) else year
        if not self.has_proportion(category, value, year):
            raise KeyError(
                f"Proportion for {category}={value} in year {year} not found."
            )
        counts = self._by_year[year][category][value]["counts"]
        if cond_var is None or cond_value is None:
            return counts["total"]

        condition = f"{cond_var}={cond_value}"
        if not self.has_condition(category, value, year, cond_var, cond_value):
            raise KeyError(
                f"Condition '{condition}' not found for value '{value}' in category '{category}' for year {year}."
            )
        return counts[condition]

    @classmethod
    def from_json(cls, json_file=MORKETALL_JSON_FILE):
        """Loads the survey data from the provided JSON file."""
        data = cls()
        with open(json_file, "r") as fd:
            data._by_year = json.load(fd)
        return data

    @classmethod
    def from_excel(cls, files_by_year=EXCEL_FILES):
        """Loads the survey data from the Excel files (can take a minute)"""

        data = cls()
        for year, filepath in files_by_year.items():
            year = str(year)
            try:
                print("Working on year", year, end="... ", flush=True)
                data._by_year[year] = {}

                # STEP 1: read the excel file with pandas and segment it into tables,
                # each comprising a number of rows
                tables_rows = segment_tables(filepath)
                print("Number of extracted tables:", len(tables_rows))

                for table_rows in tables_rows:
                    # STEP 2: read the content of each table and store them in a dictionary
                    table = read_table(table_rows)

                    category, value = get_category_and_value(table)
                    if category is None:
                        continue
                    elif category not in data._by_year[year]:
                        data._by_year[year][category] = {}

                    # STEP 3: process the table, and store the proportions in the data structure
                    # We have two cases: either the table contains a single value (e.g. "Ja" or "Nei"),
                    # or it contains multiple values (e.g. possible sectors)
                    if value:
                        data._by_year[year][category][value] = (
                            process_single_value_table(table)
                        )
                    else:
                        for value, probs_for_value in process_multi_value_table(
                            table
                        ).items():
                            data._by_year[year][category][value] = probs_for_value

            except FileNotFoundError:
                print(f"File for year {year} not found. Trying next year...")
        return data

    def to_json(self, output_file=MORKETALL_JSON_FILE):
        """Saves the survey data to a JSON file."""

        with open(output_file, "w") as fd:
            json.dump(self._by_year, fd, indent=4)


###############################################
# PROCESSING OF EXCEL FILES
###############################################


def segment_tables(excel_file: str) -> List[List[pandas.Series]]:
    """Take an excel file from the Mørketallsundersøkelse, read it with pandas,
    and segment the content in a list of tables (each spanning a set of rows)"""

    df = pandas.read_excel(excel_file)
    df = df[4:]  # We drop the first 4 rows, as they contain metadata

    # We drop the first column if it is empty
    if len(df[df.columns[0]].unique()) == 1:
        df = df.drop(columns=[df.columns[0]])

    # We use numeric indices for the columns
    df.columns = [i for i in range(len(df.columns))]

    current_table_rows = []
    tables = []
    for _, row in df[4:].iterrows():
        if not any(isinstance(cell, str) for cell in row):
            continue

        # A table is finished when "Total" or "Weight: vekt_justert" is encountered
        elif row[0] in ["TOTAL", "Weight: vekt_justert"]:
            if current_table_rows:
                tables.append(current_table_rows)
                current_table_rows = []
        else:
            current_table_rows.append(row)
    return tables


def get_covariates(table_rows):
    """Extracts the covariates from the first two rows of the table. The first row
    contains the category names, and the second row contains the values for each category.
    """

    values_by_category = {}
    for row in table_rows:
        # First line: category names
        if len(values_by_category) == 0 and isinstance(row[2], str):
            for val in row.dropna().values:
                values_by_category[val.strip()] = []
            prev_row = row

        # Second line: values for each category
        elif len(values_by_category) > 0 and isinstance(row[2], str):
            current_cat = None
            for i, val in enumerate(row[2:].values):
                cell_above = prev_row[i + 2]
                if isinstance(cell_above, str):
                    current_cat = cell_above.strip()
                values_by_category[current_cat].append(val.strip())
            break

    # We normalize the category names to make them more consistent across years
    covariates = {}
    for cat, vals in values_by_category.items():
        if "outsourc" in cat.lower():
            cat = "Outsourcet"
        elif (
            "rammeverk" in cat.lower()
            or "system for informasjonssikkerhet" in cat.lower()
        ):
            cat = "Rammeverk for informasjonssikkerhet"
        elif "ansatte" in cat.lower():
            cat = "Antall ansatte"
        elif "landsdel" in cat.lower():
            cat = "Landsdel"

        vals2 = []
        for val in vals:
            if val == "5-19 ansatte":
                val = "5 til 19 ansatte"
            elif val == "20-99 ansatte":
                val = "20 til 99 ansatte"
            vals2.append(val.strip())

        covariates[cat] = vals2
    return covariates


def read_table(table_rows):
    """Reads the content of a table from the Mørketallsundersøkelse, and returns a dictionary
    with the following keys:
    - "question": the question index (e.g. "Opplevd hendelser")
    - "question 2": the second question index (if present)
    - "question 3": the third question index (if present)
    - "count(total)": the total number of organisations in the sample
    - "count(<value>)": the number of organisations with the given value for the question
    - "P(<value>)": the probability of the given value for the question
    - "P(<value>|<covariate>=<covariate_value>)": the conditional probability of the given value for the question,
      conditioned on the given covariate and its value
    """

    table = {}
    values = get_covariates(table_rows)
    values_flat = [(k, v) for k, vs in values.items() for v in vs]

    for i, row in enumerate(table_rows):
        if len(table) == 0 and isinstance(row[0], str):
            q_index = row[0].strip()
            table = {"question": q_index}

        elif (
            len(table) == 1
            and isinstance(row[0], str)
            and isinstance(row[1], float)
            and np.isnan(row[1])
        ):
            table["question 2"] = row[0].strip()

        elif (
            len(table) == 2
            and isinstance(row[0], str)
            and isinstance(row[1], float)
            and np.isnan(row[1])
        ):
            table["question 3"] = row[0].strip()

        elif row[0] in ["BASE", "Base antall uvektet"]:
            table["count(total)"] = int(row[1])
            for i in range(len(values_flat)):
                table["count(%s=%s)" % (values_flat[i][0], values_flat[i][1])] = int(
                    row[i + 2]
                )

        elif "count(total)" in table:
            value = row[0].strip()
            total_prob = row[1] if not np.isnan(row[1]) else 0.0
            table["P(%s)" % value] = total_prob
            for i in range(len(values_flat)):
                cond_var, cond_value = values_flat[i]
                if (
                    "stilling" in cond_var.lower()
                    or "vet ikke" in cond_value.lower()
                    or "ikke aktuelt" in cond_value.lower()
                ):
                    continue
                prob = row[i + 2] if not np.isnan(row[i + 2]) else 0.0
                assignment = "P(%s|%s=%s)" % (value, cond_var, cond_value)
                table[assignment] = prob

    return table


def process_single_value_table(table):
    """Processes a table with a single value (e.g. "Ja" or "Nei") and returns the proportions
    of the positive value, both unconditional and conditional on the covariates."""

    # We first determine the name of the row that contains the positive value.
    if table["question"] == "Opplevd hendelser":
        row_name = "Opplevd en eller flere hendelser"

    elif "P(Noter antall)" in table:
        row_name = "Noter antall"
    else:
        row_name = "Ja"

    # We first extract the unconditional probability of the positive value,
    # and the total number of organisations used to compute it
    prob = round(table["P(%s)" % row_name], 5)

    # If we have a "Vet ikke" row, we need to adjust the probability (assuming the "Vet ikke"
    # organisations have the same distribution as the rest of the organisations)
    if "P(Vet ikke)" in table:
        prob = prob / (1 - table["P(Vet ikke)"])
    if "P(Vet ikke/ønsker ikke å oppgi)" in table:
        prob = prob / (1 - table["P(Vet ikke/ønsker ikke å oppgi)"])

    proportions = {"default_prob": prob, "counts": {"total": table["count(total)"]}}

    # We then extract the conditional probabilities for each covariate
    for key, prob in table.items():
        if key.startswith(f"P({row_name}|"):
            condition = key.replace(f"P({row_name}|", "").replace(")", "")

            if f"P(Vet ikke|{condition})" in table:
                prob = prob / (1 - table[f"P(Vet ikke|{condition})"])
            if f"P(Vet ikke/ønsker ikke å oppgi|{condition})" in table:
                prob = prob / (
                    1 - table[f"P(Vet ikke/ønsker ikke å oppgi|{condition})"]
                )

            proportions[condition] = round(prob, 5)
            proportions["counts"][condition] = table["count(%s)" % condition]

    return proportions


def process_multi_value_table(table):
    """Processes a table with multiple values (e.g. possible sectors) and returns the proportions
    of each value, both unconditional and conditional on the covariates."""

    denominator = 1.0
    if "P(Vet ikke)" in table:
        denominator = 1 - table["P(Vet ikke)"]

    proportions = {}
    # We first extract the unconditional probabilities for each value, and the total nb of organisations
    for key, prob in table.items():
        if "Vet ikke" in key:
            continue
        if key.startswith("P(") and "|" not in key:
            prob = prob / denominator
            value = normalise_value(key[2:-1])
            proportions[value] = {
                "default_prob": round(prob, 5),
                "counts": {"total": table["count(total)"]},
            }

    # We then extract the conditional probabilities for each value, conditioned on the covariates
    for key, prob in table.items():
        if "Vet ikke" in key:
            continue
        elif key.startswith("P(") and "|" in key:
            denominator = 1.0
            value, condition = key[2:-1].split("|")
            value = normalise_value(value.strip())
            if f"P(Vet ikke|{condition})" in table:
                denominator = 1 - table[f"P(Vet ikke|{condition})"]
                prob = prob / denominator
            proportions[value][condition] = round(prob, 5)
            proportions[value]["counts"][condition] = table["count(%s)" % condition]

    return proportions


def get_category_and_value(table) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts the category, and value from the table. For instance, for the question
    "Phishing (e-post, SMS, telefon) - Hvilke metoder ble brukt av de som utførte bedrageriet?",
    it returns ("incident_subtype", "Phishing (e-post, SMS, telefon)").
    """

    category = None
    value = None

    if table["question"] == "Opplevd hendelser":
        category = "incident_nb"
        value = ">= 1"
        return category, value

    elif table.get("question 2", "").startswith(
        "Hvor mange informasjonssikkerhetshendelser som påvirket organisasjonen negativt"
    ):
        if "P(Noter antall)" in table:
            category = "incident_nb"
            value = ">= 1"
            return category, value

    for question, question_cat in QUESTION_MAPPING.items():
        # We need to account for various formatting options (in some files, the question
        # and its answer are on the same line, in others they are split)"""
        if question in table["question"]:
            category = question_cat
            if "-" in table["question"]:
                value = table["question"].split("-")[0].split("(")[0].strip()
            elif "question 2" in table:
                value = table["question 2"].split("(")[0].strip()
            break

        elif question in table.get("question 2", ""):
            category = question_cat
            if "-" in table["question 2"]:
                value = table["question 2"].split("-")[0].split("(")[0].strip()
            elif "question 3" in table:
                value = table["question 3"].split("(")[0].strip()
            break

    if value:
        value = normalise_value(value, category)

    return category, value


def normalise_value(value: str, category: Optional[str] = None) -> Optional[str]:
    """Normalise the values for the given category, so that they are consistent across years"""

    if category == "incident_types":
        if "Misbruk" in value and "IT" in value:
            return "Misbruk av IT-systemer"
        elif "Dataskadeverk" in value:
            return "Digitalt skadeverk"
        elif "Tyveri" in value and "data" in value:
            return "Datatyveri"
        elif "Tyveri" in value and "IT" in value:
            return "Tyveri av IT-utstyr"
        elif "opplysninger" in value:
            return "Tap av opplysninger"
        elif "Datainnbrudd" in value:
            return "Datainnbrudd"
        elif "DDoS" in value:
            return "Tjenestenektangrep"

    elif "Varsel fra offentlige myndigheter" in value:
        return "Varsel fra offentlige myndigheter (f.eks. politi)"
    elif "Rutinemessig intern" in value:
        return "Rutinemessig intern sikkerhetsmonitorering"
    elif "media tilkoplet interne ressurser" in value:
        return "Flyttbar media tilkoplet interne ressurser"

    return value.strip()
