# -*- coding: utf-8 -*-

import json
import os
import re
import string
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import currency_converter
import numpy as np
import pandas
import pycountry_convert as pyco
from tqdm import tqdm

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
VCDB_DIR = os.path.join(CUR_DIR, "..", "..", "VCDB", "data", "json", "validated")
VCDB_JSON = os.path.join(CUR_DIR, "..", "data", "vcdb.json")


EMPLOYEE_COUNT_ORDINAL = [
    "1 to 10",
    "11 to 100",
    "101 to 1000",
    "1001 to 10000",
    "10001 to 25000",
    "25001 to 50000",
    "50001 to 100000",
    "Over 100000",
]
RATING_ORDINAL = ["Minor", "Moderate", "Major"]
OVERALL_RATING_ORDINAL = ["Insignificant", "Distracting", "Damaging", "Painful"]


@dataclass
class InputFeatures:
    """Dataclass representing features of a victim organisation."""

    incident_date: Optional[pandas.Timestamp] = pandas.Timestamp.now()

    country: Optional[str] = None

    continent: Optional[
        Literal[
            "Asia",
            "Europe",
            "North America",
            "South America",
            "Africa",
            "Oceania",
            "Antarctica",
        ]
    ] = None

    employee_count: Optional[
        Literal[
            "1 to 10",
            "11 to 100",
            "101 to 1000",
            "1001 to 10000",
            "10001 to 25000",
            "25001 to 50000",
            "50001 to 100000",
            "Over 100000",
        ]
    ] = None

    employee_count_simplified: Optional[Literal["Small", "Large"]] = None

    government: Optional[Literal["NA", "local", "regional", "federal"]] = None

    industry: Optional[str] = None

    revenue: Optional[int] = None

    def __post_init__(self):
        if self.incident_date is not None:
            if not isinstance(self.incident_date, pandas.Timestamp):
                raise ValueError("incident_date must be a pandas Timestamp")
        if self.country is not None:
            self.country = self.country.upper()
            pyco.country_alpha2_to_country_name(
                self.country
            )  # raises KeyError if invalid
        if self.continent is not None:
            self.continent = self.continent.title()  # type:ignore
            if self.continent not in [
                "Asia",
                "Europe",
                "North America",
                "South America",
                "Africa",
                "Oceania",
                "Antarctica",
            ]:
                raise ValueError(f"Invalid continent: {self.continent}")
        elif self.country is not None:
            continent_code = pyco.country_alpha2_to_continent_code(self.country)
            self.continent = pyco.convert_continent_code_to_continent_name(
                continent_code
            )  # type:ignore
        if self.employee_count is not None:
            if self.employee_count not in EMPLOYEE_COUNT_ORDINAL:
                raise ValueError(f"Invalid employee_count: {self.employee_count}")
        if self.employee_count_simplified is not None:
            if self.employee_count_simplified not in ["Small", "Large"]:
                raise ValueError(
                    f"Invalid employee_count_simplified: {self.employee_count_simplified}"
                )
        elif self.employee_count is not None:
            if self.employee_count in ["1 to 10", "11 to 100", "101 to 1000"]:
                self.employee_count_simplified = "Small"
            else:
                self.employee_count_simplified = "Large"
        if self.government is not None:
            valid_government_types = ["NA", "local", "regional", "federal"]
            if self.government not in valid_government_types:
                raise ValueError(f"Invalid government type: {self.government}")
        if self.industry is not None:
            if not re.match(r"^\d{2,6}$", self.industry):
                raise ValueError(
                    f"Industry must be a NAICS code (2 to 6 digits), got: {self.industry}"
                )
        if self.revenue is not None:
            if not isinstance(self.revenue, int) or self.revenue < 0:
                raise ValueError(
                    f"Revenue must be a non-negative integer, got: {self.revenue}"
                )

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "timeline.incident.date": self.incident_date,
            "victim.country": self.country,
            "victim.continent": self.continent,
            "victim.employee_count": self.employee_count,
            "victim.employee_count_simplified": self.employee_count_simplified,
            "victim.government": self.government,
            "victim.revenue.amount": self.revenue,
        }
        if self.industry is not None:
            for i in range(2, len(self.industry) + 1):
                d[f"victim.industry_naics_{i}"] = self.industry[:i]
        return d

    def to_pandas_series(self) -> pandas.Series:
        """Converts the InputFeatures to a pandas Series."""
        return pandas.Series(self.to_dict())

    @classmethod
    def from_description(cls, org_description: str) -> "InputFeatures":
        """Creates an instance of VictimFeatures by querying a language model with the organisation's
        description in natural language.
        This method requires an OpenAI API key to be set in the environment variable OPENAI_KEY."""

        from . import llm

        return llm.create_victim_feats_from_description(org_description)

    @classmethod
    def from_pandas_series(cls, series: pandas.Series) -> "InputFeatures":  # type: ignore
        """Creates an instance of VictimFeatures from a pandas Series."""

        s = series.dropna()
        feats = cls(
            incident_date=s.get("timeline.incident.date", pandas.Timestamp.now()),
            country=s.get("victim.country", None),
            continent=s.get("victim.continent", None),
            employee_count=s.get("victim.employee_count", None),
            employee_count_simplified=s.get("victim.employee_count_simplified", None),
            government=s.get("victim.government", None),
            industry=s.get("victim.industry", None),
            revenue=s.get("victim.revenue.amount", None),
        )
        for i in range(6, 1, -1):
            if f"victim.industry_naics_{i}" in s:
                feats.industry = s[f"victim.industry_naics_{i}"]
                break
        return feats


class VCDB:
    """Represents the VCDB dataset, with methods to load and save the data."""

    def __init__(self, json_path: Optional[str] = VCDB_JSON):
        """Initializes a database of VCDB incidents, either empty or loading
        from the given JSON file."""

        self.raw_incidents = []
        self.incidents = []
        if json_path is not None:
            with open(json_path) as fd:
                data = json.load(fd)
                self.raw_incidents = data["raw_incidents"]
                self._normalise()

    @classmethod
    def load_from_dir(cls, dir: str = VCDB_DIR):
        """Loads the incidents from the given directory (containing one JSON file per
        incident as in https://github.com/vz-risk/VCDB) and normalizes them.
        """

        # Load raw incidents
        obj = cls(json_path=None)
        for f in tqdm(os.listdir(dir)):
            path = os.path.join(dir, f)
            with open(path) as fd:
                try:
                    incident = json.load(fd)
                    obj.raw_incidents.append(incident)
                except json.JSONDecodeError as e:
                    print("JSON error in", path, ":", e)
                    pass

        obj._normalise()
        return obj

    def _normalise(self):
        """Normalises the raw incidents loaded from VCDB."""

        self.incidents = []
        processor = IncidentProcessor()
        for raw_incident in self.raw_incidents:
            norm_incident = flatten_data(raw_incident)
            norm_incident = processor.enrich_incident(norm_incident)
            self.incidents.append(norm_incident)

        # For boolean columns, fill missing values with False
        boolean_cols = set()
        for incident in self.incidents:
            for k, v in incident.items():
                if isinstance(v, bool):
                    boolean_cols.add(k)
        for bool_col in boolean_cols:
            for incident in self.incidents:
                if bool_col not in incident:
                    incident[bool_col] = False

    def split_train_test(
        self, test_ratio: float = 0.2, min_nb_incidents_for_col: int = 10
    ) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
        """Splits the normalized incidents into a training and testing DataFrame.
        The test set contains the test_ratio*100 % most recent incidents.
        Only columns that are present in at least min_nb_incidents_for_col incidents
        in the training set, and at least once in the testing set, are kept."""

        # Split in train/test (taking as test set the test_ratio*100 % most recent incidents)
        train_incidents = []
        test_incidents = []
        for i in range(len(self.incidents)):
            if i < len(self.incidents) * test_ratio:
                test_incidents.append(self.incidents[i])
            else:
                train_incidents.append(self.incidents[i])

        # Only keep columns that are present in at least min_nb_incidents_for_col
        # incidents in the training set, and at least once in the testing set
        all_train_columns_counts = {}
        for incident in train_incidents:
            for k in incident.keys():
                all_train_columns_counts[k] = all_train_columns_counts.get(k, 0) + 1
        all_test_columns = {k for incident in test_incidents for k in incident.keys()}
        cols_to_keep = {
            k
            for k, v in all_train_columns_counts.items()
            if v >= min_nb_incidents_for_col and k in all_test_columns
        }
        cols_to_keep = sorted(cols_to_keep)

        train_df = pandas.DataFrame(train_incidents)[cols_to_keep].set_index(
            "incident_id"
        )
        test_df = pandas.DataFrame(test_incidents)[cols_to_keep].set_index(
            "incident_id"
        )

        _, input_cols, target_cols = classify_columns(train_df)

        # For the input and target cols, we transform str and objects to
        # categorical types (with an ordering for ordinal types),
        # and numeric types to float64
        for s in input_cols + target_cols:
            if train_df[s].dtype in ["str", "object"]:
                # For ordinal types, we use a predefined ordering
                if s.endswith(".rating"):  # type: ignore
                    ordered_cats = RATING_ORDINAL
                elif s.endswith(".overall_rating"):  # type: ignore
                    ordered_cats = OVERALL_RATING_ORDINAL
                elif s.endswith(".employee_count"):  # type: ignore
                    ordered_cats = EMPLOYEE_COUNT_ORDINAL

                # For other categories, we use an alphabetical sort
                else:
                    ordered_cats = sorted(
                        set(train_df[s].dropna().unique()).union(
                            set(test_df[s].dropna().unique())
                        )
                    )
                train_df[s] = pandas.Categorical(
                    train_df[s], categories=ordered_cats, ordered=True
                )
                test_df[s] = pandas.Categorical(
                    test_df[s], categories=ordered_cats, ordered=True
                )

            # Numeric types are all converted to float64
            elif train_df[s].dtype in [
                "float",
                "float32",
                "float64",
                "int",
                "int32",
                "int64",
            ]:
                train_df[s] = train_df[s].astype(np.float64)
                test_df[s] = test_df[s].astype(np.float64)

        return train_df, test_df

    def save_to_json(self, path: str = VCDB_JSON):
        """Saves the normalized incidents to the given JSON file."""

        with open(path, "w") as fd:
            json.dump({"raw_incidents": self.raw_incidents}, fd, indent=2)

    @classmethod
    def load_from_json(cls, path: str = VCDB_JSON):
        """Loads the raw incidents from the given JSON file, and
        normalizes them."""

        return cls(path)


class IncidentProcessor:
    """Processes and normalizes individual incidents from VCDB."""

    def __init__(self):
        """Initializes the IncidentProcessor.

        Argument:
        - naics_code_length: length to which NAICS codes should be truncated
        (e.g. 3 for sector-level codes)"""

        self.converter = currency_converter.CurrencyConverter()

    def enrich_incident(self, incident):
        """Enriches and normalizes the given incident description, by:
        - mapping countries to world regions
        - adding an incident date feature
        - converting timeline durations to pandas timedeltas
        - replacing 'Unknown' values with NaN
        - converting all monetary amounts to USD
        - adding aggregated features (e.g. "action.error" without details on variety)
        - shortening NAICS codes to the specified length
        - simplifying victim employee counts into 'Small' and 'Large'
        """

        # map countries to world regions
        for k, v in list(incident.items()):
            if ".country" in k:
                country_code = v
                try:
                    continent_code = pyco.country_alpha2_to_continent_code(
                        country_code.upper()
                    )
                    continent_name = pyco.convert_continent_code_to_continent_name(
                        continent_code
                    )
                    incident[k.replace(".country", ".continent")] = continent_name
                except Exception:
                    #             print("Error:", e)
                    pass

        # Add a incident date feature (as a pandas timestamp)
        if "timeline.incident.year" in incident:
            day = incident.get("timeline.incident.day", 1)
            month = incident.get("timeline.incident.month", 1)
            year = incident["timeline.incident.year"]
            try:
                incident_day = pandas.Timestamp(year=year, month=month, day=day)
            except ValueError:
                incident_day = pandas.Timestamp(year=year, month=month, day=1)
            incident["timeline.incident.date"] = incident_day

        # Convert all timeline durations to pandas timedelta
        for x in [
            "timeline.compromise",
            "timeline.discovery",
            "timeline.containment",
            "timeline.exfiltration",
            "timeline.attribute.availability.duration",
        ]:
            if "%s.unit" % x in incident and "%s.value" % x in incident:
                unit = incident["%s.unit" % x].lower()
                value = incident["%s.value" % x]
                if unit in ["seconds", "minutes", "hours", "days"]:
                    delta = pandas.to_timedelta(str(value) + " " + unit)
                elif unit == "weeks":
                    delta = pandas.to_timedelta(value * 7, unit="days")
                elif unit == "months":
                    delta = pandas.to_timedelta(value * 30, unit="days")
                elif unit == "years":
                    delta = pandas.to_timedelta(value * 365, unit="days")
                else:
                    continue
                incident["timeline.%s" % x] = delta

        # Replace 'Unknown' values with NaN
        for k, v in list(incident.items()):
            if isinstance(v, str) and v.lower() == "unknown":
                incident[k] = np.nan

        # Convert all monetary amounts to USD
        for k, v in list(incident.items()):
            if (
                k.endswith("amount")
                and "iso_currency_code" in k
                and k.replace("amount", "iso_currency_code") in incident
                and incident[k.replace("amount", "iso_currency_code")] != "USD"
            ):
                try:
                    currency_code = incident[k.replace("amount", "iso_currency_code")]
                    new_amount = self.converter.convert(v, currency_code, "USD")
                    incident[k] = new_amount
                except Exception as e:
                    print("Error converting currency for", k, v, currency_code, ":", e)
                    pass

        # We also include aggregated features, such as "action.error"
        # (without details on the variety)
        MOVE_ON_SUFFIXES = [
            "variety",
            "result",
            "vector",
            "target",
            "motive",
            "assets",
            "confidentiality.state",
            "attribute.unknown",
            "victim.revenue",
        ]
        for k in list(incident):
            for i in range(2, k.count(".") + 1):
                coarse_cat = ".".join(k.split(".")[:i])
                if (
                    not any(coarse_cat.endswith(suffix) for suffix in MOVE_ON_SUFFIXES)
                    and coarse_cat not in incident
                ):
                    incident[coarse_cat] = True

        # We extract NAICS codes of different lengths and create separate features for them
        if "victim.industry" in incident and isinstance(
            incident["victim.industry"], str
        ):
            for i in range(2, len(incident["victim.industry"]) + 1):
                prefix = incident["victim.industry"][:i]
                incident["victim.industry_naics_" + str(i)] = prefix
            del incident["victim.industry"]

        # We simplify victim employee counts into 'Small' and 'Large'
        if "victim.employee_count" in incident:
            victim_employee_count = incident["victim.employee_count"]
            if victim_employee_count in ["Small", "Large"]:
                incident["victim.employee_count_simplified"] = victim_employee_count
                incident["victim.employee_count"] = np.nan
            elif victim_employee_count in ["1 to 10", "11 to 100", "101 to 1000"]:
                incident["victim.employee_count_simplified"] = "Small"
            elif victim_employee_count in [
                "1001 to 10000",
                "10001 to 25000",
                "25001 to 50000",
                "50001 to 100000",
                "Over 100000",
            ]:
                incident["victim.employee_count_simplified"] = "Large"

        # We cap unreasonably large monetary amounts to 1e9 USD
        for k, v in incident.items():
            if k.startswith("impact") and k.endswith("amount") and v > 1e9:
                print(
                    k,
                    "for incident",
                    incident["incident_id"],
                    "is unreasonably large:",
                    v,
                    "(most likely a data error), capping it to 1e9 USD",
                )
                incident[k] = 1e9
        return incident


def classify_columns(df, include_regions=False):
    """Classifies the columns of the given DataFrame into info, input and target columns:
    - info columns are those that provide metadata about the incident (e.g. analyst notes),
    - input columns are features of the victim organisation or the incident date (i.e. features
    that can be known before the incident occurs),
    - finally, target columns are those that can be predicted."""

    # Identify info columns by their affixes
    INFO_AFFIXES = [
        "plus.",
        "schema",
        "notes",
        "summary",
        ".year",
        ".month",
        ".day",
        ".locations_affected",
        "_id",
        "analyst",
        "security_incident",
        "confidence",
        "control_failure",
        "discovery_method",
        "value_chain",
        "overall_min_amount",
        "iso_currency_code",
        "victim.secondary",
        "actor.external.name",
        "actor.partner.name",
        ".time",
        ".unit",
        ".value",
        "reference",
        "targeted",
        "victim.state",
        ".cve",
        "malware.name",
    ]

    if not include_regions:
        INFO_AFFIXES += [".region"]
    info_cols = []
    for c in df.columns:
        if any(affix in c for affix in INFO_AFFIXES):
            info_cols.append(c)
        if str(df[c].dtype) == "String":
            nb_vals = len(df[c].unique())
            if nb_vals > 200:
                info_cols.append(c)

    # Input columns are those about the victim organisation or the incident date
    INPUT_PREFIXES = ["victim", "timeline.incident.date"]
    input_cols = [
        c
        for c in df.columns
        if c not in info_cols
        and (any(c.startswith(prefix) for prefix in INPUT_PREFIXES))
    ]

    # Finally, target columns are those that are not info or input columns
    target_cols = []
    for c in sorted(df.columns):
        if c in input_cols or c in info_cols:
            continue
        target_cols.append(c)

    return sorted(info_cols), sorted(input_cols), sorted(target_cols)


def flatten_data(incident_data):
    """Flattens the given incident data dictionary, by converting nested dictionaries
    into dot-separated keys, and processing lists appropriately."""

    flattened = {}
    for k, v in incident_data.items():
        k = k.replace(".", "")
        if isinstance(v, str):
            flattened[k] = v.strip()
        elif isinstance(v, int) or isinstance(v, float):
            flattened[k] = v
        elif isinstance(v, dict):
            v2 = flatten_data(v)
            for sub_k, sub_v in v2.items():
                sub_k = re.sub(r"[\s\-\_]+", "_", sub_k).strip().lower()
                flattened[f"{k}.{sub_k}"] = sub_v
        elif isinstance(v, list):
            list_feats = process_list(k, v)
            for sub_k, sub_v in list_feats.items():
                flattened[sub_k] = sub_v

    #    flattened = {k.replace(".variety", ""): v for k, v in flattened.items()}
    return flattened


def process_list(feat_name, incident_list):
    """Processes a list feature from an incident, returning a dictionary of
    flattened features."""

    new_feats = {}

    if feat_name.endswith("country") or feat_name.endswith("government"):
        reduced_list = [x for x in incident_list if x.lower() not in ["unknown", ""]]
        reduced_list_str = "; ".join(sorted(set(reduced_list))).strip()
        new_feats[feat_name] = reduced_list_str if reduced_list_str else np.nan

    elif feat_name.endswith("_id") or feat_name.endswith(
        ".name"
    ):  # "region", "name", "country"]:
        new_feats[feat_name] = "; ".join(
            sorted([str(x).split(";")[0].strip() for x in incident_list])
        )

    elif "_chain" in feat_name:
        for i, sub_el in enumerate(incident_list):
            v2 = flatten_data(sub_el)
            for sub_k, sub_v in v2.items():
                sub_k = re.sub(r"[\s\-\_]+", "_", sub_k).strip().lower()
                new_feats[f"{feat_name}.{i + 1}.{sub_k}"] = sub_v
    else:
        for sub_el in incident_list:
            if isinstance(sub_el, str):
                sub_el = normalise_feat_name(sub_el)
                if sub_el.lower() != "unknown":
                    new_feats[f"{feat_name}.{sub_el}"] = True
            elif isinstance(sub_el, dict):
                if "variety" in sub_el:
                    variety = normalise_feat_name(sub_el["variety"])
                    new_feats[f"{feat_name}.{variety}"] = True
                    for other_in_el in [
                        sub_el2 for sub_el2 in sub_el if sub_el2 != "variety"
                    ]:
                        other_in_el = normalise_feat_name(other_in_el)
                        new_feats[f"{feat_name}.{variety}.{other_in_el}"] = sub_el[
                            other_in_el
                        ]
                else:
                    raise RuntimeError("Cannot handle", sub_el)
            else:
                raise RuntimeError("Cannot handle", sub_el, type(sub_el))

    return new_feats


def normalise_feat_name(feat_name):
    """Normalises the given feature name by replacing disallowed characters with underscores."""
    allowed_chars = set(string.ascii_letters + string.digits + "_" + ".")
    return (
        "".join([(c if c in allowed_chars else "_") for c in feat_name])
        .strip("_")
        .strip()
    )


def add_ordinals(df: pandas.DataFrame):
    """Adds ordinal columns to the given DataFrame for date, rating, overall_rating
    and employee_count columns."""

    for col in df.columns:
        if col.endswith(".date"):  # type: ignore
            df[col + "_ordinal"] = (
                df[col].map(pandas.Timestamp.toordinal).astype(np.float64)
            )
        elif col.endswith(".rating"):  # type: ignore
            df[col + "_ordinal"] = (
                df[col]
                .map(
                    lambda x: RATING_ORDINAL.index(x) if x in RATING_ORDINAL else np.nan
                )
                .astype(np.float64)
            )
        elif col.endswith(".overall_rating"):  # type: ignore
            df[col + "_ordinal"] = (
                df[col]
                .map(
                    lambda x: OVERALL_RATING_ORDINAL.index(x)
                    if x in OVERALL_RATING_ORDINAL
                    else np.nan
                )
                .astype(np.float64)
            )
        elif col.endswith(".employee_count"):  # type: ignore
            df[col + "_ordinal"] = (
                df[col]
                .map(
                    lambda x: EMPLOYEE_COUNT_ORDINAL.index(x)
                    if x in EMPLOYEE_COUNT_ORDINAL
                    else np.nan
                )
                .astype(np.float64)
            )
    return df
