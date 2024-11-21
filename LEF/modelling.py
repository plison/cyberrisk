import pandas, dataclasses, datetime
from typing import Tuple, List, Dict, Optional, Tuple
import numpy as np
import sklearn.preprocessing
import os, re
from typing import List, Tuple, Dict, Optional, Generator
import random
import matplotlib.pyplot as plt
import matplotlib.dates
import seaborn, IPython, pandas
seaborn.set_style("whitegrid")

from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge, QuantileRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier, DummyRegressor
import sklearn.metrics
import sklearn.preprocessing

import tqdm
import optuna
import json, re 
import sklearn.ensemble 
import time

from preprocessing import IncidentData

# When running the evaluation, we wish to assign a higher weight to the more serious incidents
PRIORITY_WEIGHT = {"low":1, "medium":2, "high":10, "critical":50}


# TODO:
# further optimize parameters
# include other variables
# (and handle missing values)
# include other datasets
# hierarchical modelling
# explanations
# embeddings?

@dataclasses.dataclass
class Parameters:
    """Set of hyper-parameters for the forecasting model"""

    # Period (in days) for which we wish to predict the number of incidents
    forecast_period:int = 30

    # Whether to encode the ranges (for employees and turnover) through a numerical variable
    include_range_scales:bool = False

    # Whether to encode the ranges (for employees and turnover) through one-hot variables
    include_one_hot_ranges:bool = False

    # Past periods (in days) from which to extract the number of registered incidents,
    # and include as input features
    past_periods:Tuple[int] = ()

    # Whether to include the maximum number of incidents that occurred on the same day 
    # as input features (stretching X days in the past, were X is a past_period)
    max_per_day:bool = False

    # Whether to include the office and service locations as features
    include_locations:bool = False

    # Whether to include the business sector as features
    include_sectors:bool = False

    # Whether to include the number of failed logins in the last month as features 
    include_failed_logins:bool=False

    # Whether to include a feature denoting the recency of the forecasting date 
    include_recency:bool = False

    # Whether to include CVSS data as features
    include_cvss:bool=False

    # Whether to include Foundational Maturity Assessments (FMA) as features
    include_fma:bool=False

    # Whether to normalise the input features
    normalise:bool = False

    # Regularisation strength
    regularisation: float = 1.0

    # Whether to apply a differencing mechanism, which is particularly useful for 
    # tree-based models (see e.g. https://otexts.com/fpp2/stationarity.html)
    differencing:bool = False

    # Whether to estimate a hurdle model, which is made of two models: 
    # - one classifier predicting the probability of a nb of accidents > 0
    # - one regressor predicting the nb of accidents if we know it is > 0 
    hurdle:bool = False

    # Whether to apply a time decaying mechanism (useful when training, to 
    # give a higher weight to more recent points) 
    yearly_decay:float = 0.0

    # The type of forecasting model (currently supported: poisson or hgbt)
    regressor:str = "poisson"

    # The confidence interval to use when forecasting a range
    confidence_interval = 0.9

    # The solver to use for fitting linear models
    solver = "newton-cholesky"

    def __post_init__(self):
        self.check_consistency()

    def check_consistency(self):
        """Checks whether the parameters are consistent"""

        if self.differencing:
            if self.forecast_period not in self.past_periods:
                raise RuntimeError("To use differencing, you must have the forecasting period in self.past_periods")
            elif self.normalise:
                raise RuntimeError("Cannot use differencing and normalisation together")
        if self.max_per_day and not self.past_periods:
            print("Warning: max per day is activated, but without specified periods")
        if self.regressor not in ["poisson", "hgbt"]:
            raise RuntimeError("Invalid regressor:", self.regressor)
        if self.forecast_period <= 0:
            raise RuntimeError("Invalid forecasting period:", self.forecast_period)

class Forecaster:
    """Forecaster model, trained on past incident data"""

    def __init__(self, parameters:Optional[Parameters]=None):
        """Initialises a new forecaster using the provided hyper-parameters"""

        self.params = parameters if parameters else Parameters()

        # The main forecasting models (one per priority level)
        self.models = {}
        
        # The hurdle models (if a hurdle mechanism is employed, else it is discarded)
        self.hurdle_models = {}

        # The models for predicting the lower and upper bounds (within a confidence interval)
        self.lower_models = {}
        self.upper_models = {}

    
    def train(self, incident_data:IncidentData):
        """Trains the forecasting model based on the provided incident data"""

        # Creates a generator of training examples
        generator = ForecastingDataGenerator(incident_data=incident_data, parameters=self.params)

        # Train the models for each priority level
        for priority in ["low", "medium", "high", "critical"]:
            X_train, y_train, w_train = generator.generate(priority, decay_weights=True)
            self.train_for_priority(priority, X_train, y_train, w_train)


    def train_for_priority(self, priority:str, X_train:List[Dict[str,float]], 
                            y_train:List[int], w_train: Optional[List]=None):
        """Train the forecasting models to predict the number of incidents that will occur 
          within the forecasting period and priority level
        
        Arguments:
        - priority: the priority level to train for
        - X_train the input features for each data point
        - y_train: the target number of incidents
        - w_train: if set, the sample weights """

        if len(X_train) != len(y_train) or (w_train and len(w_train) != len(y_train)):
            raise RuntimeError("X_train, y_train, w_train must have the same size, but are",
                               len(X_train), len(y_train), len(w_train) if w_train is not None else 0)

        # If no weights are provided, assume uniform weights
        w_train = np.ones(len(y_train)) if w_train is None else w_train

        # Shuffle the training data
        X_train, y_train, w_train = sklearn.utils.shuffle(X_train, y_train, w_train)
        y_train = pandas.Series(y_train, dtype=np.float32)
        w_train = pandas.Series(w_train, dtype=np.float32)

        # Preprocess the input features
        X_train = self.preprocess(X_train, refit=True)

        # Initialise the forecasting models
        self._initialise_models(priority, X_train, y_train)

        # Fits the hurdle model if needed
        if priority in self.hurdle_models:
            t_start = time.time()        
            self.hurdle_models[priority].fit(X_train, y_train > 0, w_train)
            non_zero_indices = y_train[y_train != 0].index
            X_train = X_train.loc[non_zero_indices]
            y_train = y_train[non_zero_indices]
            w_train = w_train[non_zero_indices]

        if self.params.differencing:
            past_value = "nb_past_%s_incidents_last_%idays"%(priority, self.params.forecast_period)
            y_train = y_train - X_train[past_value]

        # Fits the main forecasting models
        self.models[priority].fit(X_train, y_train, sample_weight=w_train)
        
        # Fits the lower and upper bound models
        if priority in self.lower_models:
            self.lower_models[priority].fit(X_train, y_train, sample_weight=w_train)
            self.upper_models[priority].fit(X_train, y_train, sample_weight=w_train)

        return self

  
    def _initialise_models(self, priority:str, X_train:pandas.DataFrame, y_train:pandas.Series):
        """(Re)-initialises the forecasting models for the specified priority level. The exact choice
        of model will depend on both the hyper-parameters and the training data"""

        # Removing older models
        for sub_models in [self.hurdle_models, self.models, self.lower_models, self.upper_models]:
            if priority in sub_models:
                del sub_models[priority]

        # To use a hurdle model, we need to have both zero and non-zero values
        if self.params.hurdle and sum(y_train==0) > 0 and sum(y_train > 0) > 0:

            # if we have no features, there is nothing to train, so we just use a dummy classifier
            if len(X_train.columns) == 0:
                self.hurdle_models[priority] = DummyClassifier()

            # Otherwise, we build a classifier of same type as the main model
            elif self.params.regressor=="poisson":
                self.hurdle_models[priority] = LogisticRegression(solver=self.params.solver, C=1/self.params.regularisation)
            elif self.params.regressor=="hgbt":
                self.hurdle_models[priority] = HistGradientBoostingClassifier()

        # If we have not input features, or all targets are 0, there is nothing to train
        if sum(y_train)== 0 or len(X_train.columns)==0:
            if sum(y_train)== 0:
                print("Problem: all our values are 0")
            self.models[priority] = DummyRegressor()

        elif self.params.regressor=="poisson":
            # We build either a Poisson model or a ridge model, depending on whether differencing is applied
            if self.params.differencing:
                self.models[priority] = Ridge(alpha=self.params.regularisation)
            else:
                self.models[priority] = PoissonRegressor(solver=self.params.solver, alpha=self.params.regularisation)
        
        # If the regressor relies on histogram-based gradient boosting trees, we choose a squared_error if
        # we apply differencing, and a poisson loss otherwise
        elif self.params.regressor=="hgbt":
            loss = "poisson" if not self.params.differencing else "squared_error"
            self.models[priority] = HistGradientBoostingRegressor(loss=loss, l2_regularization=self.params.regularisation)
        else:
            raise RuntimeError("Model could not be initialized")
        
        # If we also need to computer lower and upper bounds, we build similar models as above, but with a quantile loss
        if self.params.confidence_interval:
            lower_quantile = (1-self.params.confidence_interval)/2
            upper_quantile = self.params.confidence_interval + lower_quantile
            if sum(y_train)== 0 or len(X_train.columns)==0:
                self.lower_models[priority] = DummyRegressor(strategy="quantile", quantile=lower_quantile)
                self.upper_models[priority] = DummyRegressor(strategy="quantile", quantile=upper_quantile)
            elif self.params.regressor=="poisson":
                self.lower_models[priority] = QuantileRegressor(solver="highs-ds", alpha=self.params.regularisation, 
                                                                quantile=lower_quantile)
                self.upper_models[priority] = QuantileRegressor(solver="highs-ds", alpha=self.params.regularisation, 
                                                                quantile=upper_quantile)
            elif self.params.regressor=="hgbt":
                self.lower_models[priority] = HistGradientBoostingRegressor(loss="quantile", quantile=lower_quantile,
                                                                            l2_regularization=self.params.regularisation)
                self.upper_models[priority] = HistGradientBoostingRegressor(loss="quantile", quantile=upper_quantile,
                                                                            l2_regularization=self.params.regularisation)

        return self


    def predict(self, X_test:List[Dict[str,float]], priorities:Optional[List[str]]=None, return_ci=False, debug=False):
        """Given a list of input data points (each encoded with a set of input features), predicts the
        number of incidents estimated for the forecasting period, factored by priority levels.
        
        Arguments:
        - X_test: the input, expressed as a list of data points. Each data point is expressed 
                  as a dictionary mapping feature names to their values
        - priorities: if set, only computes the predictions for the specified priorities
                      (otherwise, returns the forecasts for all priority levels)
        - return_ci: if True, returns the lower and upper bounds as well
        """

        if not self.models:
            raise RuntimeError("Model must first be trained")

        forecasts = {}
        X_test = self.preprocess(X_test, refit=False)    

        priorities = priorities if priorities else ["low", "medium", "high", "critical"]
        for priority in priorities:

            forecasts[priority] = {}

            # Generate the main forecast
            forecast = self.models[priority].predict(X_test)
            if debug:
                forecasts[priority]["base_output"] = forecast

            # If we use differencing, set forecast = past_value + difference
            if self.params.differencing:
                past_value = "nb_past_%s_incidents_last_%idays"%(priority, self.params.forecast_period)
                forecast = X_test[past_value] + forecast
                forecast = np.clip(forecast, a_min=0, a_max=None)
                if debug:
                    forecasts[priority]["previous_value"] = X_test[past_value]

            # If we use a hurdle model, incorporate the probability P(nb=0)
            if priority in self.hurdle_models:
                prob_incidents = self.hurdle_models[priority].predict_proba(X_test)[:,1]
                forecast = prob_incidents * forecast
                if debug:
                    forecasts[priority]["prob_zero_incidents"] = 1-prob_incidents

            # Register the final point estimate
            forecasts[priority]["mean"] = forecast

            # If 
            if return_ci:
                lower_bound = self.lower_models[priority].predict(X_test)
                upper_bound = self.upper_models[priority].predict(X_test)
                if self.params.hurdle:
                    lower_bound = hurdle_forecasts * lower_bound
                    upper_bound = hurdle_forecasts * upper_bound

                if self.params.differencing:
                    lower_bound = X_test[past_value] + lower_bound
                    upper_bound = X_test[past_value] + upper_bound
                
                forecasts[priority]["lower_bound"] = lower_bound
                forecasts[priority]["upper_bound"] = upper_bound
            
        return forecasts


    def evaluate(self, incident_data:IncidentData):
        """Evaluates the performance of the forecasting model on the incident data. The evaluation is done
        by cross-validation. For each fold, we leave out one customer from the training set, and then assess
        the forecasting performance of the trained models on the left-out customer.
        
        Arguments:
        - incident data: the full dataset on which to conduct the evaluation
        """

        print("Evaluation parameters:", self.params)
        customers = list(incident_data.customers.keys())
        records = []

        # We create the generator for the incident data
        generator = ForecastingDataGenerator(incident_data=incident_data, parameters=self.params)

        # We repeat the process for all priority levels, and all customers
        for priority in ["low", "medium", "high", "critical"]:
            records_for_priority = []
            for customer_to_test in customers:

                # Step 1: we train the models on a training set where we leave out one customer
                X_train, y_train, w_train = generator.generate(priority, customers_to_skip=[customer_to_test])
                self.train_for_priority(priority, X_train, y_train, w_train)

                # Step 2: We test the forecasting performance on that customer
                X_test, y_test, w_test = generator.generate(priority, customer_ids=[customer_to_test])
                forecast = self.predict(X_test, [priority])[priority]["mean"]

                # We compute here the mean absolute error and mean squared error
                mae = sklearn.metrics.mean_absolute_error(y_test, forecast, sample_weight=w_test)
                rmse = sklearn.metrics.root_mean_squared_error(y_test, forecast, sample_weight=w_test)
        
                record = {"priority":priority, "tested_customer":customer_to_test, 
                        "mae":mae, "rmse":rmse, "weighted_mae": mae * PRIORITY_WEIGHT[priority],
                        "mean_true":np.mean(y_test), "mean_pred":np.mean(forecast), 
                        **dataclasses.asdict(self.params)}   
                records_for_priority.append(record)

            records += records_for_priority

            print("Cross-validated MAE for priority %s: %.2f (+- %.2f)"%
                  (priority, np.mean([r["mae"] for r in records_for_priority]), 
                   np.std(([r["mae"] for r in records_for_priority]))))

        return np.mean([r["weighted_mae"] for r in records]) 
    

    def preprocess(self, feats:List[Dict[str,float]], refit=False):
        """Preprocess the input features by:
        1) converting them to a pandas Dataframe
        2) restricting the dataframe to the features employed for the forecast
        3) If self.params.normalise is True, normalising the feature values
        """

        feats_df = pandas.DataFrame.from_records(feats)

        if len(feats)!= len(feats_df):
            feats_df = pandas.DataFrame(index=range(len(feats)))
            
        if refit:
            self.feature_names = list(feats_df.columns)
        else:
            for f in self.feature_names:
                if f not in feats_df.columns:
                    feats_df[f] = np.nan
            feats_df = feats_df[self.feature_names]

        if self.params.normalise or self.params.regressor =="poisson":
            feats_df = feats_df.fillna(0)
                   
        if self.params.normalise and len(self.feature_names) > 0 :
            if refit:
                self.scaler = sklearn.preprocessing.StandardScaler()
                feats_df.values[:] = self.scaler.fit_transform(feats_df)
            else:
                feats_df.values[:] = self.scaler.transform(feats_df)
                
        return feats_df
    


class ForecastingDataGenerator:
    """Utility for generating training/test data from the incident data"""

    def __init__(self, incident_data:IncidentData, parameters: Optional[Parameters]=None):
        """Initialises the generator based on the provided incident data and parameters"""

        self.incident_data = incident_data
        self.params = parameters if parameters else Parameters()

        # We compute the start and cutoff dates for the incident data about each customer
        self.start_dates = {}
        self.cutoff_dates = {}
        for customer_id, incidents in self.incident_data.incidents_by_customer.items():
            self.start_dates[customer_id] = min([incident.date for incident in incidents]) - datetime.timedelta(days=1)
            self.cutoff_dates[customer_id] = max([incident.date for incident in incidents])
            
        # We also initialise a cache to speed up the data generation
        self.cache = {}


    def generate(self, priority:str, customer_ids:Optional[List[int]]=None, 
                 customers_to_skip:Optional[List[int]]=None, 
                 selected_dates:Optional[List[datetime.datetime]]=None,
                 decay_weights=False) -> Tuple[List[Dict[str,float]], List[float], List[float]]:
        """Given a particular priority level and customers to either include or exclude, generate a triple (X,y,w) where:
        - X represents for each day d the set of input features about the customer and the state of its
          technical infrastructure on that day d
        - y represents the number of incidents of the given priority level that have occurred between (d+1) and (d+n),
          where n i the forecasting period
        - w is a list of weights for each day, representing the importance of that datapoint. The weight is
          set to 1 for all "normal" examples, but lower for examples that are close to the start and end dates,
          and have therefore truncated numbers of incidents

        Arguments:
        - customer_ids: the list of customer to include in the generation. If not set, include all.
        - customers_to_skip: the list of customers to exclude from the generation (if any)
        - selected_dates: the list of dates on which to generate samples. If not set, consider 
                             all dates from start to cutoff
        - decay_weights: if set, increase the weights of more recent examples
          """ 

        if priority not in ["low", "medium", "high", "critical"]:
            raise RuntimeError("Invalid priority:", priority)

        if not customer_ids:
            customer_ids = list(self.incident_data.customers.keys())       
            
        all_feats, all_nb_incidents, all_weights = [], [], []
        for customer_id in customer_ids:
            if customers_to_skip and customer_id in customers_to_skip:
                continue

            if selected_dates is None:
                # Calculate the list of dates to include (from start to cutoff)
                max_nb_days = (self.cutoff_dates[customer_id] - self.start_dates[customer_id]).days
                selected_dates = [(self.start_dates[customer_id] + datetime.timedelta(days=nb_days)) 
                                     for nb_days in range(1, max_nb_days)]
            
            # If the data was not yet computed for the (priority, customer_id) pair, we generate it
            if (priority, customer_id) not in self.cache:
                feats, nb_incidents, weights = zip(*self._generate_for_customer(priority, customer_id, selected_dates, decay_weights))
                self.cache[(priority, customer_id)] = feats, nb_incidents, weights

            # Otherwise, we simply retrieve the cached data
            else:
                feats, nb_incidents, weights = self.cache[(priority, customer_id)]
            
            all_feats += feats
            all_nb_incidents += nb_incidents
            all_weights += weights
        
        return all_feats, all_nb_incidents, all_weights


    def _generate_for_customer(self, priority:str, customer_id:int, 
        selected_dates:List[datetime.datetime], decay_weights=False) -> Generator[Dict[str,float], float, float]:
        """Same as the main generate method, but for a single customer."""

        # We retrieve the customer features (static for all dates)
        customer_feats = self.get_customer_features()
   
        for current_date in selected_dates:
            
            # We combine the static customer features with the endogenous and exogenous features
            features = dict(customer_feats[customer_id])
            features.update(self.get_endogenous_features(customer_id=customer_id, current_date=current_date))
            features.update(self.get_exogenous_features(customer_id=customer_id, current_date=current_date))
            
            # We compute the nb of indidents occurring between the current state and the end of the
            # forecast period, for the given customer and priority level
            nb_incidents, count_quality = self.get_nb_future_incidents(current_date=current_date,
                                            forecast_period=self.params.forecast_period, 
                                            customer_id=customer_id, incident_priority=priority)
            
            # The counts of incidents for dates that are close to the cutoff dates are less precise, because we
            # don't have the counts after the cutoff date. We compensate for this lack by guessing the remaining
            # part. However, those data points are of lower quality, and should therefore have a lower weight.
            weight = count_quality

            # If we wish to decay weights to increase the influence of more recent data points
            if decay_weights:
                nb_years = (self.cutoff_date - current_date).days / 365
                weight *= (1-self.yearly_decay)**nb_years

            yield features, nb_incidents, weight
                          
    def get_customer_features(self):
        """Returns the customer features (according to the features defined in the parameters)"""

        records = []
        for customer in self.incident_data.customers.values():
            record = {}
            if self.params.include_range_scales:
                record["employee_range_scale"] = customer.employee_range_rank
                record["turnover_range_scale"] = customer.turnover_range_rank
            if self.params.include_one_hot_ranges:
                employee_range_str = self.incident_data.employee_ranges[customer.employee_range_rank]
                record["employee_range_%s"%employee_range_str] = 1
                turnover_range_str = self.incident_data.turnover_ranges[customer.turnover_range_rank]
                record["turnover_range_%s"%turnover_range_str] = 1
            if self.params.include_sectors:
                for sector in customer.sectors:
                    record["sector_%s"%sector] = 1
            if self.params.include_locations:
                for office_location in customer.office_locations:
                    record["office_location_%s"%office_location] = 1
                for service_location in customer.service_locations:
                    record["service_location_%s"%service_location] = 1
            records.append(record)
        df = pandas.DataFrame.from_records(records)
        df.index = list(self.incident_data.customers.keys())
        all_feats = df.to_dict(orient="index")
        for customer_id in all_feats:
            all_feats[customer_id] = {k: v for k, v in all_feats[customer_id].items() if pandas.notna(v)}  
        return all_feats
    
    def get_endogenous_features(self, customer_id:int, current_date):
        """Returns the endogenous features (=features extracted from the time series itself) for the customer 
        at the given date"""
        
        features = {}

        # Compute a feature expressing the recency of the date (= how close it is to the cutoff date)
        if self.params.include_recency:
            features["recency"] = ((current_date - self.start_dates[customer_id]) / 
                                   (self.cutoff_dates[customer_id] - self.start_dates[customer_id]))

        # The other endogenous features are all computed for each past_period to consider
        for past_period in self.params.past_periods:
            for past_priority in ["low", "medium", "high", "critical", "all"]:

                # We extract the number of past incidents of a given priority in that period
                feat_val = self.get_past_nb_incidents(current_date, past_period=past_period, 
                                                        customer_id=customer_id, incident_priority=past_priority)
                features["nb_past_%s_incidents_last_%idays"%(past_priority, past_period)] = feat_val

        # As well as the maximum number of incidents occurring on the same day
        if self.params.max_per_day and self.params.past_periods:
            max_period = max(self.params.past_periods)
            incidents_per_day = self.get_nb_incidents_by_day(customer_id, past_priority, current_date, max_period)
            for past_period in self.params.past_periods:
                nb_incidents_per_day_clipped = [nb for date, nb in incidents_per_day.items() 
                                                if date >= (current_date - datetime.timedelta(days=past_period))]
                feature_name = "max_nb_%s_incidents_per_day_last_%idays"%(past_priority, past_period)
                features[feature_name] = max(nb_incidents_per_day_clipped)if nb_incidents_per_day_clipped else 0

        return features
    

    def get_exogenous_features(self, customer_id: int, current_date):
        """Returns the exogenous fetaures (= features extract from contextual information outside of the time series)
        for the customer at the given date"""

        features = {}
        
        # Add the number of failed logins within last month
        if self.params.include_failed_logins and hasattr(self.incident_data, "failed_logins_by_customer"):
            failed_logins = self.incident_data.failed_logins_by_customer[customer_id]
            for severity in ["low", "medium", "high", "critical"]:
                failed_logins2 = [f for f in failed_logins if f.date <= current_date and f.severity==severity]
                if len(failed_logins2)==0:
                    continue
                last_count = max(failed_logins2, key=lambda x: x.date)
                features["nb_failed_logins_%s_last_month"%severity] = int(last_count.count)

        # Add the min, mean and max CVSS scores for the hosts
        if self.params.include_cvss and hasattr(self.incident_data, "cvss_scores_by_customer_and_host"):
            cvss_scores = self.incident_data.cvss_scores_by_customer_and_host[customer_id]
            features["nb_monitored_hosts"] = len(cvss_scores)
            latest_cvss_scores = [float(scores_for_host[current_date]) for scores_for_host in cvss_scores.values() 
                                  if current_date in scores_for_host]
            if len(latest_cvss_scores) > 0:
                features["mean_cvss_score"] = np.mean(latest_cvss_scores)
                features["min_cvss_score"] = np.min(latest_cvss_scores)
                features["max_cvss_score"] = np.max(latest_cvss_scores)

        # Add the FMA scores
        if self.params.include_fma and hasattr(self.incident_data, "fma_by_customer"):
            fma_scores = self.incident_data.fma_by_customer[customer_id]
            features.update(fma_scores)

        return features
                

    def get_past_nb_incidents(self, current_date:datetime.datetime, past_period:int, 
                        customer_id:int, incident_priority="all"):
        """Returns the number of incidents for the customer and priority level that have occurred
        between (current_date - past_period) and current_date"""

        past_nb_incidents = 0
        earliest_date = current_date - datetime.timedelta(days=past_period)
        for incident in self.incident_data.incidents_by_customer[customer_id]:
            if incident.date <= current_date and incident.date >= earliest_date:
                if incident_priority=="all" or incident_priority==incident.priority:
                    past_nb_incidents += 1
            if incident.date > current_date:
                break

        # If (current_date - past_period) is actually before the start date, we need to extrapolate
        # the number of incidents from the counts we have
        if earliest_date < self.start_dates[customer_id]:
            nb_missing_days = (self.start_dates[customer_id] - earliest_date).days
            nb_covered_days = (current_date - self.start_dates[customer_id]).days
            past_nb_incidents += (past_nb_incidents / max(1E-10, nb_covered_days)) * nb_missing_days

        return past_nb_incidents


    def get_nb_future_incidents(self, current_date:datetime.datetime, forecast_period:int, 
                        customer_id:int, incident_priority="all"):
        """Returns the number of incidents that have occurred after the current date, and within
        the forecast period, for the specified customer and priority level"""

        forecast_date = current_date + datetime.timedelta(days=forecast_period)
        forecast = 0
        for incident in reversed(self.incident_data.incidents_by_customer[customer_id]):
            if incident.date > current_date and incident.date <= forecast_date:
                if incident_priority=="all" or incident_priority==incident.priority:
                    forecast += 1
                if incident.date <= current_date:
                    break
                
        coverage = 1.0
        if forecast_date > self.cutoff_dates[customer_id]:
            nb_missing_days = (forecast_date - self.cutoff_dates[customer_id]).days
            nb_covered_days = (self.cutoff_dates[customer_id] - current_date).days
            forecast = forecast + (forecast / max(1E-10, nb_covered_days)) * nb_missing_days
            coverage = nb_covered_days /forecast_period
            if nb_missing_days > nb_covered_days and forecast == 0:
                coverage = 0.1*coverage
        return forecast, coverage

    def get_nb_incidents_by_day(self, customer_id:int, priority:str, current_date:datetime.datetime, past_period:int):
        """Returns a dictionary mapping dates to the number of incidents (for the customer and priority level)
        that have occurred on that date. 
        Only the dates between (current_date-past_period) and current_date are considered."""

        earliest_date = current_date - datetime.timedelta(days=past_period)
        nb_incidents_by_day = {}
        previous_date = None
        for incident in self.incident_data.incidents_by_customer[customer_id]:
            if incident.date >=earliest_date and incident.date <=current_date and incident.priority==priority:
                if previous_date and (incident.date - previous_date).days == 0:
                    nb_incidents_by_day[previous_date] += 1
                else:
                    nb_incidents_by_day[incident.date] = 1
                    previous_date = incident.date
            if incident.date > current_date:
                break
        return nb_incidents_by_day




class Tuner:
    """Optuna optimizer for the selection of optimal hyper-parameters """

    def __init__(self, incident_data:IncidentData=None, study_name:str="tuning", 
                 reset_study=False, past_periods=(10,30,100, 365)):
        if incident_data is None:
            incident_data = IncidentData.load()
        self.incident_data = incident_data
        if reset_study or not os.path.exists("%s.db"%study_name):
            print("Creating new study:", study_name)
            self.study = optuna.create_study(study_name=study_name, storage="sqlite:///%s.db"%study_name, 
                                             direction='minimize')
        else:
            self.study = optuna.load_study(study_name=study_name, storage="sqlite:///%s.db"%study_name)
        self.past_periods = past_periods
     

    def select_parameters(self, trial):
        p = Parameters()
        p.confidence_interval = 0.0
        p.include_range_scales = trial.suggest_categorical("include_range_scale", [True, False])
        p.include_one_hot_ranges = trial.suggest_categorical("include_one_hot_ranges", [True, False])
        if len(self.past_periods) > 0:
            self.past_periods = random.sample(self.past_periods, len(self.past_periods))
            nb_past_periods = trial.suggest_int("nb_past_periods", 0, len(self.past_periods))
            p.past_periods = self.past_periods[:nb_past_periods] 
            p.max_per_day = trial.suggest_categorical("max_per_day", [True, False])

        p.include_locations = trial.suggest_categorical("include_locations", [True, False])
        p.include_sectors = trial.suggest_categorical("include_sectors", [True, False])
        p.include_failed_logins = trial.suggest_categorical("include_failed_logins", [True, False])
        p.include_cvss = trial.suggest_categorical("include_cvss", [True, False])
        p.include_fma = trial.suggest_categorical("include_fma", [True, False])
        p.include_recency = trial.suggest_categorical("include_recency", [True, False])
  #      p.yearly_decay = trial.suggest_float("yearly_decay", 0.01, 0.95)
        p.hurdle = trial.suggest_categorical("hurdle", [True, False])
        p.regressor = trial.suggest_categorical("regressor", ["poisson", "hgbt"])
        p.regularisation = trial.suggest_float("regularisation", 0.01, 100)
        if p.regressor == "hgbt" and p.forecast_period in p.past_periods:
            p.differencing = trial.suggest_categorical("differencing", [True, False])
        p.normalise = p.regressor == "poisson"
        if p.regressor == "poisson":
            p.solver = "newton-cholesky" # trial.suggest_categorical("solver", ["newton-cholesky", "lbfgs"])
        return p

    def objective(self, trial):

        for i in range(10):
            try:
                p = self.select_parameters(trial)
                p.check_consistency()
                forecaster = Forecaster(p)
                r = forecaster.evaluate(self.incident_data)
                return r
            except KeyboardInterrupt:
                raise KeyboardInterrupt()
            except BaseException as e:
                print("Error with parameter selection:", e)

        raise RuntimeError("Could not select valid parameter")

    
    def run(self, n_trials=500):
        return self.study.optimize(self.objective, n_trials=n_trials)
