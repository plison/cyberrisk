import pandas, dataclasses, datetime
from typing import Tuple, List, Dict, Optional, Tuple
import numpy as np
import sklearn.preprocessing
import os, re
from typing import List, Tuple, Dict, Optional, Generator
import random
import matplotlib.pyplot as plt
import seaborn, IPython, pandas
seaborn.set_style("whitegrid")
import pickle

"""
Utilities for representing the incident data and the contextual information surrounding it 
(customers, failed logins, vulnerabilities, maturity assessments, etc.)
"""


@dataclasses.dataclass
class Customer:
    """Meta-data about a specific customer"""

    # Customer ID
    number : int

    # customer size based on nb of employees (rank from 1 to 9) 
    employee_range_rank: int

    # customer size based on turnover (rank from 1 to 11)
    turnover_range_rank : int

    # List of business sectors where the customer is active
    sectors: List[str]

    # List of office locations for the customer
    office_locations: List[str]

    # List of locations where the customer offers its services
    service_locations: List[str]

    def __post_init__(self):
        for locations in [self.office_locations, self.service_locations]:
            for location in locations:
                if len(location) > 3:
                    raise ValueError("Invalid location: %s"%location)
    

@dataclasses.dataclass
class Incident:
    """Data about a specific incident"""

    # Timestamp for the registration of the incident
    date: datetime.datetime

    # Initial priority level
    initial_priority: str

    # Final priority level
    priority: str

    # Customer associated with the incident
    customer: Customer

    def __post_init__(self):
        if type(self.date)==str:
            self.date = datetime.strptime(self.date, "%m.%d.%Y %H:%M")
        for priority in [self.priority, self.initial_priority]:
            if priority not in ["low", "medium", "high", "critical"]:
                raise ValueError("priority %s is not valid"%priority)

@dataclasses.dataclass
class FailedLogins:
    """Data about failed logins observed in the customer infrastructure"""

    customer: Customer
    severity: str
    date: datetime.datetime
    count: int

class IncidentData:
    """Representation of a full dataset of incidents that occurred at customer sites,
    along with contextual information on:
    - the customer itself
    - foundational maturity assessments for the customer
    - number of failed logins
    - vulnerabilities on hosts

    The current code is geared towards reading the Excel files provided by Mnemonic.
    """

    def __init__(self, file="../../data/data_til_nr/nr of incident and org context data anonomyzed.xlsx"):

        self.customers:Dict[int, Customer] = {} 
        self.incidents_by_customer:Dict[int,List[Incident]] = {}

        self._read_customers(file)
        self._read_incidents(file)

        for customer_id in self.incidents_by_customer:
            self.incidents_by_customer[customer_id] = sorted(self.incidents_by_customer[customer_id], key=lambda x: x.date)

    def _read_customers(self, file, sheet_name="Org Data"):
        """Reads the customer data in the "Org Data" sheet"""

        orgs = pandas.read_excel(file, sheet_name=sheet_name)

        self.employee_ranges = orgs.iloc[17:].set_index("Unnamed: 4")["Unnamed: 5"].to_dict()
        self.turnover_ranges = orgs.iloc[17:].set_index("Unnamed: 9")["Unnamed: 10"].to_dict()

        for _, row in orgs.iloc[3:14].iterrows():
            if np.isnan(row["Unnamed: 1"]):
                break
            customer = Customer(number=row["Unnamed: 1"],
                                employee_range_rank = row["Unnamed: 2"],
                                turnover_range_rank = row["Unnamed: 3"],
                                sectors = [sector.strip() for sector in row["Unnamed: 4"].split(",")],
                                office_locations = [loc.strip() for loc in row["Unnamed: 5"].split(",")],
                                service_locations = [loc.strip() for loc in row["Unnamed: 6"].split(",")])
            self.customers[customer.number] = customer
            
    def _read_incidents(self, file, sheet_name="Hendelser"):
        """Reads the incident data in the "Hendelser" sheet"""
        for customer_id in self.customers:
            self.incidents_by_customer[customer_id] = []

        hendelser = pandas.read_excel(file, sheet_name=sheet_name)
        for i, row in hendelser.iloc.iterrows():
            customer_id = row["CustomerName"]
            if type(customer_id)==str:
                customer_id = int(customer_id.split(" ")[0])        
            incident = Incident(initial_priority=row['InitialPriority'],
                                priority=row['Priority'], 
                                customer=self.customers[customer_id],
                                date=row['CreatedTimestamp'])
            self.incidents_by_customer[customer_id].append(incident)
        
        print("Number of incidents per customers:", {k:len(v) for k, v in self.incidents_by_customer.items()})

    def add_failed_logins(self, filename="../../data/data_til_nr/brute_force_count_anonymized.csv"):
        """Adds the failed logins to the dataset"""

        self.failed_logins_by_customer = {}
        df = pandas.read_csv(filename)
        for customer_id, customer in self.customers.items():
            self.failed_logins_by_customer[customer_id] = []
            for _, row in df[df.customer==customer_id].iterrows():
                failed_logins = FailedLogins(customer=customer, severity=row.severity, 
                                             date=pandas.to_datetime(row.month), count=row["count"])
                self.failed_logins_by_customer[customer_id].append(failed_logins)
        return self

    def add_cvss(self, directory="../../data/data_til_nr/"):
        """Adds the vulnerabilities per host to the dataset"""

        self.cvss_scores_by_customer_and_host = {}
        for f in os.listdir(directory):
            match = re.match("vulnerabilities\_customer\_(\d+)\.csv", f)
            if match:
                customer_id = int(match.group(1))
                self.cvss_scores_by_customer_and_host[customer_id] = {}
                print("Reading csv for customer", customer_id)
                with open(directory + f) as fd:
                    for i, line in enumerate(fd):
                        if i==0:
                            continue
                        date, host, score = line.rstrip("\n").split(",")
                        date = pandas.to_datetime(date)
                        if host not in self.cvss_scores_by_customer_and_host[customer_id]:
                            self.cvss_scores_by_customer_and_host[customer_id][host] = {date:score}
                        else:
                            self.cvss_scores_by_customer_and_host[customer_id][host][date] = score

                # Filling in the holes
                last_incident_date = self.incidents_by_customer[customer_id][-1].date
                for host in self.cvss_scores_by_customer_and_host[customer_id]:
                    sorted_dates = sorted(self.cvss_scores_by_customer_and_host[customer_id][host].keys())
                    for date in sorted_dates:
                        cvss_score = self.cvss_scores_by_customer_and_host[customer_id][host][date]
                        max_nb_days = (last_incident_date - date).days
                        for i in range(1,max_nb_days):
                            next_date = date + datetime.timedelta(days=i)
                            if next_date in self.cvss_scores_by_customer_and_host[customer_id][host]:
                                break
                            self.cvss_scores_by_customer_and_host[customer_id][host][next_date] = float(cvss_score)
        return self
    
    def add_fma(self, file="../../data/data_til_nr/FAM data.xlsx"):
        """Adds the foundational maturity assessment to the dataset"""

        df = pandas.read_excel(file, sheet_name="Scoring")
        self.fma_by_customer = {}
        for customer_id in df.columns[1:]:
            self.fma_by_customer[int(customer_id)] = {}
            for i, criteria in df[df.columns[0]].items():
                self.fma_by_customer[int(customer_id)][criteria] = df.loc[i][customer_id]
        return self

    def save(self, filename="incident_data.pkl"):
        with open(filename, "wb") as fd:
            pickle.dump(self, fd)

    @classmethod
    def load(cls, filename="incident_data.pkl"):
        with open(filename, "rb") as fd:
            result = pickle.load(fd)
        return result

    def get_aggregated_records(self, customer_id, period=15):
        """Creates a dataframe including all incidents from all customers and priority levels,
        and aggregating the number of incidents by the specified period."""
        records = []
        start_date = self.incidents_by_customer[customer_id][0].date - datetime.timedelta(days=1)
        cutoff_date = self.incidents_by_customer[customer_id][-1].date 
        max_nb_days = (cutoff_date - start_date).days - period
        for i in range(0, max_nb_days, period):
            current_date = start_date + datetime.timedelta(days=i)
            mid_date = current_date + datetime.timedelta(days=period/2)
            end_date = current_date + datetime.timedelta(days=period)
            for priority in ["low", "medium", "high", "critical"]:
                count = 0
                for incident in self.incidents_by_customer[customer_id]:
                    if (incident.date > current_date and incident.date <= current_date + datetime.timedelta(days=period)
                            and incident.priority == priority):
                        count += 1
                record = {"start_date":current_date, "period":mid_date, "end_date":end_date, 
                          "nb_incidents":count, "customer":customer_id, "priority":priority}
                records.append(record)
        df = pandas.DataFrame.from_records(records)
        return df
