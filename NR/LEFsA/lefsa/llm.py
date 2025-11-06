# -*- coding: utf-8 -*-

import json
import os

import openai

from lefsa.osint_data import InputFeatures

from .survey import OrgFeatures

LLM_ENDPOINT = "https://plison-2341-resource.cognitiveservices.azure.com/"

LLM_PROMPT_VCDB = """
You are given the following information about an organisation: 
{company_description}

Based on what you know or can reasonably assume about that organisation, generate a JSON output containing 
the following 7 variables:
- country: the country of operation of the organisation (primary location in case of a multinational organisation)
- continent: the continent where the organisation is primarily located
- employee_count: the number of employees of the organisation
- employee_count_simplified: a simplified categorization of the employee count ('Small' or 'Large')
- government: whether the organisation is a government entity, and if yes, where
- industry: the primary industry of the organisation
- revenue: annual revenue of the organisation, in USD

Those variables can only be assigned the following values in your JSON output:
- country: ISO 3166 code of the country (2-letter code, e.g. NO for Norway, US for United States, etc.)
- continent, among 'Asia','Europe','North America', 'South America','Africa', 'Oceania' or 'Antarctica'
- employee_count: ['1 to 10', '11 to 100', '101 to 1000', '1001 to 10000', '10001 to 25000', 
                  '25001 to 50000', '50001 to 100000', 'Over 100000']
- employee_count_simplified: either 'Small' or 'Large' where 'Small' means 1000 employees or less, and 'Large' means more than 1000 employees
- government: ['NA', 'local', 'regional', 'federal'] (use 'NA' if the organisation is not a government entity)
- industry: a NAICS code (2 to 6 digits, depending on how specific you can be - the more the better)
- revenue: an integer number expressing the approximate annual revenue of the organisation, in USD

You should only specify a value if you feel reasonably confident about your guess. Otherwise, do not include 
the variable in your JSON output. Just output the JSON object, without any other text or explanation.
"""


LLM_PROMPT_SURVEY = """
You are given the following information about a Norwegian organisation: 
{company_description}

Based on what you know/assume about that company, generate a JSON output containing the following variables: 
- antall_ansatte (number of employees of the organisation), 
- bransje (industry of the organisation), 
- sektor (public or private), 
- samfunnskritiske_tjenester (whether the organisation provides services that are essential for maintaining the safety, 
  health, and basic functioning of society, even during crises or emergencies), 
- rammeverk_for_informasjonssikkerhet (whether the organisation uses an information security management system), 
- outsourcet (whether the organisation outsources its IT infrastructure) and 
- landsdel (the location of the organisation in Norway). 

Those variables can only be assigned the following values in your JSON output:
- antall_ansatte: "5 til 19 ansatte", "20 til 99 ansatte","100 ansatte eller flere"
- bransje: "Industri etc.", "Bygg- og anleggsvirksomhet", "Varehandel etc.", "Transport og lagring", 
"Overnattings- og serveringsvirksomhet", "Tjenesteytende næringer", "Offentlig administrasjon", 
"Undervisning", "Helse og sosial", "Kulturell virksomhet"
- sektor: "Privat", "Offentlig"    
- samfunnskritiske_tjenester: "Ja", "Nei"
- rammeverk_for_informasjonssikkerhet: "Ja", "Nei"
- outsourcet: "Helt outsourcet", "Delvis outsourcet", "Organisert internt"  
- landsdel: "Nord-Norge", "Vestlandet", "Østlandet", "Sørlandet inkludert TeVe", "Midt-Norge", "Oslo"

You should only specify a value if you feel reasonably confident about your classification. Otherwise, do not include 
the variable in your JSON output. Just output the JSON object, without any additional text or explanation.
"""


def create_survey_feats_from_description(org_description: str) -> OrgFeatures:
    """Creates an instance of OrgFeatures by querying a language model with the organisation's
    description in natural language.
    This method requires an OpenAI API key to be set in the environment variable OPENAI_KEY."""

    client = openai.AzureOpenAI(
        api_version="2025-03-01-preview",
        azure_endpoint=LLM_ENDPOINT,
        api_key=os.getenv("OPENAI_KEY"),
    )

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": LLM_PROMPT_SURVEY.format(
                    company_description=org_description
                ),
            }
        ],
        max_completion_tokens=1500,
        model="gpt-5-mini",
    )
    content = response.choices[0].message.content
    data = json.loads(content)  # type: ignore
    return OrgFeatures(**data)


def create_victim_feats_from_description(org_description: str) -> InputFeatures:
    """Creates an instance of VictimFeatures by querying a language model with the organisation's
    description in natural language.
    This method requires an OpenAI API key to be set in the environment variable OPENAI_KEY."""

    client = openai.AzureOpenAI(
        api_version="2025-03-01-preview",
        azure_endpoint=LLM_ENDPOINT,
        api_key=os.getenv("OPENAI_KEY"),
    )

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": LLM_PROMPT_VCDB.format(company_description=org_description),
            }
        ],
        max_completion_tokens=1500,
        model="gpt-5-mini",
    )
    content = response.choices[0].message.content
    data = json.loads(content)  # type: ignore
    return InputFeatures(**data)
