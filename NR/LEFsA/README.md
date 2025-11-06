# Forecasting model for loss events
[Pierre Lison](mailto:plison@nr.no), November 6, 2025

The goal of this implementation is to automatically forecast the *number* and *types* of loss events (i.e. security-related incidents) that may occur in a given organization $O$ over a specified time horizon $[t, t+H]$. 

We have developed two complementary approaches to generate those forecasts based on Open Source Intelligence (OSINT) data:
- One statistical approach based on results from the [Mørketallsundersøkelse](https://www.nsr-org.no/publikasjoner/morketallsundersokelsen) surveys. This approach gives us a high-level overview of the expected number and types of loss events depending on the characteristics of the organization.
- One machine learning approach based on incidents reported in the [Veris Community Database](https://verisframework.org/vcdb.html) (VCDB). This approach allows us to generate more fine-grained forecasts about the exact nature of the incident (attack vector, affected assets, impact, etc.), again based on the characteristics of the organization.

We have prepared two notebooks to explain how those two approaches operate and how to use them in practice:
- [Walkthrough on survey-based forecasting](examples/walkthrough_survey.ipynb)
- [Walkthrough on VCDB-based forecasting](examples/walkthrough_vcdb.ipynb)

Don't hesitate to get in touch if you have any questions or comments. 