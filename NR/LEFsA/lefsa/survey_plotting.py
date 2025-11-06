# -*- coding: utf-8 -*-

import typing

import matplotlib.pyplot as plt
import pandas
import seaborn

from . import survey

COVARIATES = [
    "antall_ansatte",
    "bransje",
    "sektor",
    "samfunnskritiske_tjenester",
    "rammeverk_for_informasjonssikkerhet",
    "outsourcet",
    "landsdel",
]

COVARIATE_VALS = {
    covariate: typing.get_type_hints(survey.OrgFeatures)[covariate].__args__[0].__args__
    for covariate in COVARIATES
}


class ForecastVisualizer:
    """Visualisation tools for survey-based forecasters."""

    def __init__(self, forecaster: survey.SurveyBasedForecaster):
        seaborn.set_theme(style="whitegrid")
        self.forecaster = forecaster
        self.fig_size = (15, 6)

    def plot_all(self, feats: survey.OrgFeatures):
        yield self.plot_forecast(feats)
        yield self.plot_detailed_forecast(feats)
        yield self.plot_losses(feats)
        yield self.plot_consequences(feats)
        yield self.plot_factors(feats)
        yield self.plot_discovery(feats)

    def plot_forecast(self, feats: survey.OrgFeatures):
        title = "Forecast probabilities of at least one incident in the next 12 months"
        ylabel_left = "Prob. of >=1 $\\mathbf{major}$ incident(s), in percent"
        ylabel_right = "Prob. of >=1 incident(s) of\n given type, in percent"

        forecast_nb = self.forecaster.forecast(feats, return_type="samples")
        records_nb = [{"prob": sample} for sample in forecast_nb.get(">=1", [])]

        forecast_types = self.forecaster.forecast_by_type(
            feats, use_short_list=True, return_type="samples"
        )
        records_types = []
        for incident_type, samples in forecast_types.items():
            records_types += [
                {"Incident type": incident_type, "prob": sample}
                for sample in samples.get(">=1", [])
            ]

        fig, axes = plt.subplots(
            1, 2, figsize=self.fig_size, gridspec_kw={"width_ratios": [0.2, 0.8]}
        )
        fig.subplots_adjust(wspace=0.3)

        fig.text(
            0.5,
            axes[0].get_position().y1 + 0.08,
            title,
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            transform=fig.transFigure,
        )

        plot_prob(records_nb, axes[0], x_var=None, ylabel=ylabel_left)
        plot_prob(records_types, axes[1], x_var="Incident type", ylabel=ylabel_right)

        plt.close(fig)
        return fig

    def plot_detailed_forecast(self, feats: survey.OrgFeatures):
        title = (
            "Detailed forecasts per incident type (NB: types are not mutually exclusive,\n"
            + "may overlap,and do not always correspond to a major incident)"
        )
        ylabel = "Prob. of >=1 incident(s) of\n given type, in percent"

        forecast_types2 = self.forecaster.forecast_by_type(
            feats, use_short_list=False, return_type="samples"
        )
        records_types2 = []
        for incident_type, samples in forecast_types2.items():
            records_types2 += [
                {"Incident type": incident_type, "prob": sample}
                for sample in samples.get(">=1", [])
            ]

        fig, ax = plt.subplots(figsize=self.fig_size)

        fig.text(
            0.5,
            ax.get_position().y1 + 0.1,
            title,
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            transform=fig.transFigure,
        )
        plot_prob(records_types2, ax, x_var="Incident type", ylabel=ylabel)

        plt.close(fig)
        return fig

    def plot_losses(self, feats: survey.OrgFeatures):
        title = "Probabilities of (financial and non-financial) losses in case of major incident"
        ylabel = "Prob. of loss in case of incident, in percent"

        forecast_next_incident = self.forecaster.predict_next_incident(
            feats, return_type="samples"
        )
        fin_losses_next_incident = [
            {"Loss level": loss_level, "prob": sample}
            for loss_level, samples in forecast_next_incident["losses"].items()
            for sample in samples
        ]
        nonfin_losses_next_incident = [
            {"Loss type": loss_type, "prob": sample}
            for loss_type, samples in forecast_next_incident["other_losses"].items()
            for sample in samples
        ]

        fig, axes = plt.subplots(
            1, 2, figsize=self.fig_size, gridspec_kw={"width_ratios": [0.5, 0.5]}
        )
        fig.subplots_adjust(wspace=0.3)

        fig.text(
            0.5,
            axes[0].get_position().y1 + 0.1,
            title,
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            transform=fig.transFigure,
        )

        plot_prob(fin_losses_next_incident, axes[0], x_var="Loss level", ylabel=ylabel)
        plot_prob(
            nonfin_losses_next_incident, axes[1], x_var="Loss type", ylabel=ylabel
        )

        plt.close(fig)
        return fig

    def plot_consequences(self, feats: survey.OrgFeatures):
        title = "Probabilities of (non-mutually exclusive) consequences in case of major incident"
        ylabel = "Prob. of consequence, in percent"

        forecast_next_incident = self.forecaster.predict_next_incident(
            feats, return_type="samples"
        )

        consequences_next_incident = [
            {"Consequence": consequence, "prob": sample}
            for consequence, samples in forecast_next_incident["consequences"].items()
            for sample in samples
        ]

        fig, ax = plt.subplots(figsize=self.fig_size)
        fig.text(
            0.5,
            ax.get_position().y1 + 0.1,
            title,
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            transform=fig.transFigure,
        )

        plot_prob(consequences_next_incident, ax, x_var="Consequence", ylabel=ylabel)

        plt.close(fig)
        return fig

    def plot_factors(self, feats: survey.OrgFeatures):
        title = "Probabilities of contributing factors to a major incident"
        ylabel = "Prob. of contributing factor, in percent"

        forecast_next_incident = self.forecaster.predict_next_incident(
            feats, return_type="samples"
        )

        factors_next_incident = [
            {"Contributing factor": factor, "prob": sample}
            for factor, samples in forecast_next_incident["factors"].items()
            for sample in samples
        ]

        fig, ax = plt.subplots(figsize=self.fig_size)
        fig.text(
            0.5,
            ax.get_position().y1 + 0.1,
            title,
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            transform=fig.transFigure,
        )

        plot_prob(factors_next_incident, ax, x_var="Contributing factor", ylabel=ylabel)

        plt.close(fig)
        return fig

    def plot_discovery(self, feats: survey.OrgFeatures):
        title = "Probabilities of causes behind the discovery of an incident"
        ylabel = "Prob. of discovery cause, in percent"

        forecast_next_incident = self.forecaster.predict_next_incident(
            feats, return_type="samples"
        )

        discovery_next_incident = [
            {"Cause of discovery": discovery, "prob": sample}
            for discovery, samples in forecast_next_incident["discovery"].items()
            for sample in samples
        ]

        fig, ax = plt.subplots(figsize=self.fig_size)
        fig.text(
            0.5,
            ax.get_position().y1 + 0.1,
            title,
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            transform=fig.transFigure,
        )

        plot_prob(
            discovery_next_incident, ax, x_var="Cause of discovery", ylabel=ylabel
        )

        plt.close(fig)
        return fig

    def plot_discovery_time(self, feats: survey.OrgFeatures):
        title = "Time needed to discover the occurrence of a past incident"
        ylabel = "Probability of detection time, in percent"

        forecast_next_incident = self.forecaster.predict_next_incident(
            feats, return_type="samples"
        )

        discovery_next_incident = [
            {"Time needed to discover incident": discovery, "prob": sample}
            for discovery, samples in forecast_next_incident["discovery_time"].items()
            for sample in samples
        ]

        fig, ax = plt.subplots(figsize=self.fig_size)
        fig.text(
            0.5,
            ax.get_position().y1 + 0.1,
            title,
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            transform=fig.transFigure,
        )

        plot_prob(
            discovery_next_incident,
            ax,
            x_var="Time needed to discover incident",
            ylabel=ylabel,
        )

        plt.close(fig)
        return fig

    def plot_historical_proportions(self):
        """Plots the historical proportions of organisations that experienced at least one major incident
        during a given year, factored by covariates.
        """

        row_title = (
            "Historical proportions of organisations that experienced >=1 major incident(s) "
            + "during that year, factored by {covariate}"
        )

        survey_data = self.forecaster.survey_data

        dfs_by_covariate = {}
        for i, covariate in enumerate(COVARIATES):
            records = []
            for covariate_val in COVARIATE_VALS[covariate]:
                cond_var, cond_val = survey.OrgFeatures(
                    **{covariate: covariate_val}
                ).get_cond_assignments()[0]
                for year in survey_data.get_years():
                    if survey_data.has_condition(
                        "incident_nb",
                        ">= 1",
                        year,
                        cond_var=cond_var,
                        cond_value=cond_val,
                    ):
                        prop = survey_data.get_conditional_proportion(
                            "incident_nb",
                            ">= 1",
                            year,
                            cond_var=cond_var,
                            cond_value=cond_val,
                        )
                        records.append(
                            {
                                "Proportion": prop,
                                "year": int(year),
                                covariate: covariate_val,
                            }
                        )
                    else:
                        pass
            #          print("could not find condition for", year, cond_var, cond_val)

            df = pandas.DataFrame(records)

            # Skipping cases where we have less than two years of data
            if df["year"].nunique() <= 1:
                continue
            df.Proportion = df.Proportion * 100
            dfs_by_covariate[covariate] = df

        fig, axes = plt.subplots(
            len(dfs_by_covariate), 1, figsize=(12, 6 * len(dfs_by_covariate))
        )
        fig.subplots_adjust(hspace=0.4)

        for i, (covariate, df) in enumerate(dfs_by_covariate.items()):
            # Plot one regression line per covariate value
            for covariate_val in df[covariate].unique():
                df_sub = df[df[covariate] == covariate_val]
                seaborn.regplot(
                    data=df_sub,
                    x="year",
                    y="Proportion",
                    scatter=True,
                    ax=axes[i],
                    ci=95,
                    truncate=False,
                    label=str(covariate_val),
                )

            axes[i].set_title(
                row_title.format(covariate=covariate), fontsize=14, fontweight="bold"
            )
            axes[i].set_ylim(bottom=0)
            axes[i].legend(loc="upper left", bbox_to_anchor=(1, 1), title=covariate)
            axes[i].set_ylabel(
                "Proportion of orgs with >=1\n incidents that year, in percent"
            )
        plt.close(fig)
        return fig

    def plot_all_by_covariate(self):
        """Plots all the forecasts, losses, consequences, factors and discovery causes factored by covariates."""

        yield self.plot_forecast_by_covariate()
        yield self.plot_detailed_forecast_by_covariate()
        yield self.plot_losses_by_covariate()
        yield self.plot_consequences_by_covariate()
        yield self.plot_factors_by_covariate()
        yield self.plot_discovery_by_covariate()

    def plot_forecast_by_covariate(self):
        row_title = "Forecast probabilities of at least one incident during the next 12 months, factored by {covariate}"
        ylabel_left = "Prob. of >=1 major incident(s),\n in percent"
        ylabel_right = "Prob. of >=1 incident(s) of given type,\n in percent"

        fig, axes = plt.subplots(
            len(COVARIATES),
            2,
            figsize=(24, 7 * len(COVARIATES)),
            gridspec_kw={"width_ratios": [0.3, 0.7]},
        )
        fig.subplots_adjust(wspace=0.2, hspace=1.0)

        for i, covariate in enumerate(COVARIATES):
            # Add subtitle above each row
            fig.text(
                0.5,
                axes[i, 0].get_position().y1 + 0.01,
                row_title.format(covariate=covariate),
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
                transform=fig.transFigure,
            )

            records_nb, records_types = [], []
            for v in COVARIATE_VALS[covariate]:
                feats = survey.OrgFeatures(**{covariate: v})
                forecast_nb = self.forecaster.forecast(feats, return_type="samples")
                records_nb += [
                    {"prob": sample, covariate: v}
                    for sample in forecast_nb.get(">=1", [])
                ]

                forecast_types = self.forecaster.forecast_by_type(
                    feats, use_short_list=True, return_type="samples"
                )
                for incident_type, samples in forecast_types.items():
                    records_types += [
                        {"Incident type": incident_type, "prob": sample, covariate: v}
                        for sample in samples.get(">=1", [])
                    ]

            plot_prob(
                records_nb,
                axes[i, 0],
                x_var=covariate,
                hue_var=None,
                ylabel=ylabel_left,
            )
            plot_prob(
                records_types,
                axes[i, 1],
                x_var="Incident type",
                hue_var=covariate,
                ylabel=ylabel_right,
            )

        plt.close(fig)
        return fig

    def plot_detailed_forecast_by_covariate(self):
        row_title = (
            "Detailed forecasts per incident type, factored by {covariate} (NB: types are not "
            + "mutually exclusive, may overlap,\nand do not always correspond to a major incident)"
        )

        ylabel = "Prob. of >=1 incident(s) of given type, in percent"

        fig, axes = plt.subplots(len(COVARIATES), 1, figsize=(24, 12 * len(COVARIATES)))
        fig.subplots_adjust(hspace=1.3)

        for i, covariate in enumerate(COVARIATES):
            # Add subtitle above each row
            fig.text(
                0.5,
                axes[i].get_position().y1 + 0.005,
                row_title.format(covariate=covariate),
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
                transform=fig.transFigure,
            )

            records = []
            for v in COVARIATE_VALS[covariate]:
                org_features = survey.OrgFeatures(**{covariate: v})
                forecast_types2 = self.forecaster.forecast_by_type(
                    org_features, use_short_list=False, return_type="samples"
                )
                for incident_type, samples in forecast_types2.items():
                    records += [
                        {"Incident type": incident_type, "prob": sample, covariate: v}
                        for sample in samples.get(">=1", [])
                    ]

            plot_prob(
                records, axes[i], "Incident type", hue_var=covariate, ylabel=ylabel
            )

        plt.close(fig)
        return fig

    def plot_losses_by_covariate(self):
        row_title = "Probabilities of (financial and non-financial) losses in case of major incident, factored by {covariate}"
        ylabel_left = "Prob. of loss in case of incident,\n in percent"
        ylabel_right = "Prob. of loss in case of incident,\n in percent"

        fig, axes = plt.subplots(
            len(COVARIATES),
            2,
            figsize=(24, 7 * len(COVARIATES)),
            gridspec_kw={"width_ratios": [0.5, 0.5]},
        )
        fig.subplots_adjust(wspace=0.2, hspace=1.0)

        for i, covariate in enumerate(COVARIATES):
            # Add subtitle above each row
            fig.text(
                0.5,
                axes[i, 0].get_position().y1 + 0.01,
                row_title.format(covariate=covariate),
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
                transform=fig.transFigure,
            )

            fin_losses, nonfin_losses = [], []
            for v in COVARIATE_VALS[covariate]:
                feats = survey.OrgFeatures(**{covariate: v})
                predictions = self.forecaster.predict_next_incident(
                    feats, return_type="samples"
                )
                fin_losses += [
                    {"Loss level": loss_level, "prob": sample, covariate: v}
                    for loss_level, samples in predictions["losses"].items()
                    for sample in samples
                ]
                nonfin_losses += [
                    {"Loss type": loss_type, "prob": sample, covariate: v}
                    for loss_type, samples in predictions["other_losses"].items()
                    for sample in samples
                ]

            plot_prob(
                fin_losses,
                axes[i, 0],
                x_var="Loss level",
                hue_var=covariate,
                ylabel=ylabel_left,
                skip_legend=True,
            )
            plot_prob(
                nonfin_losses,
                axes[i, 1],
                x_var="Loss type",
                hue_var=covariate,
                ylabel=ylabel_right,
            )

        plt.close(fig)
        return fig

    def plot_consequences_by_covariate(self):
        row_title = (
            "Probabilities of (non-mutually exclusive) consequences in case of major incident, "
            + "factored by {covariate}"
        )

        ylabel = "Prob. of consequence, in percent"

        fig, axes = plt.subplots(len(COVARIATES), 1, figsize=(24, 9 * len(COVARIATES)))
        fig.subplots_adjust(hspace=1.1)

        for i, covariate in enumerate(COVARIATES):
            # Add subtitle above each row
            fig.text(
                0.5,
                axes[i].get_position().y1 + 0.005,
                row_title.format(covariate=covariate),
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
                transform=fig.transFigure,
            )

            records = []
            for v in COVARIATE_VALS[covariate]:
                org_features = survey.OrgFeatures(**{covariate: v})
                predictions = self.forecaster.predict_next_incident(
                    org_features, return_type="samples"
                )
                for consequence, samples in predictions["consequences"].items():
                    records += [
                        {"Consequence": consequence, "prob": sample, covariate: v}
                        for sample in samples
                    ]

            plot_prob(records, axes[i], "Consequence", hue_var=covariate, ylabel=ylabel)

        plt.close(fig)
        return fig

    def plot_factors_by_covariate(self):
        row_title = "Probabilities of contributing factors to a major incident, factored by {covariate}"

        ylabel = "Prob. of contributing factor, in percent"

        fig, axes = plt.subplots(len(COVARIATES), 1, figsize=(24, 11 * len(COVARIATES)))
        fig.subplots_adjust(hspace=1.7)

        for i, covariate in enumerate(COVARIATES):
            # Add subtitle above each row
            fig.text(
                0.5,
                axes[i].get_position().y1 + 0.005,
                row_title.format(covariate=covariate),
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
                transform=fig.transFigure,
            )

            records = []
            for v in COVARIATE_VALS[covariate]:
                org_features = survey.OrgFeatures(**{covariate: v})
                predictions = self.forecaster.predict_next_incident(
                    org_features, return_type="samples"
                )
                for factor, samples in predictions["factors"].items():
                    records += [
                        {"Contributing factor": factor, "prob": sample, covariate: v}
                        for sample in samples
                    ]

            plot_prob(
                records,
                axes[i],
                "Contributing factor",
                hue_var=covariate,
                ylabel=ylabel,
            )

        plt.close(fig)
        return fig

    def plot_discovery_by_covariate(self):
        row_title = "Probabilities of causes behind the discovery of an incident, factored by {covariate}"
        ylabel = "Prob. of discovery cause, in percent"

        fig, axes = plt.subplots(len(COVARIATES), 1, figsize=(24, 9 * len(COVARIATES)))
        fig.subplots_adjust(hspace=1.1)

        for i, covariate in enumerate(COVARIATES):
            # Add subtitle above each row
            fig.text(
                0.5,
                axes[i].get_position().y1 + 0.005,
                row_title.format(covariate=covariate),
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
                transform=fig.transFigure,
            )

            records = []
            for v in COVARIATE_VALS[covariate]:
                org_features = survey.OrgFeatures(**{covariate: v})
                predictions = self.forecaster.predict_next_incident(
                    org_features, return_type="samples"
                )
                for discovery, samples in predictions["discovery"].items():
                    records += [
                        {"Cause of discovery": discovery, "prob": sample, covariate: v}
                        for sample in samples
                    ]

            plot_prob(
                records, axes[i], "Cause of discovery", hue_var=covariate, ylabel=ylabel
            )

        plt.close(fig)
        return fig


def plot_prob(
    records,
    ax,
    x_var=None,
    hue_var=None,
    ylabel=None,
    max_value=None,
    conf_level=0.95,
    use_percentages=True,
    skip_legend=False,
):
    """Plots a boxplot of the probabilities in the records, optionally grouped by x_var and hue_var.
    Args:
        records (list of dict): Each dict should contain a "prob" key with the probability value.
        ax (matplotlib.axes.Axes): The axes to plot on.
        x_var (str, optional): The variable to group by on the x-axis. Defaults to None.
        hue_var (str, optional): The variable to group by for coloring. Defaults to None.
        ylabel (str, optional): The label for the y-axis. Defaults to None.
        max_value (float, optional): Maximum value for the y-axis. Defaults to None.
        conf_level (float, optional): Confidence level for the boxplot whiskers. Defaults to 0.95.
        use_percentages (bool, optional): Whether to convert probabilities to percentages. Defaults to True.
        skip_legend (bool, optional): Whether to skip displaying the legend. Defaults to False.
    """

    df = pandas.DataFrame(records)

    # Specifying that the tails of the boxplot should be at the 2.5% and 97.5% percentiles
    whis = (1 - conf_level) / 2, (1 + conf_level) / 2

    if use_percentages:
        df["prob"] = df["prob"].astype(float) * 100
        whis = (whis[0] * 100, whis[1] * 100)

    seaborn.boxplot(
        data=df,
        x=x_var,
        y="prob",
        whis=whis,  # type: ignore
        showfliers=False,
        ax=ax,
        hue=hue_var,
    )

    # We add at the top of each boxplot the mean value
    if x_var:
        means = df.groupby(x_var)["prob"].mean()
        means = means.reindex(df[x_var].drop_duplicates())
    else:
        means = {None: df["prob"].mean()}
    for i, (_, mean_val) in enumerate(means.items()):
        ax.text(
            i,
            ax.get_ylim()[1],
            f"{mean_val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            color="darkgreen",
        )

    # Add "Mean: " label at the top of the y-axis
    ax.text(
        0,
        ax.get_ylim()[1],
        "Means: ",
        ha="right",
        va="bottom",
        fontsize=11,
        color="darkgreen",
        transform=ax.get_yaxis_transform(),
    )

    ax.set_ylabel(ylabel)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylim(bottom=-0.001, top=max_value if max_value else None)
    if hue_var:
        if not skip_legend:
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1), title=hue_var)
        else:
            ax.get_legend().remove()
            ax.get_legend().remove()
