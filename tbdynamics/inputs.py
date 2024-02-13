from pathlib import Path
import pandas as pd

BASE_PATH = Path(__file__).parent.parent.resolve()
DATA_PATH = BASE_PATH / "data"

fixed_parameters = {
    "acf_screening_sensitivity": 0.75,
    "age_infectiousness_switch": 15.0,
    "age_mixing": {"age_adjust": True, "source_iso3": "VNM"},
    "age_latency": {
        "early_activation": {0: 2.4107, 5: 0.9862, 15: 0.0986},
        "late_activation": {0: 6.939769e-09, 5: 0.0023, 15: 0.0012},
        "stabilisation": {0: 4.383, 5: 4.383, 15: 1.972},
    },
    "bcg_effect": "infection",
    "calculated_outputs": ["prevalence_infectious"],
    "cdr_adjustment": 0.8003452159878636,
    "contact_rate": 0.013414102898074345,
    "country": {"country_name": "Vietnam", "iso3": "VNM"},
    "crude_birth_rate": 0.2,
    "crude_death_rate": 0.0008,
    "cumulative_output_start_time": 2020.0,
    "cumulative_start_time": 1990.0,
    "description": "BASELINE",
    "extrapulmonary_infect_multiplier": 0.0,
    "future_diabetes_multiplier": 1.0,
    "gender": {
        "adjustments": {
            "detection": {"female": 1.0, "male": 1.5},
            "infection": {"female": 1.0, "male": 2.6408657914674176},
        },
        "proportions": {"female": 0.5, "male": 0.5},
        "strata": ["male", "female"],
    },
    "incidence_props_pulmonary": 0.85,
    "incidence_props_smear_positive_among_pulmonary": 0.75,
    "infect_death_rate_dict": {
        "smear_negative": 0.025,
        "smear_positive": 0.389,
        "unstratified": 0.2,
    },
    "infectious_seed": 1.0,
    "inflate_reactivation_for_diabetes": False,
    "on_treatment_infect_multiplier": 0.08,
    "outputs_stratification": {},
    "progression_multiplier": 1.1,
    "prop_death_among_negative_tx_outcome": 0.2,
    "prop_diabetes": {0: 0.01, 5: 0.05, 15: 0.2, 35: 0.4, 50: 0.7, 70: 0.8},
    "rr_infection_latent": 0.20278196465900813,
    "rr_infection_recovered": 0.21190687223342505,
    "rr_progression_diabetes": 5.643402828077587,
    "self_recovery_rate_dict": {
        "smear_negative": 0.22723824998716693,
        "smear_positive": 0.20344728302826143,
        "unstratified": 0.2,
    },
    "smear_negative_infect_multiplier": 0.25,
    "start_population_size": 267252.06827576435,
    "stratify_by": ["age", "organ", "gender"],
    "time": {"end": 2020, "start": 1900, "step": 0.1},
    "time_variant_bcg_perc": {
        1981: 0.1,
        1990: 49.0,
        1991: 71.0,
        1992: 72.0,
        1993: 88.0,
        1994: 96.0,
        1995: 71.0,
        1996: 98.0,
        1997: 94.0,
        1998: 81.0,
        1999: 81.0,
        2000: 89.0,
        2001: 99.0,
        2002: 90.0,
        2003: 93.0,
        2004: 91.0,
        2005: 93.0,
        2006: 92.0,
        2007: 92.0,
        2008: 95.0,
        2009: 98.0,
        2010: 99.0,
        2011: 80.0,
        2012: 97.0,
        2013: 93.0,
        2014: 89.0,
        2015: 99.0,
        2016: 94.0,
        2017: 92.0,
        2018: 98.0,
        2019: 89.0,
    },
    "time_variant_screening_rate": {
        2017: 0.0,
        2019: 1.0,
    },
    # 'time_variant_tb_screening_rate': { 1986.0: 0.13,
    #                                     1987.0: 0.34,
    #                                     1988.0: 0.44,
    #                                     1989.0: 0.5,
    #                                     1990.0: 0.56,
    #                                     1991.0: 0.63,
    #                                     1992.0: 0.67,
    #                                     1993.0: 0.61,
    #                                     1994.0: 0.59,
    #                                     1995.0: 0.61,
    #                                     1996.0: 0.8,
    #                                     1997.0: 0.8,
    #                                     1998.0: 0.81,
    #                                     1999.0: 0.83,
    #                                     2000.0: 0.87},
    "time_variant_tsr": {
        1986: 0.4,
        2000: 0.92,
        2001: 0.93,
        2002: 0.92,
        2003: 0.93,
        2004: 0.93,
        2005: 0.93,
        2006: 0.93,
        2007: 0.91,
        2008: 0.92,
        2009: 0.92,
        2010: 0.92,
        2011: 0.93,
        2012: 0.91,
        2013: 0.89,
        2014: 0.91,
        2015: 0.92,
        2016: 0.92,
        2017: 0.92,
        2018: 0.91,
        2019: 0.91,
    },
    "treatment_duration": 0.5,
    "passive_screening_sensitivity_extrapulmonary": 0.5,
    "passive_screening_sensitivity_smear_negative": 0.7,
    "passive_screening_sensitivity_smear_positive": 1.0,
    "acf_scaleup_shape": 0.05,
    "acf_inflection_time": 1990,
    "acf_start_asymp": 0.0,
    "acf_end_asymp": 10.0,
}


def get_birth_rate():
    return pd.read_csv(Path(DATA_PATH / "vn_birth.csv"), index_col=0)["value"]


def process_death_rate(age_strata: list):
    data = pd.read_csv(
        Path(DATA_PATH / "data.csv"), usecols=["Age", "Time", "Population", "Deaths"]
    )
    data = data.set_index(["Age", "Time"])
    birth_rates = get_birth_rate()
    data.index = data.index.swaplevel()
    age_groups = set(data.index.get_level_values(1))
    years = set(data.index.get_level_values(0))

    # Creating the new list
    agegroup_request = [
        [start, end - 1] for start, end in zip(age_strata, age_strata[1:] + [201])
    ]
    # agegroup_request = [[0, 4], [5, 14], [15, 34], [35, 49], [50, 69], [70, 200]]
    agegroup_map = {
        low: get_age_groups_in_range(age_groups, low, up)
        for low, up in agegroup_request
    }
    agegroup_map[agegroup_request[-1][0]].append("100+")
    mapped_rates = pd.DataFrame()
    for year in years:
        for agegroup in agegroup_map:
            age_mask = [
                i in agegroup_map[agegroup] for i in data.index.get_level_values(1)
            ]
            age_year_data = data.loc[age_mask].loc[year, :]
            total = age_year_data.sum()
            mapped_rates.loc[year, agegroup] = total["Deaths"] / total["Population"]
    mapped_rates.index = mapped_rates.index + 0.5
    death_df = mapped_rates.loc[birth_rates.index]
    return death_df


def get_age_groups_in_range(age_groups, lower_limit, upper_limit):
    return [
        i
        for i in age_groups
        if "+" not in i and lower_limit <= int(i.split("-")[0]) <= upper_limit
    ]

