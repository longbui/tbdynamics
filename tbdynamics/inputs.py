from pathlib import Path
import pandas as pd

BASE_PATH = Path(__file__).parent.parent.resolve()
DATA_PATH = BASE_PATH / "data"


def get_birth_rate():
    return pd.read_csv(Path(DATA_PATH / "vn_birth.csv"), index_col=0)["value"]

def get_death_rate():
    return pd.read_csv(Path(DATA_PATH / "vn_cdr.csv"), usecols=["Age", "Time", "Population", "Deaths"]).set_index(["Time", "Age"])


def process_death_rate(data, age_strata, year_indices):
    years = set(data.index.get_level_values(0))
    age_groups = set(data.index.get_level_values(1))

    # Creating the new list
    agegroup_request = [
        [start, end - 1] for start, end in zip(age_strata, age_strata[1:] + [201])
    ]
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
    mapped_rates.index += 0.5
    death_df = mapped_rates.loc[year_indices]
    return death_df


def get_age_groups_in_range(age_groups, lower_limit, upper_limit):
    return [
        i
        for i in age_groups
        if "+" not in i and lower_limit <= int(i.split("-")[0]) <= upper_limit
    ]


fixed_parameters = {
    "age_latency": {
        "early_activation": {0: 2.4107, 5: 0.9862, 15: 0.0986},
        "late_activation": {0: 6.939769e-09, 5: 0.0023, 15: 0.0012},
        "stabilisation": {0: 4.383, 5: 4.383, 15: 1.972},
    },
    "age_infectiousness_switch": 15.0,
    "smear_negative_infect_multiplier": .25,
    "extrapulmonary_infect_multiplier": 0.,
    "incidence_props_pulmonary": 0.85,
    "incidence_props_smear_positive_among_pulmonary": 0.75,
    "prop_death_among_negative_tx_outcome": 0.2,
    "passive_screening_sensitivity_extrapulmonary": 0.5,
    "passive_screening_sensitivity_smear_negative": 0.7,
    "passive_screening_sensitivity_smear_positive": 1.0,
}
