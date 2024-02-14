from pathlib import Path
import pandas as pd

BASE_PATH = Path(__file__).parent.parent.resolve()
DATA_PATH = BASE_PATH / "data"

fixed_parameters = {
}

def get_birth_rate():
    return pd.read_csv(Path(DATA_PATH / "vn_birth.csv"), index_col=0)["value"]


def process_death_rate(age_strata: list):
    data = pd.read_csv(
        Path(DATA_PATH / "vn_cdr.csv"), usecols=["Age", "Time", "Population", "Deaths"]
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
