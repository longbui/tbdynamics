from pathlib import Path
import pandas as pd
import yaml
import numpy as np

BASE_PATH = Path(__file__).parent.parent.resolve()
DATA_PATH = BASE_PATH / "data"


def get_birth_rate():
    return pd.read_csv(Path(DATA_PATH / "vn_birth.csv"), index_col=0)["value"]


def get_death_rate():
    return pd.read_csv(
        Path(DATA_PATH / "vn_cdr.csv"), usecols=["Age", "Time", "Population", "Deaths"]
    ).set_index(["Time", "Age"])


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


def load_params(file_path: str) -> dict:
    """
    Loads a YAML file and returns its contents as a Python dictionary.

    Args:
        file_path (str): The path to the YAML file to be read.

    Returns:
        dict: The contents of the YAML file as a Python dictionary.
    """
    with open(file_path, "r") as file:
        # Load the YAML content
        data = yaml.safe_load(file)
    return data


values = [
    [369.1558, 375.7934, 977.4291, 371.6857, 175.9497, 22.9686],
    [157.4260, 1342.0096, 569.3469, 462.2655, 115.2442, 61.9570],
    [209.6730, 291.5459, 1110.4988, 525.5363, 270.5620, 70.5169],
    [150.0992, 445.6222, 989.3461, 1013.7379, 465.6254, 168.9860],
    [129.0962, 201.8443, 925.4105, 845.9774, 916.9300, 290.6150],
    [39.3506, 253.3840, 563.1870, 716.9094, 678.5934, 442.0020]
]

conmat_values = [[1272.56210834,  663.65309188, 1058.34574098,  954.06659935,  365.09929785,   31.83545663],
 [ 351.50607715, 4498.22774414, 1184.06043355,  962.36254451,  403.80391627,   53.82135531],
 [ 268.37491782,  566.88701492, 3310.79767115, 1212.31920512,  719.71866008,   49.90772713],
 [ 337.48014112,  642.71217346, 1691.11117807, 1824.44079987,  807.53123596,   94.63373651],
 [ 151.18466351,  315.7007    , 1175.29028431,  945.33674084, 1009.98512154,  134.93938447],
 [  46.80041023,  149.38310806,  289.32902908,  393.29193361,  479.04968239,  171.3871341 ]]

matrix = np.array(values)
conmat = np.array(conmat_values)
