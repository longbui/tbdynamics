from pathlib import Path
import pandas as pd
import yaml
import numpy as np
from typing import List

BASE_PATH = Path(__file__).parent.parent.resolve()
DATA_PATH = BASE_PATH / "data"
INPUT_PATH = DATA_PATH / "inputs"
DOCS_PATH = BASE_PATH / "docs"


def get_birth_rate():
    return pd.read_csv(Path(INPUT_PATH / "vn_birth.csv"), index_col=0)["value"]


def get_death_rate():
    return pd.read_csv(
        Path(INPUT_PATH / "vn_cdr.csv"), usecols=["Age", "Time", "Population", "Deaths"]
    ).set_index(["Time", "Age"])


def get_immigration():
    series = pd.read_csv(Path(INPUT_PATH / "immi.csv"), index_col=0)["value"]
    return series.astype(np.float64)


def process_death_rate(
    data: pd.DataFrame,
    age_strata: List[int], 
    year_indices: List[int],
) -> pd.DataFrame:
    """
    Processes mortality data to compute age-stratified death rates for specific years.

    This function takes a dataset containing mortality and population data, along with
    definitions for age strata and specific years of interest, to compute the death rate
    within each age stratum for those years. The death rates are calculated as the total
    deaths divided by the total population within each age stratum for each year. The
    function also adjusts age groups to include an "100+" category and handles the mapping
    of raw age groups to the defined age strata.

    Args:
        data: A pandas DataFrame indexed by (year, age_group) with at least
            two columns: 'Deaths' and 'Population', representing the total deaths and total
            population for each age group in each year, respectively.
        age_strata: A list of integers representing the starting age of each
            age stratum to be considered. The list must be sorted in ascending order.
        year_indices: A list of integers representing the years of interest
            for which the death rates are to be calculated.

    Returns:
        A pandas DataFrame indexed by the mid-point year values (year + 0.5)
            with columns for each age stratum defined in `age_strata`. Each cell contains the
            death rate for that age stratum and year.
    """
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


def load_params() -> dict:
    """
    Loads a YAML file and returns its contents as a Python dictionary.

    Args:
        file_path (str): The path to the YAML file to be read.

    Returns:
        dict: The contents of the YAML file as a Python dictionary.
    """
    with open(Path(__file__).resolve().parent / "params.yml", "r") as file:
        # Load the YAML content
        data = yaml.safe_load(file)
    return data


def load_targets():
    with open(Path(__file__).resolve().parent / "targets.yml", "r") as file:
        data = yaml.safe_load(file)

    processed_targets = {}

    for key, value in data.items():
        if isinstance(value, dict):
            # Check if the value for each key is a list of three items
            if all(isinstance(v, list) and len(v) == 3 for v in value.values()):
                # Handle as [target, lower_bound, upper_bound]
                target = pd.Series({k: v[0] for k, v in value.items()})
                lower_bound = pd.Series({k: v[1] for k, v in value.items()})
                upper_bound = pd.Series({k: v[2] for k, v in value.items()})

                processed_targets[f'{key}_target'] = target
                processed_targets[f'{key}_lower_bound'] = lower_bound
                processed_targets[f'{key}_upper_bound'] = upper_bound
            else:
                # Handle as single values
                processed_targets[key] = pd.Series(value)
        else:
            # Handle cases where value is not a dictionary
            processed_targets[key] = pd.Series(value)

    return processed_targets


values = [
    [398.97659525, 320.21837369, 724.81664047, 365.25, 194.35563715, 17.99901401],
    [
        166.01590217,
        1149.10619577,
        637.53986955,
        500.88147148,
        146.39163322,
        34.20477279,
    ],
    [
        232.06788972,
        395.42873828,
        1112.26496082,
        612.33223254,
        359.44845154,
        39.81828055,
    ],
    [142.13862133, 400.1791475, 935.13490555, 1031.58445946, 492.1288911, 88.65151431],
    [84.33487849, 206.32216764, 739.24869454, 938.24515192, 943.6458889, 197.05799339],
    [28.36639153, 251.69370346, 680.41210853, 899.59722222, 781.86007839, 307.12603272],
] #unadjusted contact matrix

conmat_values = [
    [
        1309.98923567,
        667.6194366,
        1054.38132113,
        949.00683127,
        366.05677516,
        30.98610428,
    ],
    [
        366.14966919,
        4492.85151982,
        1172.2432921,
        962.26655316,
        406.40120902,
        52.54418198,
    ],
    [
        279.31082424,
        566.21079106,
        3287.61902669,
        1213.13979038,
        720.90696077,
        49.03653283,
    ],
    [
        347.97888347,
        643.35290596,
        1679.20738837,
        1823.40872755,
        808.19633429,
        93.16298799,
    ],
    [
        157.1258821,
        318.07138435,
        1168.12264022,
        946.0903429,
        1016.19884585,
        130.88451047,
    ],
    [48.81598392, 150.93501514, 291.62499058, 400.27172231, 480.37900611, 170.9479575],
]

matrix = np.array(values)
conmat = np.array(conmat_values)
