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

def get_immigration():
    series=   pd.read_csv(Path(DATA_PATH / "immi.csv"), index_col= 0)["value"]
    return series.astype(np.float64)


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


def load_params() -> dict:
    """
    Loads a YAML file and returns its contents as a Python dictionary.

    Args:
        file_path (str): The path to the YAML file to be read.

    Returns:
        dict: The contents of the YAML file as a Python dictionary.
    """
    with open(Path(__file__).resolve().parent / 'params.yml', "r") as file:
        # Load the YAML content
        data = yaml.safe_load(file)
    return data

def load_targets():
    with open(Path(__file__).resolve().parent / 'targets.yml', 'r') as file:
        data = yaml.safe_load(file)

    # Convert the loaded YAML data to a Pandas Series (assuming the data structure allows for it)
    # This example assumes the YAML file contains a dictionary at its root
    return {key: pd.Series(value) for key, value in data.items()}


values = [
    [369.1558, 375.7934, 977.4291, 371.6857, 175.9497, 22.9686],
    [157.4260, 1342.0096, 569.3469, 462.2655, 115.2442, 61.9570],
    [209.6730, 291.5459, 1110.4988, 525.5363, 270.5620, 70.5169],
    [150.0992, 445.6222, 989.3461, 1013.7379, 465.6254, 168.9860],
    [129.0962, 201.8443, 925.4105, 845.9774, 916.9300, 290.6150],
    [39.3506, 253.3840, 563.1870, 716.9094, 678.5934, 442.0020]
]

conmat_values = [[1309.98923567,  667.6194366 , 1054.38132113,  949.00683127,
         366.05677516,   30.98610428],
       [ 366.14966919, 4492.85151982, 1172.2432921 ,  962.26655316,
         406.40120902,   52.54418198],
       [ 279.31082424,  566.21079106, 3287.61902669, 1213.13979038,
         720.90696077,   49.03653283],
       [ 347.97888347,  643.35290596, 1679.20738837, 1823.40872755,
         808.19633429,   93.16298799],
       [ 157.1258821 ,  318.07138435, 1168.12264022,  946.0903429 ,
        1016.19884585,  130.88451047],
       [  48.81598392,  150.93501514,  291.62499058,  400.27172231,
         480.37900611,  170.9479575 ]]

matrix = np.array(values)
conmat = np.array(conmat_values)
