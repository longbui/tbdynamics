from pathlib import Path
import pandas as pd
import yaml

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


matrix = [
    [
        368.223499932989,
        270.84359260056294,
        754.2616385469169,
        324.368948293599,
        174.766699217585,
        33.87835355437299,
    ],
    [
        203.11794607160599,
        1332.278342064571,
        481.375252621506,
        492.97034175629295,
        120.03151569748,
        73.040532171281,
    ],
    [
        258.15610091319894,
        336.859266533633,
        1107.030817506147,
        585.008626342693,
        336.845209118396,
        61.06830421219699,
    ],
    [
        168.535801322863,
        416.77218364367695,
        876.563810866599,
        1013.8124858329629,
        414.722937064161,
        149.360795884703,
    ],
    [
        128.640739269435,
        195.975903624563,
        699.6884920584439,
        932.9666157638899,
        917.0587228110719,
        280.17417374848196,
    ],
    [
        21.716396693841,
        209.66661416258296,
        634.099829051713,
        797.884775918731,
        710.3597383246439,
        441.38141281813995,
    ],
]
