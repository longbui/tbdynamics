from pathlib import Path
import pandas as pd
import yaml


def load_param_info(
    data_path: Path, 
    parameters: dict,
) -> pd.DataFrame:
    """
    Load specific parameter information from 
    a ridigly formatted yaml file or crash otherwise.

    Args:
        data_path: Location of the source file
        parameters: The parameters provided by the user (with their values)

    Returns:
        The parameters info DataFrame contains the following fields:
            descriptions: A brief reader-digestible name/description for the parameter
            units: The unit of measurement for the quantity (empty string if dimensionless)
            evidence: TeX-formatted full description of the evidence underpinning the choice of value
            manual_values: The values provided in the parameters argument
    """
    data_cols = ["descriptions", "units", "evidence"]
    param_keys = parameters.keys()
    out_df = pd.DataFrame(index=param_keys, columns=data_cols)
    with open(data_path, "r") as param_file:
        all_data = yaml.safe_load(param_file)
        for col in data_cols:
            working_data = all_data[col]
            if param_keys != working_data.keys():
                raise ValueError("Incorrect keys for data")
            out_df[col] = out_df.index.map(working_data)
        out_df["manual_values"] = parameters.values()
    return out_df
