import pandas as pd
import yaml as yml
from inputs.constants import INPUTS_PATH


def capture_kwargs(*args, **kwargs):
    return kwargs


def load_param_info() -> pd.DataFrame:
    """
    Load specific parameter information from a ridigly formatted yaml file, and crash otherwise.

    Args:
        data_path: Location of the source file
        parameters: The parameters provided by the user (with their values)

    Returns:
        The parameters info DataFrame contains the following fields:
            value: Enough parameter values to ensure model runs, may be over-written in calibration
            descriptions: A brief reader-digestible name/description for the parameter
            units: The unit of measurement for the quantity (empty string if dimensionless)
            evidence: TeX-formatted full description of the evidence underpinning the choice of value
            abbreviations: Short name for parameters, e.g. for some plots
    """
    with open(INPUTS_PATH / 'parameters.yml', 'r') as param_file:
        param_info = yml.safe_load(param_file)

    # Check each loaded set of keys (parameter names) are the same as the arbitrarily chosen first key
    first_key_set = param_info[list(param_info.keys())[0]].keys()
    for cat in param_info:
        working_keys = param_info[cat].keys()
        if working_keys != first_key_set:
            msg = f'Keys to {cat} category: {working_keys} - do not match first category {first_key_set}'
            raise ValueError(msg)
    
    return pd.DataFrame(param_info)