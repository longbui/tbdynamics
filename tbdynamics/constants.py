from pathlib import Path
organ_strata = [
    "smear_positive",
    "smear_negative",
    "extrapulmonary",
]
age_strata = [0, 5, 15, 35, 50, 70]

compartments = [
    "susceptible",
    "early_latent",
    "late_latent",
    "infectious",
    "on_treatment",
    "recovered",
]

latent_compartments = [
    "early_latent",
    "late_latent",
]
infectious_compartments = [
    "infectious",
    "on_treatment",
]

bcg_multiplier_dict = {
    "0": 0.3,
    "5": 0.3,
    "15": 0.7375,
    "35": 1.0,
    "50": 1.0,
    "70": 1.0,
}

PROJECT_PATH = Path(__file__).parent.parent.resolve()
DATA_PATH = PROJECT_PATH / 'data'
RUNS_PATH = PROJECT_PATH / 'runs'
OUTPUTS_PATH = PROJECT_PATH / 'outputs'

BURN_IN = 25000
OPTI_DRAWS = 100
