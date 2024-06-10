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

latent_compartments = compartments[1: 3]
infectious_compartments = compartments[3: 5]

bcg_multiplier_dict = {
    "0": 0.3,
    "5": 0.3,
    "15": 0.7375,
    "35": 1.0,
    "50": 1.0,
    "70": 1.0,
}


BURN_IN = 50000
OPTI_DRAWS = 100

PLOT_START_DATE = 1800
PLOT_END_DATE = 2023

indicator_names = {
        'total_population': 'Total Population',
        'notification': 'Notification',
        'incidence': 'Incidence (per 100,000)',
        'percentage_latent': 'Percentage Latent (%)',
        'prevalence_pulmonary': 'Prevalence Pulmonary (per 100,000)',
        'cdr': 'Case detection rate'
    }