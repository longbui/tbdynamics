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
