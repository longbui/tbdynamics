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

latent_compartments = compartments[1:3]
infectious_compartments = compartments[3:5]

bcg_multiplier_dict = {
    "0": 0.3,
    "5": 0.3,
    "15": 0.7375,
    "35": 1.0,
    "50": 1.0,
    "70": 1.0,
}

PLOT_START_DATE = 1800
PLOT_END_DATE = 2035

indicator_names = {
    "total_population": "Total population",
    "notification": "TB notifications",
    "incidence": "TB incidence (per 100,000)",
    "percentage_latent": "Percentage of latent TB infection (%)",
    "prevalence_pulmonary": "Prevalence of pulmonary TB (per 100,000)",
    "case_detection_rate": "Case detection rate",
    "mortality_raw": "TB deaths",
    "adults_prevalence_pulmonary": "Prevalence of adults pulmonary TB (per 100,000)",
    "prevalence_smear_positive": "Prevalence of adults smear positive pulmonary TB (per 100,000)",
    "detection_rate": "Screening profile"
}

params_name = {
        "contact_rate" : "Transmission scaling factor",
        "rr_infection_latent" : 'Relative risk of infection for individuals with latent infection (ref. Infection-naive)',
        "rr_infection_recovered" : 'Relative risk of infection for individuals with history of infection (ref. Infection - naive)',
        "progression_multiplier" : 'Uncertainty multiplier for the rates of TB progression',
        "smear_positive_death_rate" : 'Smear positive TB death rate',
        "smear_negative_death_rate" : 'Smear negative TB death rate',
        "smear_positive_self_recovery" : 'Smear positive TB self-recovery rate',
        "smear_negative_self_recovery" : 'Smear negative TB self-recovery rate',
        "screening_scaleup_shape" : 'Passive screening shape',
        "screening_inflection_time" : 'Passive screening inflection time',
        "time_to_screening_end_asymp" : 'Time from active TB to be screened',
        "detection_reduction" : 'Relative reduction of screening rate during COVID-19',
        "contact_reduction" : 'Relative reduction of contact rate during COVID-19',
     
}
