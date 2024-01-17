from pathlib import Path
import pandas as pd

BASE_PATH = Path(__file__).parent.parent.resolve()
DATA_PATH = BASE_PATH / "data"


# def load_pop_data():
#     csv_path = pathlib.Path(DATA_PATH / "camau.csv")
#     pop_df = pd.read_csv(csv_path)
#     pop_df = pop_df.set_index(["year"])

#     csv_path = pathlib.Path(DATA_PATH / "camau_birth.csv")
#     birth_df = pd.read_csv(csv_path)

#     csv_path = pathlib.Path(DATA_PATH / "camau_cdr.csv")
#     death_df = pd.read_csv(csv_path)

#     return pop_df, birth_df, death_df


# death_rates_by_age = {
#     0: [
#         0.03677278184202013,
#         0.02777818672989001,
#         0.021706568192537818,
#         0.021323788650189943,
#         0.026506652291030792,
#         0.016173454912698695,
#         0.014566728226477376,
#         0.01241651696156871,
#         0.009659593656645303,
#         0.006770377935370667,
#         0.00585657470931215,
#         0.005107746497854402,
#         0.004960977533876599,
#         0.00466746526830646,
#     ],
#     5: [
#         0.0044348127009039,
#         0.0035553819354682222,
#         0.0028173765143792153,
#         0.003054227317350724,
#         0.0039223028281727594,
#         0.0017644745314042065,
#         0.0013845329508603717,
#         0.0012244661278236553,
#         0.0011365593984756605,
#         0.0010044102813463434,
#         0.0009184777086575956,
#         0.0009115236813529538,
#         0.000938448858656349,
#         0.0009477553047443998,
#     ],
#     15: [
#         0.008040233365150818,
#         0.004316166955312478,
#         0.003961498903760318,
#         0.004897527507146597,
#         0.007203106257741426,
#         0.0026087173928617986,
#         0.0022665638022687546,
#         0.0021193069527976425,
#         0.0020012143228689055,
#         0.0018142639204592305,
#         0.001779469054345947,
#         0.001748401931120158,
#         0.0017472889448224304,
#         0.0017415116796319241,
#     ],
#     35: [
#         0.011020459324598286,
#         0.007056057658847496,
#         0.006377061331904774,
#         0.007492964325706753,
#         0.010226177129813297,
#         0.004680499805774958,
#         0.004027585609157603,
#         0.0035984710218771457,
#         0.0033531329551140837,
#         0.0031139847074135208,
#         0.0030947484584764448,
#         0.0030441594627912795,
#         0.002999384323702785,
#         0.002962157146801445,
#     ],
#     50: [
#         0.03314148321244275,
#         0.029175784561375272,
#         0.02916504443312662,
#         0.03316546300916001,
#         0.04026110554837792,
#         0.02901629784969708,
#         0.027940339006809926,
#         0.02781287188661504,
#         0.027876032516126522,
#         0.02736912927269689,
#         0.026265801373879558,
#         0.024142353697524314,
#         0.02272942817725227,
#         0.022019612421144846,
#     ],
# }

# death_rate_years = [
#     1952.5,
#     1957.5,
#     1962.5,
#     1967.5,
#     1972.5,
#     1977.5,
#     1982.5,
#     1987.5,
#     1992.5,
#     1997.5,
#     2002.5,
#     2007.5,
#     2012.5,
#     2017.5,
# ]


fixed_parameters = {
    "acf_screening_sensitivity": 0.75,
    "age_infectiousness_switch": 15.0,
    "age_mixing": {"age_adjust": True, "source_iso3": "VNM"},
    "age_latency": {
        "early_activation": {0: 2.4107, 5: 0.9862, 15: 0.0986},
        "late_activation": {0: 6.939769e-09, 5: 0.0023, 15: 0.0012},
        "stabilisation": {0: 4.383, 5: 4.383, 15: 1.972},
    },
    "bcg_effect": "infection",
    "calculated_outputs": ["prevalence_infectious"],
    "cdr_adjustment": 0.8003452159878636,
    "contact_rate": 0.013414102898074345,
    "country": {"country_name": "Vietnam", "iso3": "VNM"},
    "crude_birth_rate": 0.2,
    "crude_death_rate": 0.0008,
    "cumulative_output_start_time": 2020.0,
    "cumulative_start_time": 1990.0,
    "description": "BASELINE",
    "extrapulmonary_infect_multiplier": 0.0,
    "future_diabetes_multiplier": 1.0,
    "gender": {
        "adjustments": {
            "detection": {"female": 1.0, "male": 1.5},
            "infection": {"female": 1.0, "male": 2.6408657914674176},
        },
        "proportions": {"female": 0.5, "male": 0.5},
        "strata": ["male", "female"],
    },
    "incidence_props_pulmonary": 0.85,
    "incidence_props_smear_positive_among_pulmonary": 0.75,
    "infect_death_rate_dict": {
        "smear_negative": 0.025,
        "smear_positive": 0.389,
        "unstratified": 0.2,
    },
    "infectious_seed": 1.0,
    "inflate_reactivation_for_diabetes": False,
    "on_treatment_infect_multiplier": 0.08,
    "outputs_stratification": {},
    "progression_multiplier": 1.1,
    "prop_death_among_negative_tx_outcome": 0.2,
    "prop_diabetes": {0: 0.01, 5: 0.05, 15: 0.2, 35: 0.4, 50: 0.7, 70: 0.8},
    "rr_infection_latent": 0.20278196465900813,
    "rr_infection_recovered": 0.21190687223342505,
    "rr_progression_diabetes": 5.643402828077587,
    "self_recovery_rate_dict": {
        "smear_negative": 0.22723824998716693,
        "smear_positive": 0.20344728302826143,
        "unstratified": 0.2,
    },
    "smear_negative_infect_multiplier": 0.25,
    "start_population_size": 267252.06827576435,
    "stratify_by": ["age", "organ", "gender"],
    "time": {"end": 2020, "start": 1900, "step": 0.1},
    "time_variant_bcg_perc": {
        1981: 0.1,
        1990: 49.0,
        1991: 71.0,
        1992: 72.0,
        1993: 88.0,
        1994: 96.0,
        1995: 71.0,
        1996: 98.0,
        1997: 94.0,
        1998: 81.0,
        1999: 81.0,
        2000: 89.0,
        2001: 99.0,
        2002: 90.0,
        2003: 93.0,
        2004: 91.0,
        2005: 93.0,
        2006: 92.0,
        2007: 92.0,
        2008: 95.0,
        2009: 98.0,
        2010: 99.0,
        2011: 80.0,
        2012: 97.0,
        2013: 93.0,
        2014: 89.0,
        2015: 99.0,
        2016: 94.0,
        2017: 92.0,
        2018: 98.0,
        2019: 89.0,
    },
    "time_variant_screening_rate": {
        2017: 0.0,
        2019: 1.0,
    },
    # 'time_variant_tb_screening_rate': { 1986.0: 0.13,
    #                                     1987.0: 0.34,
    #                                     1988.0: 0.44,
    #                                     1989.0: 0.5,
    #                                     1990.0: 0.56,
    #                                     1991.0: 0.63,
    #                                     1992.0: 0.67,
    #                                     1993.0: 0.61,
    #                                     1994.0: 0.59,
    #                                     1995.0: 0.61,
    #                                     1996.0: 0.8,
    #                                     1997.0: 0.8,
    #                                     1998.0: 0.81,
    #                                     1999.0: 0.83,
    #                                     2000.0: 0.87},
    "time_variant_tsr": {
        1986: 0.4,
        2000: 0.92,
        2001: 0.93,
        2002: 0.92,
        2003: 0.93,
        2004: 0.93,
        2005: 0.93,
        2006: 0.93,
        2007: 0.91,
        2008: 0.92,
        2009: 0.92,
        2010: 0.92,
        2011: 0.93,
        2012: 0.91,
        2013: 0.89,
        2014: 0.91,
        2015: 0.92,
        2016: 0.92,
        2017: 0.92,
        2018: 0.91,
        2019: 0.91,
    },
    "treatment_duration": 0.5,
    "passive_screening_sensitivity_extrapulmonary": 0.5,
    "passive_screening_sensitivity_smear_negative": 0.7,
    "passive_screening_sensitivity_smear_positive": 1.0,
    "acf_scaleup_shape": 0.05,
    "acf_inflection_time": 1990,
    "acf_start_asymp": 0.0,
    "acf_end_asymp": 10.0,
}


def get_birth_rate():
    return pd.read_csv(Path(DATA_PATH / "vn_birth.csv"), index_col=0)["value"]


def process_death_rate(age_strata: list):
    data = pd.read_csv(
        Path(DATA_PATH / "data.csv"), usecols=["Age", "Time", "Population", "Deaths"]
    )
    data = data.set_index(["Age", "Time"])
    birth_rates = get_birth_rate()
    data.index = data.index.swaplevel()
    age_groups = set(data.index.get_level_values(1))
    years = set(data.index.get_level_values(0))

    # Creating the new list
    agegroup_request = [
        [start, end - 1] for start, end in zip(age_strata, age_strata[1:] + [201])
    ]
    # agegroup_request = [[0, 4], [5, 14], [15, 34], [35, 49], [50, 69], [70, 200]]
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
    mapped_rates.index = mapped_rates.index + 0.5
    death_df = mapped_rates.loc[birth_rates.index]
    return death_df


def get_age_groups_in_range(age_groups, lower_limit, upper_limit):
    return [
        i
        for i in age_groups
        if "+" not in i and lower_limit <= int(i.split("-")[0]) <= upper_limit
    ]


death_rates_by_age = {
    0: [
        0.04891194320512563,
        0.033918668027576884,
        0.024432727316924446,
        0.019408181689238014,
        0.018148370056426787,
        0.014535299525724738,
        0.013035498777442808,
        0.011254176500418787,
        0.00931061542295055,
        0.006118489529305104,
        0.005678373734277731,
        0.005333479069860507,
        0.004783694403203202,
        0.0043347634430972014,
    ],
    5: [
        0.005654833192668025,
        0.005297274566250849,
        0.0044664893264966974,
        0.003846307644222109,
        0.005107979016750706,
        0.002711430406000275,
        0.002147435231479338,
        0.0016691038222857585,
        0.0013164357375949395,
        0.0010181764308804923,
        0.0007447183497013854,
        0.0005658632755602,
        0.0005441317519979431,
        0.0005100562239187423,
    ],
    15: [
        0.0030517485729701097,
        0.002700755057428503,
        0.002489838533286184,
        0.002584612327589561,
        0.004088806859043201,
        0.0023927769572424534,
        0.002209771395787853,
        0.001885328047816437,
        0.0016581044683420282,
        0.0015277277213098862,
        0.001427063312059633,
        0.0012953587483924324,
        0.0012356506891999784,
        0.0011729827102219131,
    ],
    35: [
        0.009386682178966717,
        0.007892734017957679,
        0.006517135357865979,
        0.006260889686458326,
        0.008689175571776193,
        0.00479116361370093,
        0.004179181883878279,
        0.004093556688069908,
        0.004031166261049662,
        0.003907751109471642,
        0.0034886248607591164,
        0.003155553373359392,
        0.003040343537587886,
        0.002980208830203494,
    ],
    50: [
        0.029187727095725802,
        0.02505798857782555,
        0.022037334772103276,
        0.020736476329400615,
        0.02776525440663067,
        0.017100990428597855,
        0.015457863497793123,
        0.014571257717622483,
        0.013493466497099146,
        0.013305916072161311,
        0.013036126707949302,
        0.013429693692230692,
        0.014203172938557912,
        0.0144835258644252,
    ],
    70: [
        0.21673523458855962,
        0.21670754053560007,
        0.2194770201646575,
        0.21936426210680712,
        0.22258003180907557,
        0.19193172117658122,
        0.18132416903612072,
        0.16726913532929819,
        0.15398367853882836,
        0.1443173787545218,
        0.13469347390283717,
        0.1315409693955733,
        0.12607141390724716,
        0.12549030934272593,
    ],
}
