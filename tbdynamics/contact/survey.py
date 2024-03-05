import requests
import time
import json
import re
import tempfile
from bs4 import BeautifulSoup
from pathlib import Path
import pandas as pd
import pycountry
import numpy as np

# Define the base and data paths using pathlib for better path management
BASE_PATH = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_PATH / "data/contact/"


def is_doi(s):
    """Check if a string is a DOI."""
    return re.match(r"^10.\d{4,9}/[-._;()/:A-Z0-9]+$", s, re.IGNORECASE) is not None


def download_survey(survey, dir=DATA_PATH, sleep=1):
    dir = Path(dir) if dir else Path(tempfile.gettempdir()) / "survey_data"
    dir.mkdir(parents=True, exist_ok=True)

    survey = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", survey)
    survey = re.sub(r"#.*$", "", survey)
    is_doi_flag = is_doi(survey)
    is_url = is_doi_flag or re.match(r"^https?://", survey)

    if not is_url:
        raise ValueError("'survey' is not a DOI or URL.")

    if is_doi_flag:
        url = f"https://doi.org/{survey}"
    else:
        url = survey

    headers = {"User-Agent": "Python HTTP Requests/YourAppName"}
    response = requests.get(url, headers=headers, allow_redirects=True)
    if response.status_code == 404:
        raise Exception(f"DOI '{survey}' not found")
    if response.status_code != 200:
        raise Exception("Could not fetch the resource.")

    soup = BeautifulSoup(response.content, "html.parser")
    links = soup.find_all("link", {"type": "text/csv"})
    csv_urls = [link.get("href") for link in links]

    files = []
    for csv_url in csv_urls:
        file_name = Path(csv_url).name
        file_path = dir / file_name.lower()
        print(f"Downloading {csv_url} to {file_path}")
        with requests.get(csv_url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        time.sleep(sleep)
        files.append(file_path)

    return files


def load_survey(files):
    csv_files = [file for file in files if file.suffix == ".csv"]
    json_file = next((file for file in files if file.suffix == ".json"), None)

    participants_df = pd.DataFrame()
    contacts_df = pd.DataFrame()

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if "participant" in csv_file.stem:
            participants_df = pd.concat([participants_df, df], ignore_index=True)
        elif "contact" in csv_file.stem:
            contacts_df = pd.concat([contacts_df, df], ignore_index=True)

    reference = {}
    if json_file:
        with open(json_file, "r", encoding="utf-8") as f:
            reference = json.load(f)

    survey = {
        "participants": participants_df,
        "contacts": contacts_df,
        "reference": reference,
    }

    return survey


def clean_survey(survey, country_column="country", participant_age_column="part_age"):
    """
    Cleans the survey data, adjusting country names and participant ages.

    :param survey: A dictionary containing 'participants', 'contacts', and 'reference'.
    :param country_column: The name of the column containing country data.
    :param participant_age_column: The name of the column containing participant ages.
    :return: The cleaned survey data.
    """

    def clean_dataframe(df, country_column, participant_age_column):
        if country_column in df.columns:
            df[country_column] = df[country_column].apply(
                lambda name: (
                    pycountry.countries.lookup(name).name if pd.notnull(name) else name
                )
            )

        if participant_age_column in df.columns:
            df[participant_age_column] = df[participant_age_column].apply(
                lambda age: process_age(age) if pd.notnull(age) else age
            )

        return df

    def process_age(age):
        """Process age values, converting ranges to their midpoint or parsing as integer."""
        if isinstance(age, str) and "-" in age:
            low, high = age.split("-")
            return (int(low) + int(high)) // 2
        try:
            return int(age)
        except ValueError:
            return None

    # Clean participants and contacts DataFrames
    if "participants" in survey:
        survey["participants"] = clean_dataframe(
            survey["participants"], country_column, participant_age_column
        )
    if "contacts" in survey:
        survey["contacts"] = clean_dataframe(
            survey["contacts"], country_column, participant_age_column
        )

    return survey


# Update the get_survey function accordingly
def get_survey(
    survey_url, country_column="country", participant_age_column="part_age", **kwargs
):
    files = download_survey(survey_url)
    survey_data = load_survey(files)
    cleaned_survey = clean_survey(
        survey_data,
        country_column=country_column,
        participant_age_column=participant_age_column,
        **kwargs,
    )

    if "reference" in cleaned_survey and cleaned_survey["reference"]:
        reference_title = cleaned_survey["reference"].get("title", "this survey")
        print(
            f"Using {reference_title}. To cite this in a publication, use the 'get_citation()' function."
        )

    return cleaned_survey


def check_survey(
    survey,
    id_column="part_id",
    participant_age_column="part_age",
    country_column="country",
    year_column="year",
    contact_age_column="cnt_age",
):
    # Assume survey is a dictionary with 'participants' and 'contacts' keys
    success = True
    messages = []

    required_participant_columns = [
        id_column,
        participant_age_column,
        country_column,
        year_column,
    ]
    required_contact_columns = [id_column, contact_age_column]

    # Check for required columns in 'participants'
    missing_participant_columns = [
        col
        for col in required_participant_columns
        if col not in survey["participants"].columns
    ]
    if missing_participant_columns:
        messages.append(
            f"Missing columns in participants: {', '.join(missing_participant_columns)}"
        )
        success = False

    # Check for required columns in 'contacts'
    missing_contact_columns = [
        col for col in required_contact_columns if col not in survey["contacts"].columns
    ]
    if missing_contact_columns:
        messages.append(
            f"Missing columns in contacts: {', '.join(missing_contact_columns)}"
        )
        success = False

    # Check for additional requirements for contact age column
    if contact_age_column not in survey["contacts"].columns:
        exact_column = f"{contact_age_column}_exact"
        est_min_column = f"{contact_age_column}_est_min"
        est_max_column = f"{contact_age_column}_est_max"
        if not (
            {exact_column, est_min_column, est_max_column}
            & set(survey["contacts"].columns)
        ):
            messages.append(
                f"Either {contact_age_column}, {exact_column}, or both {est_min_column} and {est_max_column} must exist in contacts"
            )
            success = False

    # Print the result of the check
    if success:
        print("Check OK.")
    else:
        for message in messages:
            print(message)
        print("Check FAILED.")

    # Return the columns (this part can be adjusted based on need)
    return {
        "id": id_column,
        "participant_age": participant_age_column,
        "country": country_column,
        "year": year_column,
        "contact_age": contact_age_column,
    }


def validate_survey(survey):
    error_string = "The input must be a survey object (created using `survey()` or `get_survey()`)."

    # Check if survey is a dictionary with the required keys
    if not isinstance(survey, dict) or not all(
        key in survey for key in ["participants", "contacts", "reference"]
    ):
        # Assuming get_survey can also validate or retrieve a survey, you might call it here
        # For demonstration, we'll just print an error and return None
        print(error_string)
        return None
    else:
        # Assuming the survey is valid, return it
        return survey


def limits_to_agegroups(x, limits=None):
    if limits is None:
        limits = sorted(set(x.dropna().unique()))
    else:
        limits = sorted(set(limits))

    age_groups = []
    for i in range(len(limits) - 1):
        if limits[i + 1] - 1 > limits[i]:
            age_groups.append(f"{limits[i]}-{limits[i + 1] - 1}")
        else:
            age_groups.append(f"{limits[i]}")
    age_groups.append(f"{limits[-1]}+")

    # Map ages to age groups
    def map_age_to_group(age):
        for limit, group in zip(limits, age_groups):
            if age < limit:
                return group
        return age_groups[-1]

    return x.map(map_age_to_group)


def reduce_agegroups(x, limits):
    limits = sorted(set(limits))  # Ensure limits are sorted and unique

    def find_limit(age):
        """Find the closest lower limit for the given age."""
        for limit in reversed(limits):
            if age >= limit:
                return limit
        return None  # Return None if no limit is found (age is lower than the lowest limit)

    ret = [find_limit(age) for age in x]
    return ret


def list_surveys():
    """List all surveys available for download from Zenodo community."""
    response = requests.get(
        "https://zenodo.org/api/records",
        params={
            "q": "social contact data",
            "access_token": "hbEgoRr5NQ2fwwV8t74tUXvOXKujRDblZnRSTo6LD4jNzJnRrQ2ZkZpWQ4GE",
        },
    ).json()

    surveys = [
        {
            "date_added": record["metadata"]["publication_date"],
            "title": record["metadata"]["title"],
            "creator": ", ".join(
                [author["name"] for author in record["metadata"]["creators"]]
            ),
            "url": record["links"]["doi"],
        }
        for record in response["hits"]["hits"]
    ]

    return pd.DataFrame(surveys)


def survey_countries(survey_df, country_column="country"):
    """List all countries contained in a survey DataFrame."""
    return survey_df[country_column].unique().tolist()


def wpp_countries(popF_df, popM_df):
    """List all countries and regions for which there is population data, based on WPP data."""
    countries = pd.concat([popF_df["country_code"], popM_df["country_code"]]).unique()
    country_names = [
        pycountry.countries.get(numeric=code).name
        for code in countries
        if pycountry.countries.get(numeric=code)
    ]
    return country_names


def contact_matrix(
    survey_data,
    survey_pop=None,
    age_limits=None,
    bootstrap=False,
    counts=False,
    symmetric=False,
    sample_participants=False,
    estimated_participant_age="mean",
    estimated_contact_age="mean",
    missing_participant_age="remove",
    missing_contact_age="remove",
    weights=None,
    weigh_dayofweek=False,
    weigh_age=False,
):
    surveys = ["participants", "contacts"]
    validated_survey = validate_survey(survey_data)
    if validated_survey is not None:
        print("Survey is valid.")
    else:
        print("Survey validation failed.")
    survey = clean_survey(survey_data)
    columns = check_survey(survey)
    participant_age_column = columns["participant_age"]

    # Concatenate to create new column names with suffixes
    part_min_column = f"{participant_age_column}_est_min"
    part_max_column = f"{participant_age_column}_est_max"
  # Assuming 'survey' is a dictionary with 'participants' as a pandas DataFrame
    participants_df = survey["participants"]

    # Assuming 'columns' is a dictionary with necessary column names, and 'participant_age_column' is defined
    participant_age_column = columns["participant_age"]
    if participant_age_column not in participants_df.columns:
        participants_df[participant_age_column] = None
    
    if part_max_column not in participants_df.columns and participant_age_column in participants_df.columns:
        max_age = participants_df[participant_age_column].max(skipna=True) + 1
    elif part_max_column in participants_df.columns and participant_age_column in participants_df.columns:
        max_age = max(participants_df[participant_age_column].max(skipna=True), participants_df[part_max_column].max(skipna=True)) + 1
    elif part_max_column in participants_df.columns:
        max_age = participants_df[participant_age_column].max(skipna=True) + 1

    if age_limits is None:
        all_ages = survey['participants'][columns["participant.age"]].dropna().unique().astype(int)
        all_ages = np.sort(all_ages)
        age_limits = np.union1d([0], all_ages)  # Includes 0 and ensures uniqueness
    else:
        age_limits = np.array(age_limits, dtype=int)

    if pd.isnull(age_limits).any() or np.any(np.diff(age_limits) <= 0):
        raise ValueError("'age_limits' must be an increasing integer vector of lower age limits.")
    
    if part_min_column in survey['participants'].columns and part_max_column in survey['participants'].columns:
        if estimated_participant_age == "mean":
            # Calculate the mean of the min and max, then fill NA values in the age column with it
            age_mean = survey['participants'][[part_min_column, part_max_column]].mean(axis=1)
            survey['participants'].loc[survey['participants'][columns["participant_age"]].isna(), columns["participant_age"]] = age_mean.astype(int)
        
        elif estimated_participant_age == "sample":
            # For each row where age is NA, sample a random age between min and max
            for index, row in survey['participants'].loc[survey['participants'][columns["participant_age"]].isna()].iterrows():
                if row[part_min_column] <= row[part_max_column]:  # Ensuring min is not greater than max
                    sampled_age = np.random.randint(row[part_min_column], row[part_max_column] + 1)
                    survey['participants'].at[index, columns["participant_age"]] = sampled_age

    if missing_participant_age == "remove":
    # Remove participants with NA age or age below the minimum age limit
        valid_age_condition = survey['participants'][columns["participant.age"]].notna() & \
                          (survey['participants'][columns["participant.age"]] >= min(age_limits))
    if not valid_age_condition.all():
        print("Removing participants without age information or with age below the minimum age limit.")
        survey['participants'] = survey['participants'][valid_age_condition]

    contacts_df = survey['contacts']
    exact_column = f"{columns['contact_age']}_exact"
    min_column = f"{columns['contact_age']}_est_min"
    max_column = f"{columns['contact_age']}_est_max"

    # Check if the contact age column is not in the contacts DataFrame and set it to NaN if absent
    if columns['contact_age'] not in contacts_df.columns:
        contacts_df[columns['contact_age']] = None

    # If the exact age column exists in the contacts DataFrame, use it to fill in the contact age column
    if exact_column in contacts_df.columns:
        contacts_df.loc[contacts_df[exact_column].notna(), columns['contact.age']] = contacts_df[exact_column]  

    if min_column in contacts_df and max_column in contacts_df:
        min_ages = contacts_df[min_column]
        max_ages = contacts_df[max_column]
        if estimated_contact_age == "mean":
            contacts_df["contact_age"] = np.where(
                np.isnan(contacts_df["contact_age"]),
                (min_ages + max_ages) / 2,
                contacts_df["contact_age"]
            )
        elif estimated_contact_age == "sample":
            contacts_df["contact_age"] = np.where(
                np.isnan(survey["contacts"]["contact_age"]),
                np.random.randint(min_ages, max_ages + 1),
                contacts_df["contact_age"]
            )

    condition = contacts_df["contact_age"].isna() | (contacts_df["contact_age"] < age_limits[0])

  





