"""
Loads and preprocesses the three standard fairness datasets used in the experiment:
Adult Income, COMPAS Recidivism, and German Credit.

Each dataset is downloaded once and cached in ``data/`` as a clean CSV.
Preprocessing binarizes protected attributes (sex, race, age) and encodes
categorical variables as strings to preserve semantic labels for LLM prompts.

Privileged group conventions follow the original literature:
- Adult:   sex=Male (1), race=White (1)         [Kohavi, 1996]
- COMPAS:  sex=Male (1), race=non-AA (1)         [Angwin et al., 2016]
- German:  sex=Male (1), age>=25 (1)             [Dua & Graff, 2017]
"""

import pandas as pd
import numpy as np
from pathlib import Path
from ucimlrepo import fetch_ucirepo

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

COMPAS_URL = (
    "https://raw.githubusercontent.com/propublica/compas-analysis/"
    "master/compas-scores-two-years.csv"
)


def load_adult(save=True):
    """Load the Adult Income dataset from UCI or from local cache.

    Applies the following transformations:
    - Drops rows with missing values (represented as '?').
    - Binarizes ``income`` (1 if >50K, 0 if <=50K).
    - Binarizes ``sex`` (1=Male, 0=Female) and ``race`` (1=White, 0=other).
    - Encodes ordinal categorical variables with LabelEncoder.

    Args:
        save (bool): If True, saves the processed DataFrame to ``data/adult.csv``.

    Returns:
        pd.DataFrame: DataFrame with selected and preprocessed columns.
    """
    cache = DATA_DIR / "adult.csv"
    if cache.exists():
        df = pd.read_csv(cache)
        print(f"[data_loader] Adult loaded from cache: {df.shape}")
        return df

    print("[data_loader] Downloading Adult from UCI...")
    dataset = fetch_ucirepo(id=2)
    features = dataset.data.features
    targets = dataset.data.targets
    df = pd.concat([features, targets], axis=1)
    df.columns = [c.strip() for c in df.columns]
    df = df.replace("?", np.nan).dropna()

    features_keep = [
        "workclass", "hours-per-week", "sex", "age", "race",
        "occupation", "capital-loss", "education",
        "capital-gain", "marital-status", "relationship"
    ]
    target_col = "income"
    df = df[features_keep + [target_col]].copy()

    # Original label may include variants like ">50K." (with trailing dot)
    df[target_col] = (df[target_col].str.strip().str.startswith(">50K")).astype(int)
    df["sex"] = (df["sex"].str.strip() == "Male").astype(int)
    df["race"] = (df["race"].str.strip() == "White").astype(int)

    # Keep categorical columns as strings so LLMs receive semantic labels
    # (e.g. "Bachelors" instead of an arbitrary integer).
    cat_cols = ["workclass", "occupation", "education", "marital-status", "relationship"]
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip()

    if save:
        df.to_csv(cache, index=False)
    print(f"[data_loader] Adult: {df.shape}")
    return df


def load_compas(save=True):
    """Load the COMPAS dataset from the ProPublica repository or from local cache.

    Applies the standard literature filters:
    - Keeps records with ``days_b_screening_arrest`` between -30 and 30.
    - Excludes records with ``is_recid == -1``.
    - Excludes ordinance charges ('O') and score_text == 'N/A'.
    - Binarizes ``race`` (1 if NOT African-American, 0 otherwise).
    - Binarizes ``sex`` (1=Male) and ``c_charge_degree`` (1=Felony).
    - Encodes ``score_text`` on an ordinal scale 0-2 (Low/Medium/High).

    Args:
        save (bool): If True, saves the processed DataFrame to ``data/compas.csv``.

    Returns:
        pd.DataFrame: DataFrame with selected and preprocessed columns.
    """
    cache = DATA_DIR / "compas.csv"
    if cache.exists():
        df = pd.read_csv(cache)
        print(f"[data_loader] COMPAS loaded from cache: {df.shape}")
        return df

    print("[data_loader] Downloading COMPAS from ProPublica...")
    raw = pd.read_csv(COMPAS_URL)

    # Filters to remove ambiguous cases
    raw = raw[raw["days_b_screening_arrest"].between(-30, 30)]
    raw = raw[raw["is_recid"] != -1]
    raw = raw[raw["c_charge_degree"] != "O"]
    raw = raw[raw["score_text"] != "N/A"]

    features_keep = ["sex", "race", "age", "c_charge_degree", "priors_count", "score_text"]
    target_col = "two_year_recid"
    df = raw[features_keep + [target_col]].copy()

    # Race binarized: 1 = not African-American (privileged group)
    df["race"] = (df["race"] != "African-American").astype(int)
    df["sex"] = (df["sex"] == "Male").astype(int)
    df["c_charge_degree"] = (df["c_charge_degree"] == "F").astype(int)

    score_map = {"Low": 0, "Medium": 1, "High": 2}
    df["score_text"] = df["score_text"].map(score_map)

    if save:
        df.to_csv(cache, index=False)
    print(f"[data_loader] COMPAS: {df.shape}")
    return df


def load_german(save=True):
    """Load the German Credit dataset from UCI or from local cache.

    Renames generic columns to descriptive names and applies the following transformations:
    - Binarizes ``label`` (1=good credit, 0=bad credit).
    - Binarizes ``sex`` based on codes A91/A93/A94 (male=1).
    - Binarizes ``age`` (1 if >= 25 years, standard threshold in the literature).
    - Keeps remaining categorical variables as strings for LLM readability.

    Args:
        save (bool): If True, saves the processed DataFrame to ``data/german.csv``.

    Returns:
        pd.DataFrame: DataFrame with selected and preprocessed columns.
    """
    cache = DATA_DIR / "german.csv"
    if cache.exists():
        df = pd.read_csv(cache)
        print(f"[data_loader] German loaded from cache: {df.shape}")
        return df

    print("[data_loader] Downloading German Credit from UCI...")
    dataset = fetch_ucirepo(id=144)
    features = dataset.data.features
    targets = dataset.data.targets
    df = pd.concat([features, targets], axis=1)
    df.columns = [c.strip() for c in df.columns]

    # Column name map: generic UCI names -> descriptive names
    col_map = {
        "Attribute1": "credit_history", "Attribute2": "duration",
        "Attribute3": "credit_history_detail", "Attribute4": "purpose",
        "Attribute5": "credit_amount", "Attribute6": "savings",
        "Attribute7": "employment", "Attribute8": "installment_rate",
        "Attribute9": "sex", "Attribute10": "other_debtors",
        "Attribute12": "property", "Attribute13": "age",
        "Attribute19": "housing", "Attribute20": "job",
        "class": "label"
    }
    filtered_column_map = {k: v for k, v in col_map.items() if k in df.columns}
    df = df[list(filtered_column_map.keys())].rename(columns=filtered_column_map)

    df["label"] = (df["label"] == 1).astype(int)

    if "sex" in df.columns:
        # A91=male divorced/separated, A93=male single, A94=male married/widowed
        df["sex"] = df["sex"].apply(
            lambda x: 1 if str(x).strip() in ["A91", "A93", "A94"] else 0
        )
    if "age" in df.columns:
        df["age"] = (df["age"] >= 25).astype(int)

    # Keep categorical columns as strings so LLMs receive semantic labels
    # (e.g. "A12" instead of an arbitrary integer).
    cat_cols = [c for c in df.columns if df[c].dtype == object and c not in ["sex", "age", "label"]]
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip()

    if save:
        df.to_csv(cache, index=False)
    print(f"[data_loader] German: {df.shape}")
    return df


DATASET_CONFIG = {
    "adult": {
        "loader": load_adult,
        "target": "income",
        "protected": ["sex", "race"],
        "privileged": {"sex": 1, "race": 1},
    },
    "compas": {
        "loader": load_compas,
        "target": "two_year_recid",
        "protected": ["sex", "race"],
        "privileged": {"sex": 1, "race": 1},
    },
    "german": {
        "loader": load_german,
        "target": "label",
        "protected": ["sex", "age"],
        "privileged": {"sex": 1, "age": 1},
    },
}


def load_dataset(name: str):
    """Load a dataset by name and return the DataFrame along with its configuration.

    Args:
        name (str): Dataset name. Options: 'adult', 'compas', 'german'.

    Returns:
        tuple[pd.DataFrame, dict]: Preprocessed DataFrame and configuration dictionary
            with keys ``target``, ``protected``, and ``privileged``.

    Raises:
        AssertionError: If ``name`` is not in ``DATASET_CONFIG``.
    """
    assert name in DATASET_CONFIG, (
        f"Dataset '{name}' not recognized. Options: {list(DATASET_CONFIG)}"
    )
    config = DATASET_CONFIG[name]
    df = config["loader"]()
    return df, config
