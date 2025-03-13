import pandas as pd
from pathlib import Path
import re

# ------------------------ Data Loading ------------------------


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load structured notes and patient demographic datasets.
    Returns:
        tuple: (structured_notes, patients)
    """
    structured_notes = pd.read_csv(
        Path(__file__).parent.parent / "1_data_structuration" / "data" / "structured_notes_anonymized.csv"
    )
    patients = pd.read_csv(Path(__file__).parent.parent / "MIMIC_data" / "PATIENTS_anonymized.csv")
    return structured_notes, patients


# Load datasets
df, df_patients = load_data()

# ------------------------ Data Preprocessing ------------------------

# Normalize column names
df.columns = df.columns.str.strip().str.upper()
df_patients.columns = df_patients.columns.str.strip().str.upper()

# ------------------------ CAD Identification ------------------------


def identify_cad(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify patients with Coronary Artery Disease (CAD) based on pathology and medical history.

    Args:
        df (pd.DataFrame): The structured notes dataset.

    Returns:
        pd.DataFrame: Updated DataFrame with CAD presence and risk factors.
    """
    df["CAD_PRESENT"] = df["MAIN_PATHOLOGY"].str.contains(
        r"\b(?:CAD|Coronary Artery Disease)\b", case=False, na=False, regex=True
    ) | df["MEDICAL_PERSONAL_HISTORY"].str.contains(
        r"\b(?:CAD|Coronary Artery Disease)\b", case=False, na=False, regex=True
    )
    df["CAD_PRESENT"] = df["CAD_PRESENT"].astype(int)

    # Identify risk factors
    df["HYPERTENSION"] = (
        df["MEDICAL_PERSONAL_HISTORY"].str.contains("Hypertension|HTN", case=False, na=False).astype(int)
    )
    df["DIABETES"] = (
        df["MEDICAL_PERSONAL_HISTORY"]
        .str.contains("Diabetes|Diabetes Mellitus|Type 2 Diabetes", case=False, na=False)
        .astype(int)
    )
    df["FAMILY_HISTORY_CAD"] = (
        df["MEDICAL_FAMILY_HISTORY"].str.contains("CAD|Coronary", flags=re.IGNORECASE, na=False).astype(int)
    )
    df["HYPERLIPIDEMIA"] = (
        df["MEDICAL_PERSONAL_HISTORY"]
        .str.contains("Hyperlipidemia|Dyslipidemia|High Cholesterol", case=False, na=False)
        .astype(int)
    )
    df["MYOCARDIAL_INFARCTION"] = (
        df["MEDICAL_PERSONAL_HISTORY"]
        .str.contains("Myocardial Infarction|Heart Attack", case=False, na=False)
        .astype(int)
    )

    return df


df = identify_cad(df)

# ------------------------ Data Aggregation ------------------------


def aggregate_patient_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate patient-level data to remove duplicates while keeping key information.

    Args:
        df (pd.DataFrame): Processed DataFrame with CAD identification.

    Returns:
        pd.DataFrame: Aggregated DataFrame with unique patient records.
    """
    df_grouped = (
        df.groupby("SUBJECT_ID")
        .agg(
            {
                "CAD_PRESENT": "max",
                "HYPERTENSION": "max",
                "DIABETES": "max",
                "FAMILY_HISTORY_CAD": "max",
                "HYPERLIPIDEMIA": "max",
                "MYOCARDIAL_INFARCTION": "max",
            }
        )
        .reset_index()
    )

    return df_grouped


df_grouped = aggregate_patient_data(df)

# ------------------------ Merging with Demographics ------------------------


def merge_patient_info(df_grouped: pd.DataFrame, df_patients: pd.DataFrame) -> pd.DataFrame:
    """
    Merge patient demographic information and calculate age.

    Args:
        df_grouped (pd.DataFrame): Aggregated dataset of unique patients.
        df_patients (pd.DataFrame): Patient demographic dataset.

    Returns:
        pd.DataFrame: Merged dataset with demographic details.
    """
    df_merged = df_grouped.merge(df_patients[["SUBJECT_ID", "GENDER", "DOB", "DOD_HOSP"]], on="SUBJECT_ID", how="left")

    # Convert dates to datetime format
    df_merged["DOB"] = pd.to_datetime(df_merged["DOB"], errors="coerce")
    df_merged["DOD_HOSP"] = pd.to_datetime(df_merged["DOD_HOSP"], errors="coerce")

    # Calculate age based on hospitalization date
    df_merged["AGE"] = df_merged.apply(
        lambda row: (row["DOD_HOSP"].year - row["DOB"].year) if pd.notnull(row["DOD_HOSP"]) else None, axis=1
    )

    # Remove unrealistic ages (> 110 years)
    df_merged.loc[df_merged["AGE"] > 110, "AGE"] = None

    # Drop unnecessary columns
    df_merged = df_merged.drop(columns=["DOB", "DOD_HOSP"])

    return df_merged


df_grouped = merge_patient_info(df_grouped, df_patients)

# ------------------------ Dataset Balancing ------------------------


def balance_dataset(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """
    Balance dataset to have an equal number of CAD and non-CAD patients.

    Args:
        df (pd.DataFrame): Processed dataset.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        pd.DataFrame: Balanced dataset.
    """
    df_cad = df[df["CAD_PRESENT"] == 1]
    df_non_cad = df[df["CAD_PRESENT"] == 0].sample(n=len(df_cad), random_state=random_state)

    # Concatenate both groups and sort by SUBJECT_ID
    df_balanced = pd.concat([df_cad, df_non_cad]).sort_values(by="SUBJECT_ID").reset_index(drop=True)

    return df_balanced


df_balanced = balance_dataset(df_grouped)

# ------------------------ Save Processed Data ------------------------


def save_dataset(df: pd.DataFrame, filename: str):
    """
    Save the processed dataset to a CSV file.

    Args:
        df (pd.DataFrame): Final dataset to be saved.
        filename (str): Output file name.
    """
    df.to_csv(Path(__file__).parent / "data" / filename, index=False)
    print(f"Dataset saved as {filename}")


save_dataset(df_balanced, "CAD_dataset.csv")

print("Processing complete: Dataset filtered, balanced, and ready for analysis.")
