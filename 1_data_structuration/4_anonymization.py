import pandas as pd
import hashlib
from pathlib import Path

def load_data(patients_path: Path, notes_path: Path):
    """Load patient and structured notes data from CSV files."""
    df_patients = pd.read_csv(patients_path)
    df_notes = pd.read_csv(notes_path)

    if "ROW_ID" in df_patients.columns:
        df_patients.drop(columns=["ROW_ID"], inplace=True)

    return df_patients, df_notes

def generate_subject_id_map(subject_ids):
    """Generate a consistent mapping of SUBJECT_IDs to anonymized values."""
    return {sid: hashlib.sha256(str(sid).encode()).hexdigest()[:10] for sid in subject_ids}

def anonymize_subject_ids(df_patients, df_notes, subject_id_map):
    """Replace SUBJECT_IDs with their anonymized counterparts."""
    df_patients["SUBJECT_ID"] = df_patients["SUBJECT_ID"].map(subject_id_map)
    df_notes["SUBJECT_ID"] = df_notes["SUBJECT_ID"].map(subject_id_map)
    return df_patients, df_notes

def main():
    """Main execution function."""
    base_dir = Path(__file__).parent.parent
    data_dir = Path(__file__).parent / "data"

    patients_file = base_dir / "MIMIC_data" / "PATIENTS.csv"
    notes_file = data_dir / "structured_notes.csv"
    output_patients_file = base_dir / "MIMIC_data" / "PATIENTS_anonymized.csv"
    output_notes_file = data_dir / "structured_notes_anonymized.csv"
    mapping_file = Path(__file__).parent / "subject_id_mapping.csv"

    df_patients, df_notes = load_data(patients_file, notes_file)

    subject_ids = set(df_patients["SUBJECT_ID"]).union(set(df_notes["SUBJECT_ID"]))
    subject_id_map = generate_subject_id_map(subject_ids)

    df_patients, df_notes = anonymize_subject_ids(df_patients, df_notes, subject_id_map)

    print("Anonymization completed. Files saved:")
    print(f"- {output_patients_file}")
    print(f"- {output_notes_file}")

if __name__ == "__main__":
    main()
