import os
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
MIMIC_DATA_DIR = Path(__file__).parent.parent / "MIMIC_data"
INPUT_CSV = os.path.join(MIMIC_DATA_DIR, "NOTEEVENTS_DEMO.csv")
OUTPUT_CSV_1 = os.path.join(DATA_DIR, "NOTEEVENTS_DEMO_1.csv")
OUTPUT_CSV_2 = os.path.join(DATA_DIR, "NOTEEVENTS_DEMO_2.csv")

HEADER = "ROW_ID,SUBJECT_ID,HADM_ID,CHARTDATE,CHARTIME,STORETIME,CATEGORY,DESCRIPTION,CGID,ISERROR,TEXT\n"

def extract_sample(input_file, output_file, n_rows=5, skip_rows=None, add_manual_header=False) -> None:
    """
    Extracts a sample of rows from a large CSV file and saves it as a new CSV.

    Parameters:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to the output CSV file.
        n_rows (int): Number of rows to extract.
        skip_rows (int or None): Number of rows to skip before extracting.
        add_manual_header (bool): Whether to manually add a predefined header to the output file.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return

    print(f"Extracting {n_rows} rows from {input_file}...")

    df = pd.read_csv(input_file, nrows=n_rows, skiprows=skip_rows)

    with open(output_file, "w") as f:
        if add_manual_header:
            f.write(HEADER)
        df.to_csv(f, index=False, header=(not add_manual_header))  
    print(f"Extraction successful: {n_rows} rows saved to {output_file}")

if __name__ == "__main__":
    extract_sample(INPUT_CSV, OUTPUT_CSV_1, n_rows=100)
    extract_sample(INPUT_CSV, OUTPUT_CSV_2, n_rows=100, skip_rows=100, add_manual_header=True)
