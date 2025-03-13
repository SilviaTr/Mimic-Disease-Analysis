import json
import os
import queue
import threading
import time
from pathlib import Path

import pandas as pd
import requests

API_KEY = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"  # Replace with your actual API key
API_KEY = "Ff7Jgh8ccoMhwlVrwa0fKT0sO4In2NSk"
API_URL = "https://api.mistral.ai/v1/chat/completions"

HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

INPUT_CSV = Path(__file__).parent / "data" / "NOTEEVENTS_DEMO_1.csv"
OUTPUT_CSV = Path(__file__).parent / "data" / "structured_notes_demo.csv"

BATCH_SIZE = 1
REQUEST_INTERVAL = 1.1
N_ROWS = 1000000

results_queue = queue.Queue()
lock = threading.Lock()


def structure_medical_note(text, subject_id, row_id) -> None:
    """Sends a clinical note to the Mistral API and retrieves structured JSON data."""
    prompt = f"""
    Analyze this clinical note and return only a structured JSON containing:
    - main_pathology (short and general diagnosis)
    - medical_personal_history (list of concise terms, no long descriptions)
    - medical_family_history (list of dictionaries with):
        - relation (e.g., "Father", "Mother", "Sibling")
        - condition (short and general term, e.g., "Stroke", "Diabetes", "Heart disease")

    Do not include ECG findings, waveforms, or non-diagnostic observations (e.g., "ST segment changes", "T wave abnormalities", "Baseline artifact").
    Clinical note:
    {text}
    """

    payload = {
        "model": "open-mixtral-8x22b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 500,
    }

    while True:
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        response_text = response.text.strip()

        try:
            if response.status_code == 429:
                print("Too many requests, retrying in 0.2 seconds...")
                time.sleep(0.2)
                continue

            if response.status_code != 200:
                print(f"API error {response.status_code}: {response_text}")
                return None

            parsed_json = json.loads(response.json().get("choices", [{}])[0].get("message", {}).get("content", "{}"))
            result = {
                "SUBJECT_ID": subject_id,
                "MAIN_PATHOLOGY": parsed_json.get("main_pathology", ""),
                "MEDICAL_PERSONAL_HISTORY": parsed_json.get("medical_personal_history", []),
                "MEDICAL_FAMILY_HISTORY": parsed_json.get("medical_family_history", []),
            }

            with lock:
                df_result = pd.DataFrame([result])
                df_result.to_csv(OUTPUT_CSV, mode="a", header=not os.path.exists(OUTPUT_CSV), index=False)
            break
        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")
            print(f"Response content: {response_text}")
            break


def process_batch() -> None:
    """Processes a batch of clinical notes and updates the CSV file without blocking."""
    print("Checking if source file exists...")
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Source file not found: {INPUT_CSV}")

    print("Loading CSV file...")
    df = pd.read_csv(INPUT_CSV, nrows=N_ROWS)
    print("File loaded. Preview:")
    print(df.head())

    threads = []

    for i in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[i : i + BATCH_SIZE]
        print(f"Processing batch {i}-{i + BATCH_SIZE}...")

        for _, row in batch.iterrows():
            thread = threading.Thread(
                target=structure_medical_note, args=(row["TEXT"], row["ROW_ID"], row["SUBJECT_ID"])
            )
            thread.start()
            threads.append(thread)
            time.sleep(REQUEST_INTERVAL)

    for thread in threads:
        thread.join()

    print("All results have been saved.")


if __name__ == "__main__":
    print("Starting structuring process...")
    process_batch()
    print("Structuring completed.")
