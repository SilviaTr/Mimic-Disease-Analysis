import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from datetime import datetime
import ast
from pathlib import Path

# ------------------------ Data Loading ------------------------

@st.cache_data
def load_data():
    """
    Load patient and structured notes datasets.
    
    Returns:
        structured_notes (pd.DataFrame): Dataset containing medical records.
        patients (pd.DataFrame): Dataset containing patient demographic information.
    """
    structured_notes = pd.read_csv(Path(__file__).parent.parent / "1_data_structuration" / "data" / "structured_notes_anonymized.csv")
    patients = pd.read_csv(Path(__file__).parent.parent / "MIMIC_data" / "PATIENTS.csv")
    return structured_notes, patients

structured_notes, patients = load_data()

# ------------------------ Data Preprocessing ------------------------

# Merge patient demographic information with structured notes
merged_df = structured_notes.merge(patients[['SUBJECT_ID', 'GENDER', 'DOB', 'DOD_HOSP']], on='SUBJECT_ID', how='left')

# Convert dates to datetime format
merged_df['DOB'] = pd.to_datetime(merged_df['DOB'], errors='coerce')
merged_df['DOD_HOSP'] = pd.to_datetime(merged_df['DOD_HOSP'], errors='coerce')

# Calculate age at the time of hospitalization
merged_df['AGE'] = merged_df.apply(lambda row: (row['DOD_HOSP'].year - row['DOB'].year) if pd.notnull(row['DOD_HOSP']) else None, axis=1)

# Filter out unrealistic ages (> 100 years)
merged_df = merged_df[(merged_df['AGE'] >= 0) & (merged_df['AGE'] <= 100)]

# Remove duplicate patient records
merged_df = merged_df.drop_duplicates(subset=['SUBJECT_ID'])

# Get list of all unique pathologies
all_pathologies = merged_df['MAIN_PATHOLOGY'].dropna().unique().tolist()

# Get the top 10 most frequent pathologies
pathology_counts = merged_df['MAIN_PATHOLOGY'].value_counts().head(10).index.tolist()

# ------------------------ Streamlit UI ------------------------

st.title("Analyse des Pathologies et Antécédents")

# Selection mode: Dropdown list or keyword search
search_mode = st.radio("Choisissez votre mode de recherche :", ["Sélection dans la liste", "Recherche par mot-clé"])

if search_mode == "Sélection dans la liste":
    search_word = st.selectbox("Recherchez le nom d'une pathologie complète :", options=["", *pathology_counts, *all_pathologies], index=0)
elif search_mode == "Recherche par mot-clé":
    search_word = st.text_input("Recherchez une pathologie par mot clé :", "")
else:
    search_word = ""

# Filter data based on the selected pathology
filtered_df = merged_df[merged_df["MAIN_PATHOLOGY"].str.contains(search_word, case=False, na=False)] if search_word else merged_df.copy()

# Aggregate data by patient and pathology
aggregated_df = filtered_df.groupby(['SUBJECT_ID', 'MAIN_PATHOLOGY', 'GENDER', 'AGE'], as_index=False).agg({
    'MEDICAL_PERSONAL_HISTORY': lambda x: ', '.join(x.dropna().astype(str)),
    'MEDICAL_FAMILY_HISTORY': lambda x: ', '.join(x.dropna().astype(str))
})

# Select type of medical history to display
option = st.radio("Sélectionnez les antécédents à afficher :", ["Personnels", "Familiaux"])

# ------------------------ Family History Processing ------------------------

def standardize_relation(relation):
    """
    Standardize family relations into general categories.
    
    Args:
        relation (str): Raw relation string.
    
    Returns:
        str: Standardized category or None for non-relevant values.
    """
    relation = relation.lower()
    if any(sub in relation for sub in ["daughter", "son", "children"]):
        return "Children"
    elif any(sub in relation for sub in ["brother", "sister", "sibling"]):
        return "Sibling"
    elif any(sub in relation for sub in ["father", "mother", "parent"]):
        return "Parent"
    elif any(sub in relation for sub in ["non contributory", "nc", "noncontributory", r"no\b.*?\bhistory", "not", "Unknown", "Unremarkable"]):
        return None 
    return "Other relatives"

# Filter family history by selected relations
if option == "Familiaux":
    all_relations = set()
    for family_hist in aggregated_df['MEDICAL_FAMILY_HISTORY'].dropna():
        try:
            family_list = ast.literal_eval(family_hist)
            all_relations.update(standardize_relation(entry.get("relation", "")) for entry in family_list if standardize_relation(entry.get("relation", "")) is not None)
        except:
            continue

    selected_relations = st.multiselect("Sélectionnez les relations à afficher :", sorted(all_relations))

    def filter_family_history(history):
        try:
            family_list = ast.literal_eval(history)
            return [entry["condition"] for entry in family_list if standardize_relation(entry["relation"]) in selected_relations]
        except:
            return []

    display_df = aggregated_df.copy()
    if selected_relations:
        display_df["FILTERED_FAMILY_HISTORY"] = display_df["MEDICAL_FAMILY_HISTORY"].apply(filter_family_history)
        display_df = display_df[['SUBJECT_ID', 'MAIN_PATHOLOGY', 'GENDER', 'AGE', 'FILTERED_FAMILY_HISTORY']]
    else:
        display_df = display_df[['SUBJECT_ID', 'MAIN_PATHOLOGY', 'GENDER', 'AGE', 'MEDICAL_FAMILY_HISTORY']]
else:
    display_df = aggregated_df[['SUBJECT_ID', 'MAIN_PATHOLOGY', 'GENDER', 'AGE', 'MEDICAL_PERSONAL_HISTORY']]

# ------------------------ Word Cloud ------------------------

if option != "Familiaux":
    st.write("### Nuage de mots des antécédents personnels")
    all_histories = display_df['MEDICAL_PERSONAL_HISTORY'].dropna().str.cat(sep=', ').replace("'", "")
else:
    st.write("### Nuage de mots des antécédents familiaux")
    all_histories = display_df['FILTERED_FAMILY_HISTORY'].dropna().explode().str.cat(sep=', ') if 'FILTERED_FAMILY_HISTORY' in display_df else ""

if all_histories:
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_histories)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

# ------------------------ Demographics Visualization ------------------------

st.write("### Distribution démographique des patients atteints de cette pathologie")

if not filtered_df.empty:
    filtered_df['AGE_GROUP'] = pd.cut(filtered_df['AGE'], bins=range(0, 110, 10), right=False, labels=[f"{i}-{i+9}" for i in range(0, 100, 10)])
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=filtered_df, x='AGE_GROUP', hue='GENDER', multiple='stack', palette={"M": "blue", "F": "orange"}, ax=ax)
    ax.set_xlabel("Tranche d'âge")
    ax.set_ylabel("Nombre de patients")
    ax.set_title("Répartition des patients par tranche d'âge et genre")
    st.pyplot(fig)
else:
    st.write("Aucune donnée démographique disponible pour cette pathologie.")

# ------------------------ Display Filtered Data ------------------------

st.write("### Résultats filtrés :")
st.dataframe(display_df)
