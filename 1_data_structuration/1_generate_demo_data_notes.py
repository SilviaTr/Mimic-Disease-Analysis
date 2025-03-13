import pandas as pd
import random
from datetime import datetime, timedelta

# Générer des valeurs aléatoires pour les colonnes
def random_date(start, end):
    return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

start_date = datetime(2000, 1, 1)
end_date = datetime(2023, 1, 1)

categories = ["Radiology", "Discharge summary", "Physician", "Nursing", "Consult"]
descriptions = ["Chest X-ray", "Final report", "Patient history", "Nursing notes", "Neurology consult"]
text_samples = [
    "Patient admitted with COPD exacerbation and influenza. History of HIV/AIDS, GERD, and hypertension. Reports chronic back pain and previous episodes of leukopenia and anemia.",
    "Patient diagnosed with coronary artery disease. Past medical history includes hypertension, hyperlipidemia, obesity, and osteoarthritis. Family history reveals a father with hardening of the arteries and a paternal aunt with carotid artery disease.",
    "Patient with metastatic melanoma presenting with multiple intraparenchymal hematomas. History of hypertension, hypercholesterolemia, lung cancer, and asthma. Family history includes a mother who suffered a stroke.",
    "Patient presents with mitral valve regurgitation and decreased left ventricular ejection fraction. Previous myocardial infarction, mitral valve prolapse, and hypercholesterolemia. Both parents had a history of myocardial infarction.",
    "Listeria meningitis diagnosed. Past history includes ulcerative colitis, osteopenia, and chronic anemia. Family history shows a brother with ulcerative proctitis and a father with colon polyps.",
    "Subarachnoid hemorrhage due to an Acom aneurysm rupture. Patient has a history of spondylosis, prostate cancer, and chronic lymphocytic leukemia.",
    "Severe COPD exacerbation requiring respiratory support. History includes chronic back pain, previous gastrointestinal bleeding, and a history of substance abuse involving cocaine. Brother diagnosed with throat cancer.",
    "Hypertensive urgency, constipation, and abdominal pain in a patient with severe COPD and bronchiectasis. History of GERD, osteoporosis, and Schatzki’s ring. Family includes a sister with hypertension.",
    "Coronary artery disease with past myocardial infarction and hyperlipidemia. Patient has a history of noninsulin-dependent diabetes, hypothyroidism, and fibromyalgia.",
    "Patient diagnosed with neutropenic fever and community-acquired pneumonia. Past history includes a total colectomy for ulcerative colitis and ileostomy revision. Family history includes a mother with breast cancer and a brother with ulcerative proctitis.",
    "End-stage COPD and HIV/AIDS. History of chronic leukopenia, esophagitis, Schatzki’s ring, and past episodes of SBO obstruction. Brother had a history of lung cancer, mother suffered a cerebrovascular accident.",
    "Severe sepsis due to metastatic gastric adenocarcinoma. Patient has portal vein obstruction, portal hypertension, and biliary obstruction. Past medical history includes gout and esophagitis.",
    "Cardiac arrest secondary to high-degree heart block. History of hypertension, dyslipidemia, and osteoarthritis. Reports chronic low potassium levels and previous trifascicular block diagnosis.",
    "Aspiration pneumonia in a patient with esophageal motility disorder. History of peptic ulcer disease with previous gastrectomy, dementia, and coronary artery disease.",
    "Sepsis with concurrent rectal bleeding post-prostate biopsy. History of coronary artery disease, hyperlipidemia, and gout.",
    "Acute renal failure with MRSA pneumonia and left MCA territory watershed infarct. History of diabetes mellitus, dementia, hypertension, and osteoporosis.",
    "Polytrauma following a pedestrian accident. History of BRCA1 carrier status, previous bilateral mastectomies, and hysterectomy.",
    "Respiratory failure with chronic heart failure and renal failure. History includes severe aortic stenosis, coronary artery disease, nephrectomy due to renal cell carcinoma, and anemia.",
    "Myocardial infarction with past coronary artery bypass grafting. Reports hypertension, diabetes mellitus, hyperlipidemia, and peripheral arterial disease. Family history reveals early coronary disease in the father and multiple siblings.",
    "Chronic systolic heart failure and acute on chronic renal failure. Past medical history includes ventricular tachycardia, COPD, hypertension, and heart failure with reduced ejection fraction. Family history not reported.",
]

# Générer les données de la table NOTEEVENTS
data = []
for i in range(200):
    row_id = i + 1
    subject_id = random.randint(10000, 99999)
    hadm_id = random.randint(1000, 9999)
    chartdate = random_date(start_date, end_date).strftime("%Y-%m-%d")
    charttime = random_date(start_date, end_date).strftime("%H:%M:%S")
    storetime = random_date(start_date, end_date).strftime("%H:%M:%S")
    category = random.choice(categories)
    description = random.choice(descriptions)
    cgid = random.randint(1, 20)
    iserror = random.choice([None, "1"])
    text = random.choice(text_samples)

    data.append([row_id, subject_id, hadm_id, chartdate, charttime, storetime, category, description, cgid, iserror, text])

# Création du DataFrame
columns = ["ROW_ID", "SUBJECT_ID", "HADM_ID", "CHARTDATE", "CHARTIME", "STORETIME", "CATEGORY", "DESCRIPTION", "CGID", "ISERROR", "TEXT"]
df = pd.DataFrame(data, columns=columns)

# Sauvegarde en CSV
csv_filename = "MIMIC_data/NOTEEVENTS_DEMO.csv"
df.to_csv(csv_filename, index=False)


