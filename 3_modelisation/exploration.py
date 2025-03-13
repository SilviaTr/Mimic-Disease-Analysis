import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from pathlib import Path

# ------------------------ Data Loading ------------------------

df = pd.read_csv(Path(__file__).parent / "data" / "CAD_dataset.csv")

# ------------------------ Data Exploration ------------------------


def explore_data(df: pd.DataFrame):
    """
    Display basic data information including missing values and descriptive statistics.

    Args:
        df (pd.DataFrame): Dataset to explore.
    """
    print("First five rows of the dataset:")
    print(df.head())

    print("\nMissing values per column:")
    print(df.isnull().sum())

    print("\n Frequency distribution for categorical variables:")
    categorical_vars = ["GENDER"]
    for col in categorical_vars:
        print(f"\n{col} distribution:")
        print(df[col].value_counts(dropna=False))

    print("\n Frequency distribution for binary variables:")
    binary_vars = ["CAD_PRESENT", "HYPERTENSION", "DIABETES", "FAMILY_HISTORY_CAD", "HYPERLIPIDEMIA", "MYOCARDIAL_INFARCTION"]
    for col in binary_vars:
        counts = df[col].value_counts(normalize=True) * 100
        print(f"\n{col} (n, %):")
        print(df[col].value_counts(), "\n")  
        print(counts.round(2).astype(str) + "%") 

    print("\n Descriptive statistics for numerical variables:")
    numeric_vars = ["AGE"]
    print(df[numeric_vars].describe())


explore_data(df)

# ------------------------ Data Visualization ------------------------

def count_missing_ages(df: pd.DataFrame):
    """
    Counts missing values in the AGE column for CAD and non-CAD patients.

    Args:
        df (pd.DataFrame): Dataset containing "AGE" and "CAD_PRESENT".
    """
    missing_cad = df[df["CAD_PRESENT"] == 1]["AGE"].isna().sum()
    missing_non_cad = df[df["CAD_PRESENT"] == 0]["AGE"].isna().sum()

    total_cad = len(df[df["CAD_PRESENT"] == 1])
    total_non_cad = len(df[df["CAD_PRESENT"] == 0])

    print(f" Missing AGE values for CAD patients: {missing_cad} / {total_cad} ({(missing_cad / total_cad) * 100:.2f}%)")
    print(f" Missing AGE values for Non-CAD patients: {missing_non_cad} / {total_non_cad} ({(missing_non_cad / total_non_cad) * 100:.2f}%)")

count_missing_ages(df)

def plot_age_gender_cad_distribution(df: pd.DataFrame):
    """
    Plot the distribution of age and gender in stacked bars, separated by CAD presence.

    Args:
        df (pd.DataFrame): Dataset containing "AGE", "GENDER", and "CAD_PRESENT".
    """
    # Définir les tranches d'âge
    bins = [40, 50, 60, 70, 80, 90, 100]
    labels = ["40-49", "50-59", "60-69", "70-79", "80-89", "90-99"]
    df["AGE_GROUP"] = pd.cut(df["AGE"], bins=bins, labels=labels, right=False)

    # Séparer les données en fonction de CAD_PRESENT
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for i, cad_status in enumerate([0, 1]):
        subset = df[df["CAD_PRESENT"] == cad_status]
        sns.histplot(
            data=subset, x="AGE_GROUP", hue="GENDER", multiple="stack",
            palette={"M": "lightblue", "F": "peachpuff"}, shrink=0.8, ax=axes[i]
        )
        axes[i].set_title(f"Patients {'sans' if cad_status == 0 else 'avec'} CAD")
        axes[i].set_xlabel("Tranche d'âge")
        axes[i].set_ylabel("Nombre de patients" if i == 0 else "")
        axes[i].legend(title="GENDER")

    plt.suptitle("Répartition des patients par tranche d'âge, genre et CAD")
    plt.tight_layout()
    plt.show()

plot_age_gender_cad_distribution(df)

def plot_risk_factor_distributions(df: pd.DataFrame):
    """
    Plot the distribution of risk factors: Hypertension, Diabetes, Family History of CAD,
    Hyperlipidemia, and Myocardial Infarction, differentiated by CAD presence.

    Args:
        df (pd.DataFrame): Dataset containing risk factor columns and CAD indicator.
    """
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    risk_factors = ["HYPERTENSION", "DIABETES", "FAMILY_HISTORY_CAD", "HYPERLIPIDEMIA", "MYOCARDIAL_INFARCTION"]

    for i, col in enumerate(risk_factors):
        sns.countplot(data=df, x=col, hue="CAD_PRESENT", ax=axes[i], palette="Set2")

        axes[i].set_title(f"{col}")
        axes[i].set_xlabel("")  # Remove redundant x-axis label

        # Fix the issue by explicitly setting the ticks before labeling
        axes[i].set_xticks([0, 1])  
        axes[i].set_xticklabels(["Absent", "Présent"])

    plt.tight_layout()
    plt.show()


plot_risk_factor_distributions(df)


def plot_correlation_matrix(df: pd.DataFrame):
    """
    Plot a heatmap showing the correlation between numerical variables.

    Args:
        df (pd.DataFrame): Dataset containing numeric columns.
    """
    numeric_cols = df.select_dtypes(include=["number"])
    corr_matrix = numeric_cols.corr()

    plt.figure(figsize=(8, 6))

    # Ajuster les couleurs pour que les valeurs entre 0.3 et 0.4 ressortent
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    # Définir une norme pour étirer la palette de couleurs autour des valeurs 0.3-0.4
    norm = plt.Normalize(vmin=0, vmax=0.4)

    sns.heatmap(corr_matrix, annot=True, cmap=cmap, fmt=".2f", center=0, norm=norm)

    plt.title("Matrice de corrélation")
    plt.show()


plot_correlation_matrix(df)

# ------------------------ Statistical Tests ------------------------


def chi_square_tests(df: pd.DataFrame):
    """
    Perform Chi-square tests for categorical variables against CAD presence.

    Args:
        df (pd.DataFrame): Dataset containing categorical variables and CAD presence.
    """
    risk_factors = ["HYPERTENSION", "DIABETES", "FAMILY_HISTORY_CAD", "HYPERLIPIDEMIA", "MYOCARDIAL_INFARCTION"]

    for col in risk_factors:
        contingency_table = pd.crosstab(df[col], df["CAD_PRESENT"])
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        print(f"\nTest du khi-deux pour {col} :\nChi2 = {chi2:.2f}, p-value = {p:.4f}")


chi_square_tests(df)

print("\nExploration terminée. Prêt pour l'analyse statistique et la modélisation ! 🚀")
