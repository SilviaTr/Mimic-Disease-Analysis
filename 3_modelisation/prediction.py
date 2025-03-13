import warnings
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path

import numpy as np

interp = np.interp

warnings.filterwarnings("ignore")

# ------------------------ Data Loading & Preprocessing ------------------------


def impute_age_using_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing AGE values by sampling from the observed distribution of age groups
    based on CAD presence.

    Args:
        df (pd.DataFrame): The dataset containing AGE and CAD_PRESENT columns.

    Returns:
        pd.DataFrame: The dataset with imputed AGE values.
    """

    bins = list(range(40, 101, 10)) 
    labels = [f"{b}-{b+9}" for b in bins[:-1]]

    df["AGE_GROUP"] = pd.cut(df["AGE"], bins=bins, labels=labels, include_lowest=True)

    age_distribution_cad = df[df["CAD_PRESENT"] == 1]["AGE_GROUP"].value_counts(normalize=True).to_dict()
    age_distribution_non_cad = df[df["CAD_PRESENT"] == 0]["AGE_GROUP"].value_counts(normalize=True).to_dict()

    def sample_age(cad_present):
        dist = age_distribution_cad if cad_present == 1 else age_distribution_non_cad
        age_group = np.random.choice(list(dist.keys()), p=list(dist.values()))
        return np.random.randint(int(age_group.split("-")[0]), int(age_group.split("-")[1]) + 1)

    df.loc[df["AGE"].isna(), "AGE"] = df[df["AGE"].isna()]["CAD_PRESENT"].apply(sample_age)
    df.drop(columns=["AGE_GROUP"], inplace=True)
    return df

def load_and_preprocess_data(filepath: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(filepath)
    df["GENDER"] = df["GENDER"].map({"F": 0, "M": 1})
    df = impute_age_using_distribution(df)
    X = df[["HYPERTENSION", "DIABETES", "FAMILY_HISTORY_CAD", "HYPERLIPIDEMIA", "MYOCARDIAL_INFARCTION", "GENDER", "AGE"]]
    y = df["CAD_PRESENT"]
    return X, y


X, y = load_and_preprocess_data(Path(__file__).parent / "data" / "CAD_dataset.csv")

# ------------------------ Train/Test Split & Scaling ------------------------


def split_and_scale(X: pd.DataFrame, y: pd.Series, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train["AGE"] = scaler.fit_transform(X_train[["AGE"]])
    X_test["AGE"] = scaler.transform(X_test[["AGE"]])
    return X_train, X_test, y_train, y_test, scaler


X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

# ------------------------ Hyperparameter Tuning with RandomizedSearchCV ------------------------


def tune_model(model, param_grid, X_train, y_train, n_iter=10, cv=3):
    grid_search = RandomizedSearchCV(
        model, param_grid, n_iter=n_iter, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=1, random_state=42
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


param_tree = {"max_depth": [3, 5, 10], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 3, 5]}
param_rf = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 3],
}
param_gb = {
    "n_estimators": [100, 200],
    "learning_rate": [0.01, 0.1],
    "max_depth": [3, 5],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 3],
}

best_tree, best_tree_params = tune_model(DecisionTreeClassifier(random_state=42), param_tree, X_train, y_train)
best_rf, best_rf_params = tune_model(RandomForestClassifier(random_state=42), param_rf, X_train, y_train)
best_gb, best_gb_params = tune_model(GradientBoostingClassifier(random_state=42), param_gb, X_train, y_train)
# ------------------------ Model Evaluation ------------------------


def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name: str):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n {model_name}")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Matrice de confusion - {model_name}")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc_score(y_test, y_prob):.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs")
    plt.title(f"Courbe ROC - {model_name}")
    plt.legend()
    plt.show()


train_and_evaluate(best_tree, X_train, X_test, y_train, y_test, "Arbre de Décision Optimisé")
train_and_evaluate(best_rf, X_train, X_test, y_train, y_test, "Random Forest Optimisé")
train_and_evaluate(best_gb, X_train, X_test, y_train, y_test, "Gradient Boosting Optimisé")

# ------------------------ Feature Importance ------------------------


def plot_feature_importance(model, X, title: str):
    """
    Plot feature importance in descending order and add a reference line for mean importance.

    Args:
        model: Trained model with feature importances.
        X (pd.DataFrame): Feature matrix.
        title (str): Title of the plot.
    """
    feature_importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=feature_importance["Feature"], y=feature_importance["Importance"], palette="Blues_r")

    plt.title(title)
    plt.ylabel("Importance")
    plt.xlabel("Features")
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()


plot_feature_importance(best_tree, X, "Importance des variables - Arbre de Décision Optimisé")
plot_feature_importance(best_rf, X, "Importance des variables - Random Forest Optimisé")
plot_feature_importance(best_gb, X, "Importance des variables - Gradient Boosting Optimisé")


def display_accuracy(model, X_train, X_test, y_train, y_test, model_name: str):
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))

    print(f"{model_name} Accuracy:")
    print(f"Train: {train_accuracy:.4f}")
    print(f"Test: {test_accuracy:.4f}\n")

    plt.bar(["Train", "Test"], [train_accuracy, test_accuracy], color=["blue", "orange"])
    plt.ylim(0, 1)
    plt.title(f"Accuracy on Train vs Test - {model_name}")
    plt.ylabel("Accuracy")
    plt.show()


display_accuracy(best_tree, X_train, X_test, y_train, y_test, "Arbre de Décision Optimisé")
display_accuracy(best_rf, X_train, X_test, y_train, y_test, "Random Forest Optimisé")
display_accuracy(best_gb, X_train, X_test, y_train, y_test, "Gradient Boosting Optimisé")
