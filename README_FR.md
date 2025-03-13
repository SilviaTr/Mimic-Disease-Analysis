# README - Analyse des Notes Cliniques

## Présentation du Projet

Ce projet vise à structurer les notes cliniques issues de la base de données MIMIC en utilisant l'API Mistral pour extraire les antécédents médicaux des patients et de leur famille. En transformant des textes non structurés en un format organisé, nous cherchons à identifier les principaux facteurs de risque et les corrélations entre les pathologies. L'analyse suit un pipeline structuré : extraction des données, visualisation des tendances médicales à l'aide de nuages de mots et de graphiques statistiques, puis modélisation pour explorer les associations entre maladies.

## Jeu de Données

Nous avons utilisé la **base de données MIMIC-III**, en particulier la table `NOTESEVENTS`, afin d'extraire les notes cliniques en texte libre. Les données extraites ont été structurées à l'aide de **l'API Mistral** pour obtenir :
- **Pathologie principale** : Diagnostic principal du patient.
- **Antécédents médicaux personnels** : Pathologies antérieures significatives.
- **Antécédents médicaux familiaux** : Conditions médicales présentes chez les proches.

Deux lots de **100 000 patients** ont été traités, et les données structurées ont été stockées sous forme de fichiers CSV.

### ⚠️ Mesures de Sécurité et de Confidentialité des Données

Pour respecter les **réglementations de partage des données MIMIC**, ce dépôt **n'inclut PAS** de données brutes issues de MIMIC. Les mesures de sécurité suivantes ont été mises en place :

- **Aucune donnée réelle de `NOTESEVENTS` n'est incluse**. Un jeu de données de démonstration est fourni, généré via `1_generate_demo_data_notes.py`.
- **La table `PATIENTS.csv` est exclue du dépôt**. En revanche, `PATIENTS_anonymized.csv` est disponible, où les `SUBJECT_ID` ont été **entièrement chiffrés** afin d'empêcher tout lien avec MIMIC.
- **Le jeu de données des notes structurées (`structured_notes.csv`) a été fortement transformé**, ne contenant que les pathologies principales et les antécédents médicaux extraits via l'API Mistral. Les `SUBJECT_ID` y sont également **anonymisés**.
- **Fichiers exclus du dépôt** : `NOTEEVENTS.csv`, `PATIENTS.csv`, `NOTEEVENTS_1.csv`, `NOTEEVENTS_2.csv`, ainsi que `structured_notes.csv` (versions originales avec identifiants réels et notes cliniques non anonymisées).

## Structure du Projet
```
├── 1_data_structuration                 # Scripts de structuration et préparation des données
│   ├── data
│   │   ├── NOTEEVENTS_DEMO_1.csv        # Jeu de données de démonstration
│   │   ├── NOTEEVENTS_DEMO_2.csv        # Second sous-ensemble de démonstration
│   │   ├── structured_notes_anonymized.csv # Notes structurées (anonymisées)
│   ├── 1_generate_demo_data_notes.py    # Script de génération de fausses données `NOTESEVENTS`
│   ├── 2_prepare_medical_notes.py       # Extraction des sous-ensembles de `NOTESEVENTS`
│   ├── 3_medical_notes_structuring.py   # Traitement via l'API Mistral (optimisé multithreading)
│   ├── 4_anonymization.py               # Anonymisation des `SUBJECT_ID`
├── 2_visualization                      # Scripts de visualisation et application Streamlit
│   └── visualization_data.py            # Nuages de mots et graphiques statistiques
├── 3_modelisation                       # Modélisation des données et apprentissage automatique
│   ├── data
│   │   └── CAD_dataset.csv              # Jeu de données pour l'analyse de la maladie coronarienne (CAD)
│   ├── data_preprocessing.py            # Pipeline de nettoyage et de prétraitement des données
│   ├── exploration.py                    # Analyse exploratoire des données (EDA)
│   ├── prediction.py                     # Modèle de machine learning pour tester des hypothèses
├── MIMIC_data                           # Espace réservé aux données MIMIC à accès restreint
│   ├── NOTEEVENTS_DEMO.csv              # Jeu de données de démonstration (remplaçant les données réelles MIMIC)
│   ├── PATIENTS_anonymized.csv          # Données patient anonymisées
├── Images                               # Visualisations des données
│   ├── app.png                          # Capture d'écran de l'application Streamlit
│   ├── distribution_key_risk_factors.png# Distribution des facteurs de risque
│   ├── matrix_correlation.png           # Carte de corrélation
├── README.md                            # Documentation du projet
├── requirements.txt                     # Dépendances Python
```
## Configuration & Installation

### Prérequis
- Python 3.8+
- Bibliothèques requises (installation avec pip) :
  ```bash
  pip install -r requirements.txt
  ```

### Exécution du Pipeline
#### 1. Structuration

##### 1.1-2. Préparer les Notes Médicales
   ```bash
   python 1_data_structuration/2_prepare_medical_notes.py
   ```
   - Extrait et divise ```NOTEEVENTS.csv``` en deux sous-ensembles, chacun contenant 100 000 lignes.
   - Étant donné que l'ensemble de données MIMIC III est très volumineux, la structuration est effectuée uniquement sur une partie de l'ensemble initial.
   - Comme le fichier réel ```NOTEEVENTS.csv``` n'est pas inclus dans ce dépôt en raison des restrictions d'accès aux données MIMIC III, un jeu de données de démonstration a été généré à l'aide de ```1_data_structuration/1_generate_demo_notes.py```. Ce jeu de données reproduit la structure et les caractéristiques des données originales à des fins d'illustration et de test.

   - **⚠️ Remarque Importante**
   Vous pouvez tester entièrement les scripts de structuration et de prétraitement à l'aide de ces fichiers de démonstration, sans nécessiter d'accès au jeu de données restreint.

Si vous souhaitez exécuter la structuration sur les données réelles de MIMIC III, vous devez d'abord demander l'accès via PhysioNet en suivant ces étapes :

- Créez un compte sur PhysioNet : https://physionet.org/login/
  - Complétez le processus de certification requis, y compris la formation sur la confidentialité et la sécurité des données.
- Demandez l'accès à l'ensemble de données MIMIC-III : https://physionet.org/content/mimiciii/
- Une fois approuvé, téléchargez les tables nécessaires (NOTEEVENTS.csv, PATIENTS.csv, etc.) et placez-les dans le répertoire approprié (MIMIC_data/).

##### 1.3. Extraction des Données avec l'API Mistral
  ```bash
  python 1_data_structuration/4_medical_notes_structuring.py
  ```
  - Traite les notes cliniques en utilisant l'API Mistral avec le multithreading pour optimiser le temps de traitement et gérer efficacement de grands volumes de texte clinique.
  - Enregistre la sortie structurée sous forme de fichier CSV dans ```1_data_structuration/data/```, qui sera ensuite utilisé pour l'analyse et la modélisation.

**⚠️ Important** : Avant d'exécuter le script, vous devez remplacer l'espace réservé à la clé API dans ```1_data_structuration/medical_notes_structuring.py``` par votre propre clé :
  ```python 
  API_KEY = "XXXXXXXXXXXXXXXXXXXXXXXXX"  # Remplacez par votre clé API réelle
  ```
  Vous pouvez générer une clé API gratuite en vous inscrivant sur la plateforme officielle Mistral AI
   👉 [Plateforme API Mistral](https://mistral.ai/fr/news/la-plateforme)

##### 1.4 Anonymisation des Données
  ```bash
   python 1_data_structuration/4_anonymization.py
 ```
- Chiffrement de ```SUBJECT_ID``` : Haché avec un algorithme sécurisé et irréversible (ex. SHA-256 tronqué).
- Garantit qu'aucun lien avec le jeu de données original de MIMIC n'est possible.
- Cohérence entre les fichiers : Le même ```SUBJECT_ID``` anonymisé est utilisé dans ```PATIENTS_anonymized.csv``` et ```structured_notes_anonymized.csv```, afin d'assurer la correspondance entre les deux tables pour la phase de modélisation.


#### 2. Visualisation des Données
   ```bash
   streamlit run 2_visualization/visualization.py
   ```
   - Lance l'application Streamlit pour la visualisation des données. Également accessible via ce lien 👉 [mimic-disease-exploration.streamlit.app](https://mimic-disease-exploration.streamlit.app/)

   - Affiche des nuages de mots et des graphiques statistiques afin d'identifier les antécédents médicaux personnels et familiaux fréquemment associés à certaines pathologies.

   ![Alt text](Images/app.png)

   
#### 3. Modélisation
##### 3.1 Prétraitement des Données
   ```bash
   python 3_modelisation/data_preprocessing.py
   ```
   - Préparation des données

##### 3.2 Exploration des Données & Tests d'Hypothèses
   ```bash
   python 3_modelisation/data_exploration.py
   ```
- Effectue une analyse exploratoire des données (EDA).
- Exécute le test statistique du Chi2 pour valider les hypothèses.


##### 3.3 Prédiction de la Maladie Coronarienne (CAD)
   ```bash
   python 3_modelisation/prediction.py
   ```
   - Exécute des modèles de machine learning pour valider la corrélation entre les conditions de la maladie coronarienne (CAD) et les antécédents personnels/familiaux tels que le diabète, l'hypertension, etc.

## Modélisation CAD  

### Méthodologie  

- **Objectif :**  
  L'objectif de cette étude était d'analyser l'influence des **antécédents médicaux** sur une maladie spécifique. Pour maximiser les données disponibles, nous nous sommes concentrés sur la **maladie coronarienne (Coronary Artery Disease - CAD)**, la pathologie la plus fréquente dans notre ensemble de données.  

- **Identification des principaux facteurs de risque :**  
  Grâce à notre **visualisation sous forme de nuage de mots**, nous avons observé que les antécédents médicaux personnels les plus fréquemment associés à la CAD étaient :  
  - **Hypertension**  
  - **Diabète**  
  - **Hyperlipidémie**  
  - **Infarctus du myocarde**  

   ![Texte alternatif](Images/cad_history.png)

  Dans les antécédents médicaux familiaux, **les cas de CAD étaient hautement récurrents**, renforçant la nécessité d’évaluer l’influence de ces facteurs.  

- **Construction du jeu de données :**  
  Pour évaluer l'impact de ces facteurs de risque sur la CAD, nous avons :  
  - Extrait tous les patients diagnostiqués avec la CAD.  
  - Vérifié s’ils présentaient les **facteurs de risque** identifiés précédemment.  
  - Appliqué un **encodage one-hot** pour stocker ces conditions sous forme de variables binaires (1 si présente, 0 sinon).  
    - **Pourquoi l'encodage one-hot ?**  
      - Il permet de convertir les variables catégoriques (ex. présence d’hypertension) en un format numérique adapté aux modèles d’apprentissage automatique.  
      - Il évite d’attribuer une relation ordinale là où il n’y en a pas.  

- **Équilibrage du jeu de données :**  
  - Le jeu de données contenait **12 000 patients atteints de CAD**.  
  - Pour maintenir un **équilibre des classes (répartition 50-50)**, nous avons sélectionné aléatoirement **12 000 patients non atteints de CAD**.  
  - Nous avons enrichi le jeu de données en ajoutant **l'âge et le sexe**, car notre analyse a montré que :  
    - **Les patients âgés de 70 à 90 ans étaient les plus touchés.**  
    - **Les hommes étaient plus souvent diagnostiqués avec la CAD que les femmes.**  
  - Ces variables supplémentaires ont été intégrées en fusionnant le jeu de données avec la **table démographique des patients**.  

   ![Texte alternatif](Images/cad_distribution_age_gender.png)

Pour mieux comprendre le jeu de données, nous avons réalisé une exploration initiale portant sur :  

- **Structure du jeu de données** : affichage des premières lignes, recherche des valeurs manquantes et résumé des statistiques clés.  
- **Prévalence des principaux facteurs de risque** : analyse de la distribution de l’**Hypertension, du Diabète, des Antécédents familiaux de CAD, de l’Hyperlipidémie et de l’Infarctus du myocarde**.  
- **Analyse des corrélations** : examen des relations entre les facteurs de risque et la présence de CAD.  
- **Tests statistiques** : réalisation de **tests du Khi-deux (Chi-square tests)** pour déterminer les associations significatives entre la CAD et les facteurs de risque clés.  

---  
### Résultats de l'exploration  

#### **1. Distribution des principaux facteurs de risque**  
La fréquence des principaux facteurs de risque a été visualisée à l’aide de **diagrammes en barres**, mettant en évidence les points suivants :  

- ***L’hypertension*** est la condition la plus fréquente, avec une proportion significativement plus élevée chez les patients atteints de CAD que chez les non-CAD.  
- ***Le diabète et l’hyperlipidémie*** sont également plus fréquents chez les patients CAD. Bien qu’ils soient très présents dans la population générale, leur prévalence reste légèrement inférieure.  
- ***L’infarctus du myocarde*** est moins représenté dans l’ensemble de données, mais sa présence est nettement plus élevée chez les patients CAD.  
- ***Les antécédents familiaux de CAD*** sont relativement rares, mais bien plus courants chez les patients atteints de CAD, ce qui suggère une composante héréditaire potentielle.  
  - ***Remarque :*** Une grande quantité de données sur les antécédents familiaux n’est pas mentionnée, ce qui indique une sous-déclaration potentielle et un biais possible dans leur représentation.  

![Texte alternatif](Images/distribution_key_risk_factors.png)

#### **2. Répartition de l'âge et du sexe entre les deux populations**  
- **Sexe** : Les patients atteints de CAD sont majoritairement des hommes, alors que la distribution des sexes est plus équilibrée chez les non-CAD. Cela suggère que la CAD est plus fréquente chez les hommes.  
- **Âge** : Les patients atteints de CAD sont le plus souvent âgés de 70 à 80 ans, indiquant que cette tranche d’âge est la plus touchée. Les patients non-CAD présentent une distribution d’âge plus étalée, allant de 50 à 90 ans, sans pic marqué comme chez les patients CAD.  
- **Données manquantes** : Il semble y avoir moins de patients CAD en général, mais cela est en partie dû aux valeurs d’âge manquantes, qui sont 5 % plus fréquentes chez les non-CAD (80,63 %) que chez les CAD (75,71 %). Ces données manquantes doivent être prises en compte lors de l’interprétation des tendances observées.  

![Texte alternatif](Images/cad-non_cad_gender_age.png)

#### **3. Matrice de Corrélation**  
La **carte thermique des corrélations** a mis en évidence que :  
- **L'hypertension, le diabète, l'hyperlipidémie et l'infarctus du myocarde présentent des corrélations modérées avec la présence de la maladie coronarienne (CAD)** (entre **0,25 et 0,35**).  
- **L'âge est également modérément corrélé avec la présence de CAD** (**0,32**). Une part importante des données initiales sur l'ÂGE est manquante dans le jeu de données, ce qui pourrait introduire un biais dans le calcul de la corrélation.  
- **Les antécédents familiaux de CAD ont une corrélation plus faible (~0,20)** mais restent significatifs. Cependant, il est important de noter que de nombreux dossiers d'antécédents familiaux pourraient être absents des notes cliniques, car seule une petite partie des patients avait cette colonne renseignée. Cela pourrait conduire à une sous-estimation de la véritable corrélation, car les cas omis pourraient être pertinents mais non explicitement documentés.  

   ![Texte alternatif](Images/matrix_correlation.png)  

#### **4. Tests du Khi-deux**  
Pour valider statistiquement les associations, nous avons effectué des **tests du Khi-deux**. Le test du Khi-deux est un test non paramétrique spécialement conçu pour évaluer l'association entre des variables catégorielles. Étant donné que la présence de CAD (binaire : 0 ou 1) et les facteurs de risque (binaire : présence ou absence) sont tous deux des variables catégorielles, le test du Khi-deux est bien adapté pour déterminer si la distribution de ces facteurs diffère significativement entre les individus avec et sans CAD. Les résultats montrent que **tous les facteurs de risque ont une relation hautement significative avec la CAD** (**p-value < 0,0001**) :  
- **Hypertension** : χ² = 3267,36, **p-value < 0,0001**  
- **Diabète** : χ² = 1525,07, **p-value < 0,0001**  
- **Antécédents familiaux de CAD** : χ² = 1013,71, **p-value < 0,0001**  
- **Hyperlipidémie** : χ² = 2165,31, **p-value < 0,0001**  
- **Infarctus du myocarde** : χ² = 1984,90, **p-value < 0,0001**  

Une p-value faible (< 0,05) indique que l'association observée est peu susceptible d'être due au hasard, ce qui suggère une forte relation statistique entre le facteur de risque et la CAD. À l'inverse, une p-value élevée (> 0,05) signifierait que l'association observée dans l'échantillon pourrait être due au hasard, indiquant qu'il n'y a pas de relation significative entre les variables.  

Ces résultats confirment que **tous les facteurs de risque sélectionnés sont statistiquement associés à la présence de la CAD, ce qui signifie que les individus présentant ces conditions sont significativement plus susceptibles d'avoir une CAD par rapport à ceux qui n'en souffrent pas.**  

### Méthodologie de Prédiction :  
Pour prédire la présence d'une maladie coronarienne (CAD) en fonction des caractéristiques des patients, nous avons utilisé trois modèles d'apprentissage supervisé :  

- **Decision Tree Classifier** : Un modèle simple mais interprétable, utile pour identifier les principaux facteurs de risque grâce à l'importance des caractéristiques.  
- **Random Forest Classifier** : Une technique d'apprentissage par ensemble qui réduit le sur-apprentissage et améliore la généralisation en entraînant plusieurs arbres de décision.  
- **Gradient Boosting Classifier** : Une approche de boosting qui corrige séquentiellement les erreurs des modèles précédents afin d'optimiser les performances.  

Étant donné la nature structurée du jeu de données (variables binaires et numériques), les modèles basés sur les arbres sont bien adaptés car ils peuvent gérer efficacement les variables catégorielles.  

**1. Prétraitement :**

#### Gestion des valeurs manquantes dans l'AGE ####
- Un nombre significatif de valeurs ```AGE``` était manquant (75-80%), en particulier chez les patients non-CAD.
L'imputation directe avec une valeur unique (moyenne/médiane) pourrait biaiser le jeu de données, car les patients CAD et non-CAD ont des distributions d'âge différentes.
- La solution adoptée a consisté à calculer les pourcentages de distribution de l'âge pour les patients CAD et non-CAD par tranches de 10 ans. Les valeurs manquantes ont été attribuées proportionnellement à la distribution d'âge existante.
- Cela permet d'assurer un remplissage réaliste des valeurs manquantes, en préservant la distribution initiale de l'âge pour chaque classe.

#### Binarisation de la colonne GENDER : #### 
- GENDER était initialement une variable catégorielle ("M" / "F").
- Elle a été convertie en valeurs binaires : 0 pour Femme, 1 pour Homme afin d'être utilisée comme une caractéristique numérique.

**2. Division Train/Test & Mise à l'échelle**

Le jeu de données a été prétraité avant la division afin d'assurer des transformations cohérentes et d'éviter les fuites de données.

- Les valeurs manquantes dans ```AGE``` ont été imputées avant la division pour maintenir une distribution d'âge homogène dans les ensembles d'entraînement et de test.
- Une division stratifiée a été appliquée pour préserver les proportions CAD vs non-CAD dans les deux ensembles.
- La mise à l'échelle a été appliquée uniquement à ```AGE``` après la division, en utilisant **StandardScaler** sur l'ensemble d'entraînement afin d'éviter les fuites de données.
Les variables binaires catégoriques (ex : GENDER) n'ont pas nécessité de mise à l'échelle.

**3. Optimisation des hyperparamètres avec RandomizedSearchCV**
Chaque modèle subit un ajustement des hyperparamètres pour trouver les meilleures configurations.

- **RandomizedSearchCV** explore efficacement l'espace des hyperparamètres en sélectionnant des combinaisons aléatoires plutôt qu'une recherche exhaustive en grille.
- La métrique de scoring utilisée est **AUC-ROC**, idéale pour les problèmes de classification déséquilibrés.

**4. Entraînement et évaluation des modèles**

Chaque modèle a été entraîné sur le jeu de données prétraité et évalué selon plusieurs métriques de performance pour mesurer leur pouvoir prédictif. Les critères d'évaluation incluent :

- **Précision (Accuracy)** : Mesure la justesse globale, mais peut être trompeuse en cas de jeu de données déséquilibré.
- **Précision & Rappel (Precision & Recall)** :
  - La précision (« Precision ») évalue le nombre de cas CAD prédits correctement.
  - Le rappel (« Recall ») mesure la capacité du modèle à identifier les vrais cas CAD.
- **Score F1** : Moyenne harmonique de la précision et du rappel, fournissant une mesure équilibrée des performances du modèle.
- **AUC-ROC** (Aire sous la courbe - Receiver Operating Characteristic) : 
  - Évalue la capacité du modèle à distinguer les cas CAD et non-CAD.
- **Matrice de confusion** :
  - Fournit des informations sur les vrais positifs, vrais négatifs, faux positifs et faux négatifs, ce qui est crucial pour les prédictions médicales.

### Résultat de la Prédiction :

Les trois modèles - Arbre de Décision, Forêt Aléatoire et Gradient Boosting - affichent des performances similaires sur toutes les métriques d'évaluation. Cependant, ces différences peuvent aider à guider la sélection finale du modèle.

1. **Gradient Boosting** : Meilleur Modèle Global
- AUC-ROC le plus élevé (0.8219) → Meilleur pour distinguer les cas de CAD (maladie coronarienne) des non-CAD sur différents seuils de probabilité.
- Précision et Rappel équilibrés (0.73 / 0.75) → Identifie efficacement les patients atteints de CAD tout en limitant les faux positifs.
- Contribution des caractéristiques plus robuste → Comparé aux arbres de décision, il ajuste progressivement l'importance des caractéristiques, réduisant ainsi le biais d'un seul prédicteur.

2. **Forêt Aléatoire :** Fort Rappel & Répartition Plus Équilibrée des Caractéristiques
- Rappel plus élevé (0.77 pour les patients CAD, classe 1) → Capture plus de vrais cas de CAD que les Arbres de Décision ou le Gradient Boosting.
- AUC-ROC (0.8198) proche du Gradient Boosting → Légèrement inférieur, mais reste un modèle solide.
- Écart plus faible entre la précision d'entraînement (74.83%) et celle du test (74.12%), suggérant une bonne généralisation.
- Meilleure distribution de l'importance des caractéristiques → Ne dépend pas excessivement d'une seule variable comme les Arbres de Décision.

**3. Arbre de Décision :** Plus Simple mais Moins Fiable
- Précision compétitive (73.2%) → Bien qu'étant le modèle le plus simple, il ne performe que légèrement moins bien que les autres.
- AUC-ROC plus faible (0.8108) → Légèrement moins efficace pour différencier les patients CAD et non-CAD.
- Importance des caractéristiques biaisée → Dépend excessivement de certaines caractéristiques (ex. : l'hypertension domine la contribution des caractéristiques).
- Plus grand écart entre la précision d'entraînement (76.11%) et celle du test (73.72%), suggérant un léger surapprentissage.

![Texte alternatif](Images/prediction_result.png)

Étant donné ces résultats, le classificateur Gradient Boosting sera choisi pour présenter les résultats finaux, car il offre le meilleur compromis entre précision, rappel et performance globale.

**Matrice de Confusion**

La matrice de confusion pour le Gradient Boosting Classifier montre :

- 1 830 vrais positifs (cas CAD correctement prédits).
- 1 837 vrais négatifs (cas non-CAD correctement prédits).
- 640 faux négatifs (cas CAD manqués).
- 647 faux positifs (classés à tort comme CAD).

Cela suggère que le modèle est légèrement plus efficace pour détecter la maladie coronarienne (faux négatif - 25%) que pour éviter les erreurs de classification des cas non coronariens (faux positif - 27%), ce qui est préférable dans un contexte clinique, mais peut encore être amélioré.

![Texte alternatif](Images/confusion_matrix.png)

**Courbe ROC & AUC**

La courbe ROC du modèle Gradient Boosting indique :
- AUC = 0.8219, ce qui signifie que le modèle a un bon pouvoir discriminant.
- Le modèle performe bien pour différencier les cas CAD et non-CAD, mais il n'est pas parfait.

![Texte alternatif](Images/ROC.png)

**Importance des Variables**

Les facteurs les plus influents dans la prédiction de la coronaropathie (CAD) incluent :

- Hypertension (prédicteur le plus fort)
- Âge
- Infarctus du myocarde
- Hyperlipidémie
- Diabète
- Antécédents familiaux de CAD (informations potentiellement manquantes)
- Genre (impact le plus faible)

![Texte alternatif](Images/var_importance.png)

Ces résultats sont en accord avec les analyses exploratoires initiales, confirmant que les patients souffrant d'hypertension, d'infarctus du myocarde et d'un âge avancé présentent un risque plus élevé de CAD.

### Conclusion :
- Le Gradient Boosting offre les meilleures performances et est utilisé pour les résultats finaux.
- L'hypertension et l'infarctus du myocarde sont des prédicteurs cliniques clés de la CAD.
- Les améliorations futures pourraient inclure :
  - La collecte de davantage de données pour réduire les valeurs manquantes (âge, antécédents familiaux, etc.).
  - La restriction de l'étude à un groupe d'âge homogène (ex. : 60-80 ans) afin de minimiser les biais liés à l'âge et de mieux identifier les véritables facteurs de risque médicaux.
  - L'exploration de facteurs médicaux supplémentaires au-delà du jeu de données, tels que les facteurs de mode de vie (tabagisme, alimentation, activité physique) ou les facteurs socio-économiques (accès aux soins de santé, niveaux de stress).

---

Pour toute question ou contribution, n'hésitez pas à nous contacter.