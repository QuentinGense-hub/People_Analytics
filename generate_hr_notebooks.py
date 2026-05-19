from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK_DIR = ROOT / "notebooks"


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip("\n").split("\n")],
    }


def code_cell(code: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip("\n").split("\n")],
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


IMPORT_BLOCK = """
from pathlib import Path
import importlib
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "raw").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import hr_analysis_utils as utils
importlib.reload(utils)

MahalanobisAnomalyDetector = utils.MahalanobisAnomalyDetector
random_oversample_minority = utils.random_oversample_minority
SimpleLogisticRegression = utils.SimpleLogisticRegression
attrition_by_group = utils.attrition_by_group
attrition_gap_summary = utils.attrition_gap_summary
coefficient_importance = utils.coefficient_importance
correlation_with_attrition = utils.correlation_with_attrition
dataset_overview = utils.dataset_overview
kpi_table = utils.kpi_table
load_data = utils.load_data
missing_summary = utils.missing_summary
numeric_summary = utils.numeric_summary
plot_bar = utils.plot_bar
plot_box_by_attrition = utils.plot_box_by_attrition
plot_correlation_heatmap = utils.plot_correlation_heatmap
plot_histogram = utils.plot_histogram
plot_top_coefficients = utils.plot_top_coefficients
prepare_model_data = utils.prepare_model_data
prepare_numeric_anomaly_data = utils.prepare_numeric_anomaly_data
pr_auc_score_manual = utils.pr_auc_score_manual
roc_auc_score_manual = utils.roc_auc_score_manual
salary_gap_by_gender = utils.salary_gap_by_gender
average_by_group = utils.average_by_group
top_attrition_segments = utils.top_attrition_segments
find_best_threshold = utils.find_best_threshold
classification_metrics = utils.classification_metrics

DATA_PATH = PROJECT_ROOT / "raw" / "people_analytics_dataset.csv"
df = load_data(DATA_PATH)
pd.set_option("display.max_columns", 100)
"""


NOTEBOOKS = {
    "01_eda_people_analytics.ipynb": notebook(
        [
            md_cell(
                """
# Partie 1 - Analyse Exploratoire des Donnees

Ce notebook documente l'exploration initiale du dataset RH : structure, qualite des donnees, distributions, premiers signaux sur l'attrition et variables potentiellement utiles pour la suite du projet.
"""
            ),
            code_cell(IMPORT_BLOCK),
            md_cell(
                """
## 1. Chargement et vue d'ensemble

On commence par verifier la taille du dataset, le taux d'attrition et la presence de donnees manquantes.
"""
            ),
            code_cell(
                """
dataset_overview(df)
"""
            ),
            code_cell(
                """
df.head()
"""
            ),
            md_cell(
                """
## 2. Variables et qualite de donnees

Le CSV reel contient 28 variables. Par rapport a l'enonce initial, on note notamment la presence de `years_at_company` et `accented_name_flag`, tandis que `salary_band` n'est pas present.
"""
            ),
            code_cell(
                """
missing_summary(df)
"""
            ),
            code_cell(
                """
numeric_summary(df).head(15)
"""
            ),
            md_cell(
                """
## 3. Distribution de la cible

L'attrition est tres desequilibree : environ 2,6 % des collaborateurs quittent l'entreprise. Cette caracteristique aura un impact direct sur l'evaluation du modele.
"""
            ),
            code_cell(
                """
attrition_table = attrition_by_group(df, "attrition", ascending=True)
attrition_table
"""
            ),
            code_cell(
                """
plot_histogram(df, "age", bins=15)
plot_histogram(df, "salary", bins=20)
plot_histogram(df, "engagement_score", bins=20)
"""
            ),
            md_cell(
                """
## 4. Premiers signaux par segment

Cette section cherche a identifier les segments ou le turnover semble plus eleve, avant toute modelisation.
"""
            ),
            code_cell(
                """
attrition_department = attrition_by_group(df, "department")
attrition_country = attrition_by_group(df, "country")
attrition_remote = attrition_by_group(df, "remote_band", ascending=True)

display(attrition_department)
display(attrition_country)
display(attrition_remote)
"""
            ),
            code_cell(
                """
plot_bar(attrition_department, "department", "attrition_rate_pct", title="Taux d'attrition par departement", rotation=25)
plot_bar(attrition_country, "country", "attrition_rate_pct", title="Taux d'attrition par pays")
plot_bar(attrition_remote, "remote_band", "attrition_rate_pct", title="Taux d'attrition par niveau de teletravail", rotation=0)
"""
            ),
            md_cell(
                """
## 5. Variables numeriques et attrition

On compare ensuite plusieurs dimensions RH entre les collaborateurs restes et ceux ayant quitte l'entreprise.
"""
            ),
            code_cell(
                """
attrition_gap_summary(df)
"""
            ),
            code_cell(
                """
plot_box_by_attrition(df, "engagement_score")
plot_box_by_attrition(df, "psychological_safety_score")
plot_box_by_attrition(df, "training_hours")
plot_box_by_attrition(df, "salary")
"""
            ),
            code_cell(
                """
correlation_with_attrition(df)
"""
            ),
            code_cell(
                """
plot_correlation_heatmap(
    df,
    columns=[
        "age",
        "years_at_company",
        "engagement_score",
        "training_hours",
        "absenteeism_days",
        "overtime_hours",
        "psychological_safety_score",
        "salary",
        "attrition",
    ],
    figsize=(9, 7),
)
"""
            ),
            md_cell(
                """
## 6. Faits marquants EDA

- Le dataset couvre **8 020 collaborateurs** et presente un **taux d'attrition faible (2,59 %)**.
- Les principales valeurs manquantes concernent `performance_rating`, `engagement_score` et `salary`, mais dans des proportions limitees.
- Les pays et departements ne sont pas exposes de maniere uniforme : **la France** et **le departement IT** figurent parmi les zones les plus touchees par l'attrition.
- Les collaborateurs partis montrent en moyenne un **engagement plus bas**, une **securite psychologique plus faible**, **moins de formation**, et un peu plus d'absence et d'heures supplementaires.
- Les correlations lineaires avec l'attrition restent faibles, ce qui suggere un signal predictif modere et potentiellement diffus.
"""
            ),
        ]
    ),
    "02_kpi_analyse_rh.ipynb": notebook(
        [
            md_cell(
                """
# Partie 2 - KPI et Analyse RH

Objectif : transformer les donnees brutes en indicateurs RH exploitables pour la decision, avec un angle turnover, engagement, carriere, remuneration et mobilite interne.
"""
            ),
            code_cell(IMPORT_BLOCK),
            md_cell(
                """
## 1. Tableau de bord KPI
"""
            ),
            code_cell(
                """
kpi_table(df)
"""
            ),
            md_cell(
                """
## 2. Turnover

Le turnover global est faible, mais il est utile de l'observer selon plusieurs segments RH pour faire ressortir les zones de vigilance.
"""
            ),
            code_cell(
                """
display(attrition_by_group(df, "department"))
display(attrition_by_group(df, "country"))
display(attrition_by_group(df, "manager_level", ascending=True))
display(attrition_by_group(df, "salary_quartile", ascending=True))
"""
            ),
            code_cell(
                """
plot_bar(attrition_by_group(df, "department"), "department", "attrition_rate_pct", title="Attrition par departement", rotation=25)
plot_bar(attrition_by_group(df, "country"), "country", "attrition_rate_pct", title="Attrition par pays")
"""
            ),
            md_cell(
                """
## 3. Engagement et conditions de travail

On relie ici l'engagement, la securite psychologique, l'absence et les heures supplementaires aux dynamiques de retention.
"""
            ),
            code_cell(
                """
display(attrition_by_group(df, "engagement_band", ascending=True))
display(attrition_by_group(df, "overtime_band", ascending=True))
display(attrition_by_group(df, "absence_band", ascending=True))
display(average_by_group(df, "department", "engagement_score"))
display(average_by_group(df, "department", "psychological_safety_score"))
"""
            ),
            code_cell(
                """
plot_bar(
    average_by_group(df, "department", "engagement_score"),
    "department",
    "avg_engagement_score",
    title="Engagement moyen par departement",
    rotation=25,
)
plot_bar(
    average_by_group(df, "department", "psychological_safety_score"),
    "department",
    "avg_psychological_safety_score",
    title="Securite psychologique moyenne par departement",
    rotation=25,
)
"""
            ),
            md_cell(
                """
## 4. Evolution des carrieres
"""
            ),
            code_cell(
                """
career_table = pd.DataFrame(
    {
        "promotion_rate_pct": df.groupby("department")["promotion_last_3y"].mean() * 100,
        "avg_training_hours": df.groupby("department")["training_hours"].mean(),
        "avg_internal_mobility": df.groupby("department")["internal_mobility_count"].mean(),
    }
).round(2).sort_values("promotion_rate_pct", ascending=False)

career_table
"""
            ),
            code_cell(
                """
plot_bar(career_table.reset_index(), "department", "promotion_rate_pct", title="Taux de promotion par departement", rotation=25)
plot_bar(career_table.reset_index(), "department", "avg_training_hours", title="Heures de formation moyennes par departement", rotation=25)
"""
            ),
            md_cell(
                """
## 5. Structure de remuneration

Cette partie permet d'ouvrir la discussion sur les niveaux de salaire, la progression par niveau hierarchique et les ecarts eventuels entre groupes.
"""
            ),
            code_cell(
                """
salary_by_manager = average_by_group(df, "manager_level", "salary").sort_values("manager_level")
salary_by_manager
"""
            ),
            code_cell(
                """
salary_gap_by_gender(df)
"""
            ),
            code_cell(
                """
plot_bar(salary_by_manager, "manager_level", "avg_salary", title="Salaire moyen par niveau manager", rotation=0)
plot_box_by_attrition(df, "salary")
"""
            ),
            md_cell(
                """
## 6. Mobilite interne
"""
            ),
            code_cell(
                """
mobility_table = pd.DataFrame(
    {
        "mobility_rate_pct": df.groupby("department")["internal_mobility_count"].apply(lambda s: (s > 0).mean() * 100),
        "avg_internal_mobility": df.groupby("department")["internal_mobility_count"].mean(),
        "attrition_rate_pct": df.groupby("department")["attrition"].mean() * 100,
    }
).round(2).sort_values("mobility_rate_pct", ascending=False)

mobility_table
"""
            ),
            code_cell(
                """
plot_bar(mobility_table.reset_index(), "department", "mobility_rate_pct", title="Taux de mobilite interne par departement", rotation=25)
"""
            ),
            md_cell(
                """
## 7. Synthese analytique

- **Effectif** : 8 020 collaborateurs.
- **Attrition globale** : 2,59 %.
- **Engagement moyen** : 68,15 / 100.
- **Securite psychologique moyenne** : 66,19 / 100.
- **Promotion sur 3 ans** : 24,64 %.
- **Mobilite interne** : 63,72 % des collaborateurs ont connu au moins une mobilite.
- **Salaire moyen** : 62 313 par an.

### Points d'attention

- L'attrition est plus elevee en **France**, en **IT** et dans certaines populations managers de niveau 1.
- Les collaborateurs avec **engagement faible** et **fort absentisme** presentent davantage de risque de depart.
- Les collaborateurs partis ont en moyenne **moins de formation**, **un salaire plus faible** et **une securite psychologique plus basse**.
- Le salaire moyen des femmes ressort a un niveau inferieur a celui des hommes dans une lecture brute du dataset. Il faudrait prolonger par une analyse controlee des postes, pays et niveaux.
"""
            ),
        ]
    ),
    "03_modele_attrition.ipynb": notebook(
        [
            md_cell(
                """
# Partie 3 - Machine Learning : prediction de l'attrition

Objectif : construire un premier modele de prediction du depart d'un collaborateur, l'evaluer, identifier les variables contributives et discuter ses limites.

Compte tenu de l'environnement disponible, le modele ci-dessous repose sur une **regression logistique codee en Python/Numpy** plutot que sur `scikit-learn`.
"""
            ),
            code_cell(IMPORT_BLOCK),
            md_cell(
                """
## 1. Preparation des donnees

- exclusion de `employee_id`
- imputation simple des valeurs manquantes
- encodage one-hot des variables qualitatives
- standardisation des variables
- split stratifie train / validation / test
"""
            ),
            code_cell(
                """
prepared = prepare_model_data(df, drop_columns=["employee_id"])

print("Train:", prepared.X_train.shape, prepared.y_train.mean().round(4))
print("Validation:", prepared.X_val.shape, prepared.y_val.mean().round(4))
print("Test:", prepared.X_test.shape, prepared.y_test.mean().round(4))
"""
            ),
            md_cell(
                """
## 2. Entrainement du modele

On utilise une regression logistique avec ponderation des classes pour tenir compte du fort desequilibre de la cible.
"""
            ),
            code_cell(
                """
model = SimpleLogisticRegression(
    learning_rate=0.05,
    epochs=4000,
    reg_strength=0.02,
    class_weight="balanced",
)
model.fit(prepared.X_train, prepared.y_train)

val_scores = model.predict_proba(prepared.X_val)
best_threshold = find_best_threshold(prepared.y_val, val_scores)
best_threshold
"""
            ),
            md_cell(
                """
## 3. Evaluation sur le jeu de test
"""
            ),
            code_cell(
                """
test_scores = model.predict_proba(prepared.X_test)

metrics = classification_metrics(
    prepared.y_test,
    test_scores,
    threshold=best_threshold["threshold"],
)

evaluation = pd.DataFrame(
    [
        ("ROC-AUC", roc_auc_score_manual(prepared.y_test, test_scores)),
        ("PR-AUC", pr_auc_score_manual(prepared.y_test, test_scores)),
        ("Precision", metrics["precision"]),
        ("Recall", metrics["recall"]),
        ("F1-score", metrics["f1_score"]),
        ("Accuracy", metrics["accuracy"]),
        ("Specificite", metrics["specificity"]),
    ],
    columns=["Metrique", "Valeur"],
)

evaluation["Valeur"] = evaluation["Valeur"].round(4)
evaluation
"""
            ),
            code_cell(
                """
pd.DataFrame(
    {
        "Mesure": ["TP", "TN", "FP", "FN"],
        "Valeur": [metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"]],
    }
)
"""
            ),
            md_cell(
                """
## 4. Variables importantes

Les coefficients ci-dessous sont interpretes comme des effets directionnels dans le cadre de cette regression logistique standardisee. Ils ne doivent pas etre lus comme des causalites.
"""
            ),
            code_cell(
                """
importance = coefficient_importance(model, prepared.feature_names, top_n=20)
importance
"""
            ),
            code_cell(
                """
plot_top_coefficients(model, prepared.feature_names, top_n=20)
"""
            ),
            md_cell(
                """
## 5. Experience d'oversampling

On teste ici un **random oversampling** sur le jeu d'entrainement uniquement. L'objectif n'est pas de "creer de l'information", mais de verifier si une meilleure representation de la classe minoritaire aide le modele a detecter davantage de departs.
"""
            ),
            code_cell(
                """
oversample = random_oversample_minority(prepared.X_train, prepared.y_train, seed=42)

pd.DataFrame(
    {
        "Jeu": ["Train original", "Train oversample"],
        "Positifs": [oversample.original_positive_count, oversample.resampled_positive_count],
        "Negatifs": [oversample.original_negative_count, oversample.resampled_negative_count],
    }
)
"""
            ),
            code_cell(
                """
oversampled_model = SimpleLogisticRegression(
    learning_rate=0.05,
    epochs=4000,
    reg_strength=0.02,
    class_weight=None,
)
oversampled_model.fit(oversample.X_resampled, oversample.y_resampled)

oversampled_val_scores = oversampled_model.predict_proba(prepared.X_val)
oversampled_best_threshold = find_best_threshold(prepared.y_val, oversampled_val_scores)
oversampled_test_scores = oversampled_model.predict_proba(prepared.X_test)
oversampled_metrics = classification_metrics(
    prepared.y_test,
    oversampled_test_scores,
    threshold=oversampled_best_threshold["threshold"],
)

comparison = pd.DataFrame(
    [
        (
            "Baseline ponderee",
            roc_auc_score_manual(prepared.y_test, test_scores),
            pr_auc_score_manual(prepared.y_test, test_scores),
            metrics["precision"],
            metrics["recall"],
            metrics["f1_score"],
        ),
        (
            "Random oversampling",
            roc_auc_score_manual(prepared.y_test, oversampled_test_scores),
            pr_auc_score_manual(prepared.y_test, oversampled_test_scores),
            oversampled_metrics["precision"],
            oversampled_metrics["recall"],
            oversampled_metrics["f1_score"],
        ),
    ],
    columns=["Modele", "ROC-AUC", "PR-AUC", "Precision", "Recall", "F1-score"],
).round(4)

comparison
"""
            ),
            md_cell(
                """
### Interpretation attendue

- Si le `recall` augmente, l'oversampling aide le modele a rater moins de departs.
- Si la `precision` baisse fortement, cela signifie qu'il produit davantage de faux positifs.
- Si le gain reste faible sur `PR-AUC` et `F1-score`, on peut conclure que le frein principal est la **faiblesse du signal predictif**, plus que le seul desequilibre de classe.
"""
            ),
            md_cell(
                """
## 6. Experience de detection d'anomalie

Comme l'attrition est rare, on peut aussi tester une logique semi-supervisee : apprendre le profil "normal" des collaborateurs qui restent, puis mesurer a quel point certains profils s'en ecartent.

Ici, on utilise une distance de **Mahalanobis regularisee** sur les variables numeriques, en apprenant le profil de reference uniquement sur les collaborateurs sans attrition du train.
"""
            ),
            code_cell(
                """
numeric_data = prepare_numeric_anomaly_data(df, drop_columns=["employee_id"])

normal_train = numeric_data.X_train[numeric_data.y_train == 0]
anomaly_model = MahalanobisAnomalyDetector(regularization=0.1)
anomaly_model.fit(normal_train)

anomaly_val_scores = anomaly_model.score_samples(numeric_data.X_val)
anomaly_thresholds = np.quantile(anomaly_val_scores, np.linspace(0.70, 0.99, 60))
anomaly_best_threshold = find_best_threshold(
    numeric_data.y_val,
    anomaly_val_scores,
    thresholds=np.unique(anomaly_thresholds),
)
anomaly_test_scores = anomaly_model.score_samples(numeric_data.X_test)
anomaly_metrics = classification_metrics(
    numeric_data.y_test,
    anomaly_test_scores,
    threshold=anomaly_best_threshold["threshold"],
)

pd.DataFrame(
    [
        ("Anomaly score threshold", anomaly_best_threshold["threshold"]),
        ("ROC-AUC", roc_auc_score_manual(numeric_data.y_test, anomaly_test_scores)),
        ("PR-AUC", pr_auc_score_manual(numeric_data.y_test, anomaly_test_scores)),
        ("Precision", anomaly_metrics["precision"]),
        ("Recall", anomaly_metrics["recall"]),
        ("F1-score", anomaly_metrics["f1_score"]),
    ],
    columns=["Metrique", "Valeur"],
).round(4)
"""
            ),
            code_cell(
                """
all_comparison = pd.DataFrame(
    [
        (
            "Baseline ponderee",
            roc_auc_score_manual(prepared.y_test, test_scores),
            pr_auc_score_manual(prepared.y_test, test_scores),
            metrics["precision"],
            metrics["recall"],
            metrics["f1_score"],
        ),
        (
            "Random oversampling",
            roc_auc_score_manual(prepared.y_test, oversampled_test_scores),
            pr_auc_score_manual(prepared.y_test, oversampled_test_scores),
            oversampled_metrics["precision"],
            oversampled_metrics["recall"],
            oversampled_metrics["f1_score"],
        ),
        (
            "Detection d'anomalie",
            roc_auc_score_manual(numeric_data.y_test, anomaly_test_scores),
            pr_auc_score_manual(numeric_data.y_test, anomaly_test_scores),
            anomaly_metrics["precision"],
            anomaly_metrics["recall"],
            anomaly_metrics["f1_score"],
        ),
    ],
    columns=["Approche", "ROC-AUC", "PR-AUC", "Precision", "Recall", "F1-score"],
).round(4)

all_comparison
"""
            ),
            md_cell(
                """
### Lecture critique

- Cette approche est interessante quand la classe positive est rare, car elle ne suppose pas de bien "apprendre" beaucoup de cas de depart.
- En revanche, elle repose sur une hypothese forte : les departs seraient des profils **atypiques** par rapport aux collaborateurs qui restent.
- Si les collaborateurs qui quittent l'entreprise ne sont pas des anomalies nettes mais des cas proches du reste de la population, la methode restera limitee.
"""
            ),
            md_cell(
                """
## 7. Lecture business du modele

Variables associees a un **risque plus eleve** dans cette version du modele :

- anciennete (`years_at_company`) dans certaines configurations
- roles ou pays specifiques comme la France
- absentisme plus eleve

Variables associees a un **risque plus faible** :

- plus d'heures de formation
- meilleure securite psychologique
- salaire plus eleve
- un temps contractuel plus important
- davantage de teletravail dans ce dataset
"""
            ),
            md_cell(
                """
## 8. Analyse critique

### Ce que le modele apporte

- un premier cadre de priorisation des populations a surveiller
- une lecture interpretable grace aux coefficients
- une base solide pour une future industrialisation

### Limites majeures

- la cible est **tres desequilibree** (2,59 % de departs)
- les performances restent **modestes** : le modele detecte un signal faible
- l'oversampling peut ameliorer la detection de la classe minoritaire, mais il ne compense pas l'absence de variables fortement explicatives
- la detection d'anomalie est pertinente comme comparaison, mais elle n'est utile que si les departs se comportent vraiment comme des cas atypiques
- plusieurs variables sont potentiellement sensibles ou discutables d'un point de vue ethique (`gender`, `accented_name_flag`)
- les donnees semblent relativement peu structurees autour d'un signal fort d'attrition, ce qui limite la precision operationnelle

### Conclusion

Le modele est utile comme **outil d'exploration et d'alerte faible**, mais insuffisant a ce stade pour piloter seul des decisions RH individuelles.
"""
            ),
        ]
    ),
    "04_restitution_finale.ipynb": notebook(
        [
            md_cell(
                """
# Partie 4 - Restitution Finale

Notebook redige en vue d'un export PDF. L'objectif est de fournir une restitution complete, argumentee et exploitable pour la prise de decision RH.
"""
            ),
            code_cell(IMPORT_BLOCK),
            md_cell(
                """
## Contexte

La DRH souhaite mieux comprendre les dynamiques de depart, d'engagement, d'evolution de carriere, de remuneration et de mobilite interne au sein d'une entreprise internationale d'environ 8 000 collaborateurs.

Le dataset analyse contient **8 020 collaborateurs** et **28 variables**. L'objectif etait double :

1. transformer les donnees RH en indicateurs de pilotage
2. construire un premier modele de prediction de l'attrition
"""
            ),
            md_cell(
                """
## Faits marquants de l'EDA et de l'analyse RH

### 1. Une attrition globalement faible mais non homogene

- Le taux d'attrition global s'etablit a **2,59 %**.
- Les niveaux les plus eleves apparaissent en **France (3,81 %)** et dans le **departement IT (2,89 %)**.
- Le niveau de teletravail **75 %** ressort egalement comme une modalite plus exposee dans ce dataset.

### 2. L'engagement et la securite psychologique jouent un role de fond

- Les collaborateurs partis ont un **engagement plus faible** que les collaborateurs restes.
- Leur **securite psychologique** est egalement plus basse.
- Les populations avec engagement faible affichent un taux d'attrition superieur aux autres groupes.

### 3. Les conditions de travail donnent des signaux complementaires

- Les collaborateurs partis presentent un peu plus d'**absenteisme** et d'**heures supplementaires**.
- Ils ont aussi recu **moins de formation** en moyenne.

### 4. Carriere, mobilite et remuneration

- Environ **24,64 %** des collaborateurs ont eu une promotion sur 3 ans.
- La **mobilite interne** concerne **63,72 %** des collaborateurs au moins une fois.
- Le **salaire moyen** est de **62 313** par an, avec une progression nette selon le niveau hierarchique.
- Un ecart brut apparait entre salaire moyen des femmes et des hommes, ce qui justifie une analyse plus fine par poste, pays et niveau.
"""
            ),
            code_cell(
                """
kpi_table(df)
"""
            ),
            code_cell(
                """
plot_bar(attrition_by_group(df, "department"), "department", "attrition_rate_pct", title="Attrition par departement", rotation=25)
plot_bar(attrition_by_group(df, "country"), "country", "attrition_rate_pct", title="Attrition par pays")
"""
            ),
            md_cell(
                """
## Resultats du modele predictif

Le modele retenu est une regression logistique interpretable. Il a ete entraine sur des donnees nettoyees et encodees, avec ponderation des classes pour compenser la rarete des departs.
"""
            ),
            code_cell(
                """
prepared = prepare_model_data(df, drop_columns=["employee_id"])
model = SimpleLogisticRegression(learning_rate=0.05, epochs=4000, reg_strength=0.02, class_weight="balanced")
model.fit(prepared.X_train, prepared.y_train)
best_threshold = find_best_threshold(prepared.y_val, model.predict_proba(prepared.X_val))
test_scores = model.predict_proba(prepared.X_test)
test_metrics = classification_metrics(prepared.y_test, test_scores, threshold=best_threshold["threshold"])

pd.DataFrame(
    [
        ("ROC-AUC", roc_auc_score_manual(prepared.y_test, test_scores)),
        ("PR-AUC", pr_auc_score_manual(prepared.y_test, test_scores)),
        ("Recall", test_metrics["recall"]),
        ("Precision", test_metrics["precision"]),
        ("F1-score", test_metrics["f1_score"]),
    ],
    columns=["Metrique", "Valeur"],
).round(4)
"""
            ),
            code_cell(
                """
coefficient_importance(model, prepared.feature_names, top_n=15)
"""
            ),
            md_cell(
                """
### Interpretation

- Les performances du modele restent **modestes**. Il identifie un signal, mais insuffisant pour une decision individuelle fiable.
- Les variables les plus contributives vont dans le sens des analyses descriptives : **formation**, **securite psychologique**, **salaire** et **absenteisme** participent au signal.
- Ce resultat est coherent avec un contexte RH ou les departs sont rares et probablement influences par des facteurs non captures dans le dataset.
"""
            ),
            md_cell(
                """
## Experience complementaire : detection d'anomalie

Une approche alternative a ete testee pour tenir compte de la faible representation de l'attrition : la **detection d'anomalie**. L'idee consiste a apprendre le profil "habituel" des collaborateurs qui restent, puis a identifier les profils qui s'en ecartent le plus.
"""
            ),
            code_cell(
                """
numeric_data = prepare_numeric_anomaly_data(df, drop_columns=["employee_id"])
anomaly_model = MahalanobisAnomalyDetector(regularization=0.1)
anomaly_model.fit(numeric_data.X_train[numeric_data.y_train == 0])
anomaly_val_scores = anomaly_model.score_samples(numeric_data.X_val)
anomaly_thresholds = np.quantile(anomaly_val_scores, np.linspace(0.70, 0.99, 60))
anomaly_best_threshold = find_best_threshold(
    numeric_data.y_val,
    anomaly_val_scores,
    thresholds=np.unique(anomaly_thresholds),
)
anomaly_test_scores = anomaly_model.score_samples(numeric_data.X_test)
anomaly_metrics = classification_metrics(
    numeric_data.y_test,
    anomaly_test_scores,
    threshold=anomaly_best_threshold["threshold"],
)

oversample = random_oversample_minority(prepared.X_train, prepared.y_train, seed=42)
oversampled_model = SimpleLogisticRegression(
    learning_rate=0.05,
    epochs=4000,
    reg_strength=0.02,
    class_weight=None,
)
oversampled_model.fit(oversample.X_resampled, oversample.y_resampled)
oversampled_best_threshold = find_best_threshold(
    prepared.y_val,
    oversampled_model.predict_proba(prepared.X_val),
)
oversampled_test_scores = oversampled_model.predict_proba(prepared.X_test)
oversampled_metrics = classification_metrics(
    prepared.y_test,
    oversampled_test_scores,
    threshold=oversampled_best_threshold["threshold"],
)

pd.DataFrame(
    [
        (
            "Regression logistique ponderee",
            roc_auc_score_manual(prepared.y_test, test_scores),
            pr_auc_score_manual(prepared.y_test, test_scores),
            test_metrics["recall"],
            test_metrics["f1_score"],
        ),
        (
            "Random oversampling",
            roc_auc_score_manual(prepared.y_test, oversampled_test_scores),
            pr_auc_score_manual(prepared.y_test, oversampled_test_scores),
            oversampled_metrics["recall"],
            oversampled_metrics["f1_score"],
        ),
        (
            "Detection d'anomalie",
            roc_auc_score_manual(numeric_data.y_test, anomaly_test_scores),
            pr_auc_score_manual(numeric_data.y_test, anomaly_test_scores),
            anomaly_metrics["recall"],
            anomaly_metrics["f1_score"],
        ),
    ],
    columns=["Approche", "ROC-AUC", "PR-AUC", "Recall", "F1-score"],
).round(4)
"""
            ),
            md_cell(
                """
### Lecture de cette experience

- La detection d'anomalie n'est **pas superieure** au modele supervise en capacite globale de classement.
- En revanche, elle permet de **remonter un peu plus de departs** dans cette experience, au prix d'une precision faible.
- Cette approche peut donc etre utile comme **outil exploratoire complementaire**, mais pas comme solution principale de prediction.
"""
            ),
            md_cell(
                """
## Recommandations

1. Renforcer le suivi des populations a risque dans les segments les plus exposes : France, IT et certaines equipes managers de niveau 1.
2. Utiliser l'engagement, la securite psychologique, l'absenteisme et la charge de travail comme **signaux de prevention** plutot que comme variables de sanction.
3. Cibler davantage les actions de **formation** et de **mobilite interne** pour soutenir la retention.
4. Realiser un audit complementaire sur la **structure de remuneration**, en particulier sur les ecarts par genre, poste, pays et niveau.
5. Enrichir le dataset avec des variables plus proches des causes de depart : historique managerial, changements d'equipe, enquetes qualitatives, intentions de mobilite, etc.
"""
            ),
            md_cell(
                """
## Limites

- Donnees desequilibrees avec peu de cas de depart
- Variables explicatives faiblement correlees a l'attrition
- Certaines variables peuvent poser des questions d'equite et d'ethique
- Analyse effectuee sur un dataset unique, sans validation temporelle

La suite logique serait de completer l'approche quantitative par des entretiens RH et une meilleure historisation des evenements de carriere.
"""
            ),
        ]
    ),
    "05_synthese_executive.ipynb": notebook(
        [
            md_cell(
                """
# Partie 5 - Synthese Executive

Format court, redige pour un comite de direction. Ce notebook peut etre exporte en PDF 1 a 2 pages.
"""
            ),
            code_cell(IMPORT_BLOCK),
            md_cell(
                """
## 1. Ce qu'il faut retenir

- L'entreprise presente un **turnover faible (2,59 %)**, mais avec des poches de risque identifiables.
- Les principaux signaux lies au depart sont un **engagement plus faible**, une **securite psychologique plus basse**, **moins de formation**, et un peu plus d'**absence** et d'**heures supplementaires**.
- Les zones a surveiller en priorite sont la **France**, le **departement IT** et certaines populations managers de niveau 1.
"""
            ),
            code_cell(
                """
kpi_table(df).head(8)
"""
            ),
            md_cell(
                """
## 2. Facteurs cles influencant turnover et engagement

### Turnover

- plus eleve dans certains segments organisationnels
- associe a des conditions d'emploi moins favorables dans ce dataset
- plus difficile a predire a l'echelle individuelle du fait du faible nombre de departs

### Engagement

- plus faible dans les populations qui quittent l'entreprise
- probablement nourri par des leviers de management, de charge de travail, de reconnaissance et de formation
"""
            ),
            md_cell(
                """
## 3. Resultat du modele predictif

Le modele detecte un **signal faible** :

- il est utile pour orienter la vigilance RH
- il ne doit pas etre utilise seul pour prendre des decisions individuelles
- il confirme l'importance de la formation, de la securite psychologique, de la remuneration et de l'absenteisme dans la lecture du risque
- une experience de **detection d'anomalie** a egalement ete testee : elle remonte un peu plus de departs, mais reste globalement moins robuste que le modele supervise
"""
            ),
            code_cell(
                """
prepared = prepare_model_data(df, drop_columns=["employee_id"])
model = SimpleLogisticRegression(learning_rate=0.05, epochs=4000, reg_strength=0.02, class_weight="balanced")
model.fit(prepared.X_train, prepared.y_train)
best_threshold = find_best_threshold(prepared.y_val, model.predict_proba(prepared.X_val))
scores = model.predict_proba(prepared.X_test)

pd.DataFrame(
    {
        "Metrique": ["ROC-AUC", "PR-AUC", "Recall", "Precision"],
        "Valeur": [
            roc_auc_score_manual(prepared.y_test, scores),
            pr_auc_score_manual(prepared.y_test, scores),
            classification_metrics(prepared.y_test, scores, threshold=best_threshold["threshold"])["recall"],
            classification_metrics(prepared.y_test, scores, threshold=best_threshold["threshold"])["precision"],
        ],
    }
).round(4)
"""
            ),
            md_cell(
                """
## 4. Recommandations operationnelles

1. Prioriser les plans d'action RH sur les segments les plus exposes au turnover.
2. Integrer l'engagement, la securite psychologique et l'absenteisme dans un dispositif de veille RH trimestriel.
3. Renforcer les parcours de formation et de mobilite interne comme leviers de retention.
4. Lancer une analyse plus fine de l'equite salariale.
5. Ameliorer la qualite des donnees et enrichir le modele avant toute utilisation plus large.
"""
            ),
        ]
    ),
}


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

    for filename, content in NOTEBOOKS.items():
        target = NOTEBOOK_DIR / filename
        with target.open("w", encoding="utf-8") as handle:
            json.dump(content, handle, ensure_ascii=False, indent=1)
            handle.write("\n")

    print(f"{len(NOTEBOOKS)} notebooks generated in {NOTEBOOK_DIR}")


if __name__ == "__main__":
    main()
