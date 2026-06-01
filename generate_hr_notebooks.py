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
add_derived_columns = utils.add_derived_columns
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
raw_df = pd.read_csv(DATA_PATH)
duplicate_employee_ids = int(raw_df["employee_id"].duplicated().sum()) if "employee_id" in raw_df.columns else 0
source_row_count = len(raw_df)
df = add_derived_columns(raw_df.drop_duplicates(subset=["employee_id"]).copy())
clean_row_count = len(df)
pd.set_option("display.max_columns", 100)
"""


NOTEBOOKS = {
    "01_eda_people_analytics.ipynb": notebook(
        [
            md_cell(
                """
# Partie 1 - Analyse Exploratoire des Donnees

Ce notebook repart du dataset actuel et documente l'exploration initiale des donnees RH : structure, qualite, variables disponibles, premiers signaux sur l'attrition et points de vigilance pour la suite du projet.
"""
            ),
            code_cell(IMPORT_BLOCK),
            md_cell(
                """
## 1. Chargement et vue d'ensemble

On commence par verifier la taille du dataset, le taux d'attrition et les principaux points de qualite de donnees.
"""
            ),
            code_cell(
                """
dataset_overview(df)
"""
            ),
            code_cell(
                """
pd.DataFrame(
    [
        ("Lignes source", source_row_count),
        ("Lignes apres dedoublonnage employee_id", clean_row_count),
        ("Doublons employee_id supprimes", duplicate_employee_ids),
    ],
    columns=["Controle", "Valeur"],
)
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

Le CSV reel contient 28 variables source. Par rapport a l'enonce initial, deux variables meritent une attention particuliere :

- `years_at_company`, utile pour analyser l'anciennete et la retention
- `accented_name_flag`, variable sensible a traiter avec prudence d'un point de vue ethique

Le champ `salary_band` n'est pas present dans le fichier.
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
            code_cell(
                """
display(attrition_by_group(df, "accented_name_flag", ascending=False))

years_at_company_quartile = pd.qcut(df["years_at_company"], 4, duplicates="drop")
(df.groupby(years_at_company_quartile, observed=False)["attrition"].mean() * 100).round(2)
"""
            ),
            md_cell(
                """
## 3. Distribution de la cible

L'attrition represente environ **18,8 %** des effectifs apres dedoublonnage. La classe positive reste minoritaire, mais le niveau de desequilibre est compatible avec une analyse supervisee exploitable.
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
top_segments = top_attrition_segments(df, ["department", "country"], min_count=120, top_n=10)

display(attrition_department)
display(attrition_country)
display(attrition_remote)
display(top_segments)
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

- Le dataset source contient **8 020 lignes**, dont **20 doublons exacts d'`employee_id`**. L'analyse repose donc sur **8 000 collaborateurs uniques**.
- Les principales valeurs manquantes concernent `performance_rating`, `engagement_score` et `salary`, mais dans des proportions limitees.
- Les pays et departements ne sont pas exposes de maniere uniforme : **l'Espagne** et **la France** sont les pays les plus touches, tandis que **HR** et **Finance** ressortent comme les departements les plus exposes.
- Les collaborateurs partis montrent en moyenne un **engagement plus bas**, une **securite psychologique beaucoup plus faible**, **davantage d'absences**, **plus d'heures supplementaires**, **moins de mobilite interne** et une **anciennete plus faible**.
- Les variables `years_at_company` et `accented_name_flag`, absentes du brief initial, apportent une lecture complementaire. L'anciennete plus faible est associee a davantage d'attrition, tandis que `accented_name_flag` devra etre traite avec prudence dans l'analyse et surtout dans le modele.
"""
            ),
            md_cell(
                """
## 7. Validation d'hypotheses complementaires

Cette section teste plusieurs hypotheses d'analyse plus fines sur le dataset actuel. L'objectif est de verifier ce qui est vraiment confirme par les donnees, ce qui reste nuance, et ce qui ne doit pas etre surinterprete.
"""
            ),
            code_cell(
                """
visibility_by_remote = df.groupby("remote_ratio")["visibility_score"].mean().round(2)
promotion_by_remote = (df.groupby("remote_ratio")["promotion_last_3y"].mean() * 100).round(2)

age_mobility_rows = []
for label, mask_age in {
    "22-30": df["age"].between(22, 30),
    "31-40": df["age"].between(31, 40),
    "41+": df["age"] >= 41,
}.items():
    for mobility_label, mobility_mask in {
        "0 mobilite": df["internal_mobility_count"] == 0,
        "1+ mobilite": df["internal_mobility_count"] >= 1,
    }.items():
        subset = df[mask_age & mobility_mask]
        age_mobility_rows.append(
            (
                label,
                mobility_label,
                len(subset),
                round(subset["attrition"].mean() * 100, 2),
            )
        )

age_mobility_table = pd.DataFrame(
    age_mobility_rows,
    columns=["Age", "Mobilite", "Effectif", "Attrition (%)"],
)

parents_ot_table = pd.DataFrame(
    [
        (
            "Meres, OT >= 30h",
            len(df[(df["parental_status"] == "Parent") & (df["gender"] == "Female") & (df["overtime_hours"] >= 30)]),
            round(df[(df["parental_status"] == "Parent") & (df["gender"] == "Female") & (df["overtime_hours"] >= 30)]["attrition"].mean() * 100, 2),
            round(df[(df["parental_status"] == "Parent") & (df["gender"] == "Female") & (df["overtime_hours"] >= 30)]["engagement_score"].mean(), 2),
        ),
        (
            "Peres, OT >= 30h",
            len(df[(df["parental_status"] == "Parent") & (df["gender"] == "Male") & (df["overtime_hours"] >= 30)]),
            round(df[(df["parental_status"] == "Parent") & (df["gender"] == "Male") & (df["overtime_hours"] >= 30)]["attrition"].mean() * 100, 2),
            round(df[(df["parental_status"] == "Parent") & (df["gender"] == "Male") & (df["overtime_hours"] >= 30)]["engagement_score"].mean(), 2),
        ),
    ],
    columns=["Segment", "Effectif", "Attrition (%)", "Engagement moyen"],
)

high_safe = df["psychological_safety_score"] >= 60
low_safe = df["psychological_safety_score"] < 40
high_ot = df["overtime_hours"] >= 25
low_ot = df["overtime_hours"] < 25

ot_safety_table = pd.DataFrame(
    [
        ("OT eleve + safety haute", len(df[high_ot & high_safe]), round(df[high_ot & high_safe]["attrition"].mean() * 100, 2)),
        ("OT faible + safety basse", len(df[low_ot & low_safe]), round(df[low_ot & low_safe]["attrition"].mean() * 100, 2)),
        ("OT eleve + safety basse", len(df[high_ot & low_safe]), round(df[high_ot & low_safe]["attrition"].mean() * 100, 2)),
    ],
    columns=["Segment", "Effectif", "Attrition (%)"],
)

display(pd.DataFrame({"visibility_score_moyen": visibility_by_remote, "promotion_rate_pct": promotion_by_remote}))
display(age_mobility_table)
display(parents_ot_table)
display(ot_safety_table)
"""
            ),
            md_cell(
                """
### Lecture des hypotheses

#### Confirme

- **Overtime x securite psychologique** : la securite psychologique joue bien un role moderateur majeur. Avec un decoupage plausible (`OT >= 25`, `safety >= 60`, `safety < 40`), on observe environ **6 %** d'attrition en `OT eleve + safety haute` contre **41-43 %** dans les combinaisons a `safety basse`.
- **Age x mobilite interne** : l'effet mobilite est tres robuste. Sans mobilite, l'attrition est autour de **32-36 %** selon l'age ; avec `1+` mobilite, elle tombe autour de **9-10 %** pour tous les groupes.
- **Remote -> visibilite** : la visibilite baisse de facon quasi lineaire avec le remote, de **72,3** en presentiel a **57,6** en full remote.
- **Parents + fortes heures supplementaires** : chez les parents avec `OT >= 30h`, l'attrition atteint environ **36,5 %** chez les femmes contre **17,1 %** chez les hommes, tandis que l'engagement reste proche dans les deux groupes.

#### Partiellement confirme

- **Remote x engagement x attrition** : le signal existe surtout en hybride. En full remote, l'ecart d'attrition selon l'engagement est faible.
- **PhD x salaire bas** : le couple `haut niveau de diplome + bas salaire` est bien plus expose, mais le phenomene n'est pas specifique aux PhD.
- **Haute performance x non promotion** : la promotion protege clairement, mais l'absence de promotion ne cree pas un risque extremement plus fort chez les top performers que chez les autres.

#### A ne pas surinterpreter

- **Visibilite x mobilite** : la mobilite explique beaucoup plus l'attrition que la visibilite.
- **Remote x promotion** : le remote reduit nettement la visibilite, mais ne se traduit pas ici par un effondrement du taux de promotion.
- **Formation x niveau hierarchique -> performance** : le signal est faible dans ce dataset, donc cet axe ne doit pas etre presente comme un resultat fort.
"""
            ),
        ]
    ),
    "02_kpi_analyse_rh.ipynb": notebook(
        [
            md_cell(
                """
# Partie 2 - KPI et Analyse RH

Objectif : transformer le dataset nettoye en indicateurs RH exploitables pour la decision, avec un angle turnover, engagement, carriere, remuneration et mobilite interne.
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

Le turnover est significatif. Il est utile de l'observer selon plusieurs segments RH pour faire ressortir les zones de vigilance et les priorites d'action.
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

Cette partie met particulierement a profit les variables `years_at_company`, `internal_mobility_count` et `promotion_last_3y`, qui n'etaient pas toutes explicites dans l'enonce initial mais deviennent centrales dans le dataset actuel.
"""
            ),
            code_cell(
                """
career_table = pd.DataFrame(
    {
        "promotion_rate_pct": df.groupby("department")["promotion_last_3y"].mean() * 100,
        "avg_training_hours": df.groupby("department")["training_hours"].mean(),
        "avg_internal_mobility": df.groupby("department")["internal_mobility_count"].mean(),
        "avg_years_at_company": df.groupby("department")["years_at_company"].mean(),
    }
).round(2).sort_values("promotion_rate_pct", ascending=False)

career_table
"""
            ),
            code_cell(
                """
plot_bar(career_table.reset_index(), "department", "promotion_rate_pct", title="Taux de promotion par departement", rotation=25)
plot_bar(career_table.reset_index(), "department", "avg_training_hours", title="Heures de formation moyennes par departement", rotation=25)
plot_bar(career_table.reset_index(), "department", "avg_years_at_company", title="Anciennete moyenne par departement", rotation=25)
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

- **Effectif analyse** : 8 000 collaborateurs uniques.
- **Attrition globale** : 18,77 %.
- **Engagement moyen** : 67,79 / 100.
- **Securite psychologique moyenne** : 53,72 / 100.
- **Promotion sur 3 ans** : 23,88 % environ.
- **Mobilite interne** : 61,91 % des collaborateurs ont connu au moins une mobilite.
- **Salaire moyen** : 62 314 environ par an.

### Points d'attention

- L'attrition est plus elevee en **Espagne**, en **France**, ainsi que dans les departements **HR**, **Finance** et **Sales**.
- Les collaborateurs avec **fort absentisme**, **heures supplementaires elevees** et **engagement plus faible** presentent davantage de risque de depart.
- Les collaborateurs partis ont en moyenne une **securite psychologique beaucoup plus basse**, **moins de mobilite interne**, **un salaire plus faible** et **moins d'anciennete**.
- L'anciennete, la promotion recente et la mobilite interne apparaissent comme des dimensions RH particulierement structurantes dans ce dataset.
- Le salaire moyen des femmes ressort a un niveau inferieur a celui des hommes dans une lecture brute du dataset. Il faudrait prolonger par une analyse controlee des postes, pays et niveaux.
"""
            ),
            md_cell(
                """
## 8. Validation de KPI et signaux croises

Cette section consolide quelques hypotheses utiles au pilotage RH, en reliant plusieurs dimensions a la fois.
"""
            ),
            code_cell(
                """
validation_table = pd.DataFrame(
    [
        ("OT x safety psychologique", "Confirme", "La safety psychologique modere fortement l'effet de la surcharge ; safety basse = attrition tres elevee meme avec OT non extreme."),
        ("Age x mobilite interne", "Confirme", "L'absence de mobilite fait monter l'attrition a plus de 30 % quel que soit l'age ; avec mobilite, elle retombe autour de 9-10 %."),
        ("Remote -> visibilite", "Confirme", "La visibilite baisse regulierement quand le remote augmente."),
        ("Parents x OT x genre", "Confirme", "Chez les parents en forte surcharge, l'attrition des femmes est environ deux fois plus elevee que celle des hommes."),
        ("Remote x engagement", "Partiel", "L'effet de l'engagement sur l'attrition est plus visible en hybride qu'en full remote."),
        ("PhD x bas salaire", "Partiel", "Le risque existe, mais il n'est pas reserve aux profils PhD."),
        ("Visibilite x mobilite", "Nuance", "La mobilite interne explique bien plus les departs que la visibilite seule."),
        ("Remote x promotion", "Nuance", "Le remote degrade la visibilite, mais l'effet sur la promotion reste limite dans ce dataset."),
    ],
    columns=["Hypothese", "Statut", "Lecture KPI"],
)

validation_table
"""
            ),
            md_cell(
                """
### Implications RH

- La **mobilite interne** ressort comme un levier de retention tres transversal, au-dela des differences d'age ou de visibilite.
- La **securite psychologique** doit etre suivie comme un indicateur avance de risque, en particulier dans les contextes de surcharge.
- Les analyses croisees suggerent aussi une vigilance specifique sur les **parents en forte charge de travail**, avec un possible enjeu d'equite de genre.
- La **visibilite** est un bon indicateur d'environnement de travail, mais pas un determinant direct des departs dans ce dataset.
"""
            ),
        ]
    ),
    "03_modele_attrition.ipynb": notebook(
        [
            md_cell(
                """
# Partie 3 - Machine Learning : prediction de l'attrition

Objectif : construire un modele de prediction du depart d'un collaborateur a partir du dataset actuel nettoye, l'evaluer, identifier les variables contributives et discuter ses limites.

Compte tenu de l'environnement disponible, le modele ci-dessous repose sur une **regression logistique codee en Python/Numpy** plutot que sur `scikit-learn`.
"""
            ),
            code_cell(IMPORT_BLOCK),
            md_cell(
                """
## 1. Preparation des donnees

- dedoublonnage des `employee_id` strictement dupliques
- exclusion de `employee_id`
- utilisation de **toutes les autres variables disponibles** du dataset
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
            code_cell(
                """
pd.DataFrame(
    {
        "Indicateur": [
            "Variables exclues",
            "Nombre de features apres encodage",
        ],
        "Valeur": [
            "employee_id uniquement",
            len(prepared.feature_names),
        ],
    }
)
"""
            ),
            md_cell(
                """
## 2. Entrainement du modele

On utilise une regression logistique avec ponderation des classes pour tenir compte du desequilibre de la cible, tout en preservant une lecture interpretable des variables. Le dataset actuel contient suffisamment de cas positifs pour produire une evaluation plus stable que precedemment.
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

Les coefficients ci-dessous sont interpretes comme des effets directionnels dans le cadre de cette regression logistique standardisee. Ils ne doivent pas etre lus comme des causalites, surtout lorsque plusieurs variables proches coexistent dans le modele.
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

On teste ici un **random oversampling** sur le jeu d'entrainement uniquement. L'objectif n'est pas de "creer de l'information", mais de verifier si une meilleure representation de la classe minoritaire permet surtout d'ameliorer le compromis entre `recall` et `precision`.
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
- Si le gain reste marginal sur `PR-AUC` et `F1-score`, on peut conclure que le modele de base capte deja l'essentiel du signal utile, et que l'oversampling n'apporte qu'un ajustement secondaire.
"""
            ),
            md_cell(
                """
## 6. Experience de detection d'anomalie

En complement du modele supervise, on peut aussi tester une logique semi-supervisee : apprendre le profil "normal" des collaborateurs qui restent, puis mesurer a quel point certains profils s'en ecartent.

Ici, on utilise une distance de **Mahalanobis regularisee** sur l'ensemble des variables exploitables, apres encodage des variables qualitatives et standardisation. Le profil de reference est appris uniquement sur les collaborateurs sans attrition du train.
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

- Cette approche est interessante quand la classe positive reste minoritaire, car elle ne suppose pas de modeliser finement un grand nombre de cas de depart.
- En revanche, elle repose sur une hypothese forte : les departs seraient des profils **atypiques** par rapport aux collaborateurs qui restent.
- Si les collaborateurs qui quittent l'entreprise ne sont pas des anomalies nettes mais des cas proches du reste de la population, la methode restera limitee.
"""
            ),
            md_cell(
                """
## 7. Lecture business du modele

Variables associees a un **risque plus eleve** dans cette version du modele :

- absentisme plus eleve
- davantage d'heures supplementaires
- certains effets lies a des roles ou sous-populations specifiques, a interpreter avec prudence
- des situations de moindre stabilisation RH : anciennete plus faible, absence de promotion recente et mobilite interne plus faible

Variables associees a un **risque plus faible** :

- meilleure securite psychologique
- un temps contractuel plus important
- davantage de mobilite interne
- un engagement plus eleve
- la presence d'une promotion recente
"""
            ),
            md_cell(
                """
## 8. Analyse critique

### Ce que le modele apporte

- un cadre robuste de priorisation des populations a surveiller
- une lecture interpretable grace aux coefficients
- une base solide pour une future industrialisation

### Limites majeures

- la cible reste **minoritaire** (18,8 % de departs), mais elle est bien plus exploitable qu'auparavant
- les performances du modele supervise sont **solides** sur ce dataset, mais elles doivent etre confirmees hors echantillon temporel
- l'oversampling n'apporte qu'un gain marginal par rapport a la regression logistique ponderee
- la detection d'anomalie reste nettement moins performante que l'approche supervisee
- plusieurs variables sont potentiellement sensibles ou discutables d'un point de vue ethique (`gender`, `accented_name_flag`)
- le resultat peut dependre de la maniere dont certaines variables ont ete construites dans le dataset, notamment l'absenteisme ou la securite psychologique

### Conclusion

Le modele devient ici un **outil d'aide a la priorisation RH credible**, utile pour cibler les actions de prevention, tout en restant insuffisant pour justifier a lui seul des decisions individuelles.
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

Le dataset source contient **8 020 lignes** et **28 variables**. Apres suppression de **20 doublons exacts d'`employee_id`**, l'analyse porte sur **8 000 collaborateurs uniques**. L'objectif etait double :

1. transformer les donnees RH en indicateurs de pilotage
2. construire un premier modele de prediction de l'attrition
"""
            ),
            md_cell(
                """
## Faits marquants de l'EDA et de l'analyse RH

### 1. Une attrition elevee mais heterogene

- Le taux d'attrition global s'etablit a **18,77 %** apres dedoublonnage.
- Les niveaux les plus eleves apparaissent en **Espagne (20,06 %)** et en **France (19,66 %)**.
- Cote departements, **HR (21,53 %)** et **Finance (20,63 %)** sont les plus exposes.

### 2. L'engagement et la securite psychologique jouent un role de fond

- Les collaborateurs partis ont un **engagement plus faible** que les collaborateurs restes.
- Leur **securite psychologique** est tres nettement plus basse.
- Les populations avec engagement faible affichent un taux d'attrition superieur aux autres groupes.

### 3. Les conditions de travail donnent des signaux complementaires

- Les collaborateurs partis presentent beaucoup plus d'**absenteisme** et davantage d'**heures supplementaires**.
- Ils ont aussi connu **moins de mobilite interne** et une remuneration legerement plus basse.

### 4. Carriere, mobilite et remuneration

- Environ **23,9 %** des collaborateurs ont eu une promotion sur 3 ans.
- La **mobilite interne** concerne un peu plus de **61,8 %** des collaborateurs au moins une fois.
- Le **salaire moyen** est d'environ **62 300** par an, avec une progression nette selon le niveau hierarchique.
- Un ecart brut apparait entre salaire moyen des femmes et des hommes, ce qui justifie une analyse plus fine par poste, pays et niveau.

### 5. Hypotheses croisees confirmees

- La **mobilite interne** est un signal transversal majeur : sans mobilite, l'attrition depasse 30 % dans toutes les classes d'age ; avec mobilite, elle retombe autour de 9-10 %.
- La **securite psychologique** modere fortement l'effet de la surcharge : une safety haute maintient une attrition faible meme avec des heures supplementaires elevees.
- Le **remote** reduit nettement la visibilite interne, mais cette baisse ne se traduit pas directement par une baisse forte des promotions dans ce dataset.
- Chez les parents en forte surcharge, l'attrition des femmes est environ deux fois plus elevee que celle des hommes, ce qui signale un enjeu RH et d'equite a investiguer.
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

Le modele retenu est une regression logistique interpretable. Il a ete entraine sur les donnees nettoyees, dedoublonnees et encodees, en utilisant **toutes les variables exploitables** du dataset. Seul `employee_id` est exclu, car il s'agit d'un identifiant technique.
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
        ("Features utilisees apres encodage", len(prepared.feature_names)),
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

- Les performances du modele sont **solides** sur ce dataset, ce qui rend la priorisation des populations a risque plus credible.
- Les variables les plus contributives vont dans le sens des analyses descriptives : **absenteisme**, **heures supplementaires**, **securite psychologique**, **mobilite interne** et **engagement** participent fortement au signal.
- Le modele utilise desormais l'ensemble des variables exploitables, y compris les variables qualitatives encodees et les variables ajoutees comme `years_at_company` ou `accented_name_flag`.
- Ce resultat suggere que le dataset actuel embarque un signal predictif nettement plus exploitable qu'un simple bruit organisationnel diffus.
"""
            ),
            md_cell(
                """
## Experience complementaire : detection d'anomalie

Une approche alternative a egalement ete testee : la **detection d'anomalie**. L'idee consiste a apprendre le profil "habituel" des collaborateurs qui restent, puis a identifier les profils qui s'en ecartent le plus. Cette experience utilise elle aussi toutes les variables exploitables apres encodage, afin d'etre comparable au modele supervise.
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
            test_metrics["precision"],
            test_metrics["recall"],
            test_metrics["f1_score"],
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
"""
            ),
            md_cell(
                """
### Lecture de cette experience

- La regression logistique ponderee reste l'approche la plus pertinente : elle combine performance solide et interpretation metier.
- L'oversampling n'ameliore pas vraiment le modele : il modifie legerement le compromis precision/recall, mais n'apporte pas de gain net.
- La detection d'anomalie devient nettement moins convaincante lorsque toutes les variables sont prises en compte : son `ROC-AUC` est proche de 0,50, ce qui indique une capacite de classement tres limitee.
"""
            ),
            md_cell(
                """
## Recommandations

1. Renforcer le suivi des populations a risque dans les segments les plus exposes : **Espagne**, **France**, ainsi que les departements **HR**, **Finance** et **Sales**.
2. Utiliser la **securite psychologique**, l'**absenteisme**, la **charge de travail** et l'**engagement** comme signaux de prevention a traiter en priorite.
3. Cibler davantage les actions de **mobilite interne**, de **promotion** et d'integration des collaborateurs les moins anciens pour soutenir la retention.
4. Investiguer specifiquement les populations **parents + forte surcharge**, ou l'ecart d'attrition femmes/hommes est tres marque.
5. Realiser un audit complementaire sur la **structure de remuneration**, en particulier sur les ecarts par genre, poste, pays et niveau.
6. Encadrer strictement l'usage des variables sensibles ou discutables (`gender`, `accented_name_flag`) et enrichir le dataset avec des variables plus directement causales : historique managerial, changements d'equipe, enquetes qualitatives, intentions de mobilite, etc.
"""
            ),
            md_cell(
                """
## Limites

- Dataset unique, sans validation temporelle ni test sur une autre cohorte
- Certaines variables peuvent etre tres proches du phenomene a predire, ce qui peut gonfler artificiellement la performance
- Certaines variables peuvent poser des questions d'equite et d'ethique
- L'utilisation de toutes les variables ameliore l'exhaustivite du modele, mais impose une vigilance accrue sur les variables sensibles et les variables potentiellement proxy
- Resultats a confirmer avant toute utilisation operationnelle large

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

- L'entreprise presente un **turnover eleve (18,77 %)**, avec des poches de risque bien identifiees.
- Les principaux signaux lies au depart sont une **securite psychologique plus faible**, un **absenteisme plus eleve**, davantage d'**heures supplementaires**, moins de **mobilite interne** et un **engagement plus bas**.
- Les zones a surveiller en priorite sont **l'Espagne**, **la France**, ainsi que les departements **HR**, **Finance** et **Sales**.
- Les analyses croisees confirment trois leviers prioritaires : **mobilite interne**, **securite psychologique en contexte de surcharge**, et vigilance sur les **parents en forte charge de travail**.
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
- plus marque chez les collaborateurs les moins ancres dans l'organisation : moindre anciennete, moins de promotions et moins de mobilite interne
- desormais predible avec un niveau de performance utile pour la priorisation RH

### Engagement

- plus faible dans les populations qui quittent l'entreprise
- articule avec des leviers de management, de charge de travail, de reconnaissance et de securite psychologique
"""
            ),
            md_cell(
                """
## 3. Resultat du modele predictif

Le modele detecte un **signal exploitable** :

- il est utile pour orienter la vigilance RH et prioriser les actions de retention
- il ne doit pas etre utilise seul pour prendre des decisions individuelles
- il utilise toutes les variables exploitables du dataset, hors identifiant technique `employee_id`
- il confirme l'importance de la securite psychologique, de l'absenteisme, de l'engagement, de la mobilite interne, de l'anciennete et de la charge de travail dans la lecture du risque
- l'oversampling n'apporte pas de gain net ; la **detection d'anomalie** reste nettement moins robuste que le modele supervise
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
        "Metrique": ["Features utilisees", "ROC-AUC", "PR-AUC", "Recall", "Precision"],
        "Valeur": [
            len(prepared.feature_names),
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
2. Integrer l'engagement, la securite psychologique, l'absenteisme et la charge de travail dans un dispositif de veille RH trimestriel.
3. Renforcer les parcours de mobilite, de promotion et d'integration des profils les moins anciens comme leviers de retention.
4. Traiter les situations de forte surcharge, en particulier chez les parents, avec une lecture specifique des ecarts femmes/hommes.
5. Lancer une analyse plus fine de l'equite salariale et du role des variables sensibles.
6. Ameliorer la qualite des donnees, encadrer l'usage des variables sensibles et confirmer les resultats sur de nouvelles cohortes avant toute utilisation plus large.
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
