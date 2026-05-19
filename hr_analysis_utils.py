from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_DATA_PATH = Path("raw/people_analytics_dataset.csv")
TARGET_COLUMN = "attrition"
DERIVED_COLUMNS = {
    "age_group",
    "remote_band",
    "engagement_band",
    "overtime_band",
    "absence_band",
    "salary_quartile",
}


def load_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return add_derived_columns(df)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    if "age" in data.columns:
        data["age_group"] = pd.cut(
            data["age"],
            bins=[0, 29, 39, 49, 100],
            labels=["<30", "30-39", "40-49", "50+"],
            include_lowest=True,
        )

    if "remote_ratio" in data.columns:
        remote_map = {0: "0%", 25: "25%", 50: "50%", 75: "75%", 100: "100%"}
        data["remote_band"] = data["remote_ratio"].map(remote_map).fillna("Other")

    if "engagement_score" in data.columns:
        data["engagement_band"] = pd.cut(
            data["engagement_score"],
            bins=[0, 60, 70, 80, 100],
            labels=["Low", "Medium", "High", "Very High"],
            include_lowest=True,
        )

    if "overtime_hours" in data.columns:
        data["overtime_band"] = pd.cut(
            data["overtime_hours"],
            bins=[-1, 5, 15, 25, 10_000],
            labels=["0-5", "6-15", "16-25", "26+"],
            include_lowest=True,
        )

    if "absenteeism_days" in data.columns:
        data["absence_band"] = pd.cut(
            data["absenteeism_days"],
            bins=[-1, 2, 5, 10, 10_000],
            labels=["0-2", "3-5", "6-10", "11+"],
            include_lowest=True,
        )

    if "salary" in data.columns and data["salary"].notna().any():
        salary_filled = data["salary"].fillna(data["salary"].median())
        try:
            data["salary_quartile"] = pd.qcut(
                salary_filled,
                4,
                labels=["Q1", "Q2", "Q3", "Q4"],
                duplicates="drop",
            )
        except ValueError:
            data["salary_quartile"] = "Q1"

    return data


def get_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_cols = [
        col
        for col in df.columns
        if col != TARGET_COLUMN and pd.api.types.is_numeric_dtype(df[col])
    ]
    categorical_cols = [
        col
        for col in df.columns
        if col != TARGET_COLUMN and not pd.api.types.is_numeric_dtype(df[col])
    ]
    return numeric_cols, categorical_cols


def dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    source_columns = [col for col in df.columns if col not in DERIVED_COLUMNS]
    source_df = df[source_columns].copy()

    duplicate_ids = 0
    if "employee_id" in source_df.columns:
        duplicate_ids = int(source_df["employee_id"].duplicated().sum())

    rows = [
        ("Nombre de lignes", len(source_df)),
        ("Variables sources", len(source_columns)),
        ("Variables ajoutees", len(df.columns) - len(source_columns)),
        ("Taux d'attrition (%)", round(source_df[TARGET_COLUMN].mean() * 100, 2)),
        ("Cellules manquantes (source)", int(source_df.isna().sum().sum())),
        ("employee_id dupliques", duplicate_ids),
    ]
    return pd.DataFrame(rows, columns=["Indicateur", "Valeur"])


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    source_columns = [col for col in df.columns if col not in DERIVED_COLUMNS]
    source_df = df[source_columns].copy()
    summary = pd.DataFrame(
        {
            "missing_count": source_df.isna().sum(),
            "missing_pct": source_df.isna().mean() * 100,
            "dtype": source_df.dtypes.astype(str),
        }
    )
    summary = summary.query("missing_count > 0").sort_values("missing_count", ascending=False)
    return summary.round({"missing_pct": 2})


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols, _ = get_feature_lists(df)
    summary = df[numeric_cols].describe().T
    return summary[["mean", "std", "min", "25%", "50%", "75%", "max"]].round(2)


def kpi_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("Effectif total", len(df)),
        ("Taux d'attrition (%)", round(df[TARGET_COLUMN].mean() * 100, 2)),
        ("Score d'engagement moyen", round(df["engagement_score"].mean(), 2)),
        (
            "Score de securite psychologique moyen",
            round(df["psychological_safety_score"].mean(), 2),
        ),
        ("Taux de promotion sur 3 ans (%)", round(df["promotion_last_3y"].mean() * 100, 2)),
        (
            "Taux de mobilite interne (%)",
            round((df["internal_mobility_count"] > 0).mean() * 100, 2),
        ),
        ("Heures de formation moyennes", round(df["training_hours"].mean(), 2)),
        ("Absenteisme moyen (jours)", round(df["absenteeism_days"].mean(), 2)),
        ("Heures supp. moyennes", round(df["overtime_hours"].mean(), 2)),
        ("Salaire moyen annuel", round(df["salary"].mean(), 2)),
        ("Bonus annuel moyen", round(df["bonus"].mean(), 2)),
    ]
    return pd.DataFrame(rows, columns=["KPI", "Valeur"])


def attrition_by_group(
    df: pd.DataFrame,
    group_col: str,
    *,
    min_count: int = 0,
    ascending: bool = False,
) -> pd.DataFrame:
    grouped = (
        df.groupby(group_col, dropna=False)
        .agg(
            effectif=(TARGET_COLUMN, "size"),
            departs=(TARGET_COLUMN, "sum"),
            attrition_rate_pct=(TARGET_COLUMN, lambda s: s.mean() * 100),
        )
        .reset_index()
    )
    grouped = grouped[grouped["effectif"] >= min_count]
    return grouped.sort_values("attrition_rate_pct", ascending=ascending).round(
        {"attrition_rate_pct": 2}
    )


def average_by_group(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    table = (
        df.groupby(group_col, dropna=False)[value_col]
        .mean()
        .sort_values(ascending=False)
        .reset_index(name=f"avg_{value_col}")
    )
    return table.round(2)


def salary_gap_by_gender(df: pd.DataFrame) -> pd.DataFrame:
    salary = df.groupby("gender")["salary"].mean().round(2)
    result = salary.reset_index(name="avg_salary")
    male_salary = salary.get("Male")
    if male_salary:
        result["gap_vs_male_pct"] = ((result["avg_salary"] - male_salary) / male_salary * 100).round(2)
    else:
        result["gap_vs_male_pct"] = np.nan
    return result


def attrition_gap_summary(df: pd.DataFrame) -> pd.DataFrame:
    compare_cols = [
        "engagement_score",
        "psychological_safety_score",
        "overtime_hours",
        "absenteeism_days",
        "training_hours",
        "salary",
        "visibility_score",
        "manager_favoritism_score",
        "years_at_company",
        "internal_mobility_count",
    ]
    existing_cols = [col for col in compare_cols if col in df.columns]
    departed = df[df[TARGET_COLUMN] == 1]
    stayed = df[df[TARGET_COLUMN] == 0]
    summary = pd.DataFrame(
        {
            "depart_mean": departed[existing_cols].mean(),
            "reste_mean": stayed[existing_cols].mean(),
        }
    )
    summary["ecart_depart_moins_reste"] = summary["depart_mean"] - summary["reste_mean"]
    return summary.round(2)


def correlation_with_attrition(df: pd.DataFrame) -> pd.Series:
    numeric_cols, _ = get_feature_lists(df)
    columns = [col for col in numeric_cols if col != TARGET_COLUMN] + [TARGET_COLUMN]
    corr = df[columns].corr(numeric_only=True)[TARGET_COLUMN].sort_values(ascending=False)
    return corr.round(3)


def top_attrition_segments(
    df: pd.DataFrame,
    group_cols: Iterable[str],
    *,
    min_count: int = 80,
    top_n: int = 10,
) -> pd.DataFrame:
    grouped = (
        df.groupby(list(group_cols), dropna=False)
        .agg(
            effectif=(TARGET_COLUMN, "size"),
            attrition_rate_pct=(TARGET_COLUMN, lambda s: s.mean() * 100),
            engagement_mean=("engagement_score", "mean"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["effectif"] >= min_count]
    grouped = grouped.sort_values("attrition_rate_pct", ascending=False).head(top_n)
    return grouped.round({"attrition_rate_pct": 2, "engagement_mean": 2})


def plot_bar(
    table: pd.DataFrame,
    category_col: str,
    value_col: str,
    *,
    title: str,
    color: str = "#0F766E",
    rotation: int = 45,
    horizontal: bool = False,
    figsize: tuple[int, int] = (9, 5),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)

    if horizontal:
        ax.barh(table[category_col].astype(str), table[value_col], color=color)
    else:
        ax.bar(table[category_col].astype(str), table[value_col], color=color)
        ax.tick_params(axis="x", rotation=rotation)

    ax.set_title(title)
    ax.set_xlabel(category_col)
    ax.set_ylabel(value_col)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.show()


def plot_histogram(
    df: pd.DataFrame,
    column: str,
    *,
    bins: int = 20,
    color: str = "#2563EB",
    figsize: tuple[int, int] = (8, 4),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(df[column].dropna(), bins=bins, color=color, alpha=0.85, edgecolor="white")
    ax.set_title(f"Distribution de {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequence")
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.show()


def plot_box_by_attrition(
    df: pd.DataFrame,
    column: str,
    *,
    figsize: tuple[int, int] = (7, 4),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    groups = [
        df.loc[df[TARGET_COLUMN] == 0, column].dropna(),
        df.loc[df[TARGET_COLUMN] == 1, column].dropna(),
    ]
    ax.boxplot(groups, labels=["Reste", "Depart"], patch_artist=True)
    ax.set_title(f"{column} selon l'attrition")
    ax.set_ylabel(column)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    figsize: tuple[int, int] = (10, 8),
) -> None:
    numeric_cols, _ = get_feature_lists(df)
    use_cols = columns or [col for col in numeric_cols if col != TARGET_COLUMN] + [TARGET_COLUMN]
    corr = df[use_cols].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(use_cols)))
    ax.set_yticks(range(len(use_cols)))
    ax.set_xticklabels(use_cols, rotation=90)
    ax.set_yticklabels(use_cols)
    ax.set_title("Matrice de correlation")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35, 35)
    return 1.0 / (1.0 + np.exp(-clipped))


def _add_intercept(matrix: np.ndarray) -> np.ndarray:
    intercept = np.ones((matrix.shape[0], 1))
    return np.hstack([intercept, matrix])


@dataclass
class PreparedModelData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    means: pd.Series
    stds: pd.Series


@dataclass
class OversampleResult:
    X_resampled: np.ndarray
    y_resampled: np.ndarray
    original_positive_count: int
    original_negative_count: int
    resampled_positive_count: int
    resampled_negative_count: int


@dataclass
class PreparedNumericData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    medians: pd.Series
    means: pd.Series
    stds: pd.Series


def prepare_model_data(
    df: pd.DataFrame,
    *,
    drop_columns: Iterable[str] | None = None,
    seed: int = 42,
) -> PreparedModelData:
    data = df.copy()
    excluded = set(drop_columns or [])
    excluded.add(TARGET_COLUMN)

    feature_cols = [col for col in data.columns if col not in excluded]
    numeric_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(data[col])]
    categorical_cols = [col for col in feature_cols if not pd.api.types.is_numeric_dtype(data[col])]

    for col in numeric_cols:
        data[col] = data[col].fillna(data[col].median())
    for col in categorical_cols:
        mode = data[col].mode(dropna=True)
        replacement = mode.iloc[0] if not mode.empty else "Unknown"
        data[col] = data[col].fillna(replacement)

    X = pd.get_dummies(data[feature_cols], drop_first=False).astype(float)
    y = data[TARGET_COLUMN].astype(int).to_numpy()

    train_idx, val_idx, test_idx = stratified_split_indices(y, seed=seed)

    X_train = X.iloc[train_idx].copy()
    X_val = X.iloc[val_idx].copy()
    X_test = X.iloc[test_idx].copy()

    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0).replace(0, 1)

    X_train = ((X_train - means) / stds).to_numpy()
    X_val = ((X_val - means) / stds).to_numpy()
    X_test = ((X_test - means) / stds).to_numpy()

    return PreparedModelData(
        X_train=_add_intercept(X_train),
        X_val=_add_intercept(X_val),
        X_test=_add_intercept(X_test),
        y_train=y[train_idx],
        y_val=y[val_idx],
        y_test=y[test_idx],
        feature_names=X.columns.tolist(),
        means=means,
        stds=stds,
    )


def random_oversample_minority(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 42,
) -> OversampleResult:
    rng = np.random.default_rng(seed)

    positive_idx = np.where(y == 1)[0]
    negative_idx = np.where(y == 0)[0]

    if len(positive_idx) == 0 or len(negative_idx) == 0:
        return OversampleResult(
            X_resampled=X.copy(),
            y_resampled=y.copy(),
            original_positive_count=int(len(positive_idx)),
            original_negative_count=int(len(negative_idx)),
            resampled_positive_count=int(len(positive_idx)),
            resampled_negative_count=int(len(negative_idx)),
        )

    if len(positive_idx) < len(negative_idx):
        extra_idx = rng.choice(positive_idx, size=len(negative_idx) - len(positive_idx), replace=True)
    elif len(negative_idx) < len(positive_idx):
        extra_idx = rng.choice(negative_idx, size=len(positive_idx) - len(negative_idx), replace=True)
    else:
        extra_idx = np.array([], dtype=int)

    all_idx = np.concatenate([np.arange(len(y)), extra_idx])
    rng.shuffle(all_idx)

    X_resampled = X[all_idx]
    y_resampled = y[all_idx]

    return OversampleResult(
        X_resampled=X_resampled,
        y_resampled=y_resampled,
        original_positive_count=int(len(positive_idx)),
        original_negative_count=int(len(negative_idx)),
        resampled_positive_count=int((y_resampled == 1).sum()),
        resampled_negative_count=int((y_resampled == 0).sum()),
    )


def stratified_split_indices(
    y: np.ndarray,
    *,
    train_size: float = 0.6,
    val_size: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    positive_idx = np.where(y == 1)[0]
    negative_idx = np.where(y == 0)[0]

    rng.shuffle(positive_idx)
    rng.shuffle(negative_idx)

    def _split(indexes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_rows = len(indexes)
        train_end = int(n_rows * train_size)
        val_end = int(n_rows * (train_size + val_size))
        return indexes[:train_end], indexes[train_end:val_end], indexes[val_end:]

    pos_train, pos_val, pos_test = _split(positive_idx)
    neg_train, neg_val, neg_test = _split(negative_idx)

    train_idx = np.concatenate([pos_train, neg_train])
    val_idx = np.concatenate([pos_val, neg_val])
    test_idx = np.concatenate([pos_test, neg_test])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return train_idx, val_idx, test_idx


def prepare_numeric_anomaly_data(
    df: pd.DataFrame,
    *,
    drop_columns: Iterable[str] | None = None,
    seed: int = 42,
) -> PreparedNumericData:
    data = df.copy()
    excluded = set(drop_columns or [])
    excluded.add(TARGET_COLUMN)

    numeric_cols = [
        col
        for col in data.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(data[col])
    ]

    numeric_frame = data[numeric_cols].copy()
    medians = numeric_frame.median()
    numeric_frame = numeric_frame.fillna(medians)

    y = data[TARGET_COLUMN].astype(int).to_numpy()
    train_idx, val_idx, test_idx = stratified_split_indices(y, seed=seed)

    X_train = numeric_frame.iloc[train_idx].copy()
    X_val = numeric_frame.iloc[val_idx].copy()
    X_test = numeric_frame.iloc[test_idx].copy()

    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0).replace(0, 1)

    X_train = ((X_train - means) / stds).to_numpy()
    X_val = ((X_val - means) / stds).to_numpy()
    X_test = ((X_test - means) / stds).to_numpy()

    return PreparedNumericData(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y[train_idx],
        y_val=y[val_idx],
        y_test=y[test_idx],
        feature_names=numeric_cols,
        medians=medians,
        means=means,
        stds=stds,
    )


class SimpleLogisticRegression:
    def __init__(
        self,
        *,
        learning_rate: float = 0.05,
        epochs: int = 4000,
        reg_strength: float = 0.02,
        class_weight: str = "balanced",
    ) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.reg_strength = reg_strength
        self.class_weight = class_weight
        self.weights_: np.ndarray | None = None
        self.loss_history_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleLogisticRegression":
        n_rows, n_cols = X.shape
        weights = np.zeros(n_cols)

        if self.class_weight == "balanced":
            positives = max(int(y.sum()), 1)
            negatives = max(int(len(y) - y.sum()), 1)
            pos_weight = len(y) / (2 * positives)
            neg_weight = len(y) / (2 * negatives)
            sample_weights = np.where(y == 1, pos_weight, neg_weight)
        else:
            sample_weights = np.ones_like(y, dtype=float)

        for _ in range(self.epochs):
            scores = X @ weights
            predictions = _sigmoid(scores)
            weighted_error = (predictions - y) * sample_weights
            gradient = (X.T @ weighted_error) / n_rows
            gradient[1:] += (self.reg_strength / n_rows) * weights[1:]
            weights -= self.learning_rate * gradient

            loss = -np.mean(
                sample_weights
                * (
                    y * np.log(predictions + 1e-12)
                    + (1 - y) * np.log(1 - predictions + 1e-12)
                )
            )
            self.loss_history_.append(float(loss))

        self.weights_ = weights
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise ValueError("Le modele doit etre entraine avant predict_proba.")
        return _sigmoid(X @ self.weights_)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


class MahalanobisAnomalyDetector:
    def __init__(self, *, regularization: float = 0.1) -> None:
        self.regularization = regularization
        self.mean_: np.ndarray | None = None
        self.inv_cov_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "MahalanobisAnomalyDetector":
        if X.ndim != 2:
            raise ValueError("X doit etre une matrice 2D.")

        self.mean_ = X.mean(axis=0)
        covariance = np.cov(X, rowvar=False)
        if covariance.ndim == 0:
            covariance = np.array([[float(covariance)]])
        covariance = covariance + np.eye(covariance.shape[0]) * self.regularization
        self.inv_cov_ = np.linalg.pinv(covariance)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.inv_cov_ is None:
            raise ValueError("Le detecteur doit etre entraine avant score_samples.")

        centered = X - self.mean_
        scores = np.sqrt(np.sum((centered @ self.inv_cov_) * centered, axis=1))
        return scores


def roc_auc_score_manual(y_true: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    positive_mask = y_true == 1
    n_pos = int(positive_mask.sum())
    n_neg = int((~positive_mask).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    return float((ranks[positive_mask].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pr_auc_score_manual(y_true: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    sorted_y = y_true[order]
    positives = max(int((sorted_y == 1).sum()), 1)
    tp = np.cumsum(sorted_y == 1)
    fp = np.cumsum(sorted_y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives
    precision = np.r_[1, precision]
    recall = np.r_[0, recall]
    return float(np.trapezoid(precision, recall))


def classification_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    predictions = (scores >= threshold).astype(int)
    tp = int(((predictions == 1) & (y_true == 1)).sum())
    tn = int(((predictions == 0) & (y_true == 0)).sum())
    fp = int(((predictions == 1) & (y_true == 0)).sum())
    fn = int(((predictions == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "specificity": specificity,
        "accuracy": accuracy,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def find_best_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    thresholds: np.ndarray | None = None,
) -> dict[str, float]:
    threshold_grid = thresholds if thresholds is not None else np.linspace(0.05, 0.95, 91)
    all_metrics = [classification_metrics(y_true, scores, threshold=value) for value in threshold_grid]
    return max(all_metrics, key=lambda item: item["f1_score"])


def coefficient_importance(
    model: SimpleLogisticRegression,
    feature_names: list[str],
    *,
    top_n: int = 15,
) -> pd.DataFrame:
    if model.weights_ is None:
        raise ValueError("Le modele doit etre entraine avant l'analyse des coefficients.")

    coefficients = pd.Series(model.weights_[1:], index=feature_names, name="coefficient")
    table = coefficients.reset_index().rename(columns={"index": "feature"})
    table["abs_coefficient"] = table["coefficient"].abs()
    table["odds_ratio"] = np.exp(table["coefficient"])
    table = table.sort_values("abs_coefficient", ascending=False).head(top_n)
    return table.round(4)


def plot_top_coefficients(
    model: SimpleLogisticRegression,
    feature_names: list[str],
    *,
    top_n: int = 15,
    figsize: tuple[int, int] = (10, 6),
) -> None:
    table = coefficient_importance(model, feature_names, top_n=top_n).sort_values("coefficient")
    colors = ["#B91C1C" if value > 0 else "#1D4ED8" for value in table["coefficient"]]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(table["feature"], table["coefficient"], color=colors)
    ax.set_title("Variables les plus contributives du modele")
    ax.set_xlabel("Coefficient standardise")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.show()
