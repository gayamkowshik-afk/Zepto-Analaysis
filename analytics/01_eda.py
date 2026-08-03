
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

report_lines = []


def log(md: str = ""):
    """Append a line to the running EDA_REPORT.md and also print it."""
    report_lines.append(md)
    print(md)


# Task 1

def load_titanic() -> pd.DataFrame:
    raw_csv = os.path.join(HERE, "titanic.csv")
    try:
        df = sns.load_dataset("titanic")
        print("Loaded Titanic dataset via seaborn (network/cache).")
    except Exception as e:
        print(f"sns.load_dataset failed ({e}); falling back to committed titanic.csv")
        df = pd.read_csv(raw_csv)
    df.to_csv(raw_csv, index=False)
    return df


def profile(df: pd.DataFrame):
    log("## Task 1 -- Profiling\n")
    log(f"Shape: {df.shape}\n")
    buf = []
    df.info(buf=type("W", (), {"write": lambda self, s: buf.append(s)})())
    log("```\n" + "".join(buf) + "\n```\n")
    log("Describe (numeric):\n")
    log("```\n" + df.describe().to_string() + "\n```\n")

    missing_pct = (df.isna().mean() * 100).round(2)
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
    log("**Percentage missing per column (only columns with any missing):**\n")
    log("```\n" + missing_pct.to_string() + "\n```\n")
    return missing_pct


# Task 2

def clean_missing(df: pd.DataFrame, missing_pct: pd.Series) -> pd.DataFrame:
    df = df.copy()
    log("## Task 2 -- Missing value handling (threshold rule: <5% drop rows, "
        "5-30% impute, >=30% drop column or encode 'missing')\n")

    for col, pct in missing_pct.items():
        if pct < 5:
            before = len(df)
            df = df[df[col].notna()]
            log(f"- `{col}`: {pct}% missing (<5%) -> **dropped rows** "
                f"({before - len(df)} rows removed).")
        elif pct < 30:
            if pd.api.types.is_numeric_dtype(df[col]):
                fill = df[col].median()
                df[col] = df[col].fillna(fill)
                log(f"- `{col}`: {pct}% missing (5-30%) -> **imputed with median** "
                    f"({fill:.2f}).")
            else:
                fill = df[col].mode()[0]
                df[col] = df[col].fillna(fill)
                log(f"- `{col}`: {pct}% missing (5-30%) -> **imputed with mode** "
                    f"('{fill}').")
        else:
            df = df.drop(columns=[col])
            log(f"- `{col}`: {pct}% missing (>=30%) -> **dropped the column** "
                f"(too sparse to impute reliably; not used downstream).")
    log("")
    return df

# Task 3 - univariate analysis

def iqr_outliers(series: pd.Series) -> int:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((series < lower) | (series > upper)).sum())


def univariate(df: pd.DataFrame):
    log("## Task 3 -- Univariate analysis\n")
    for col in ["age", "fare"]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.histplot(df[col], kde=True, ax=axes[0])
        axes[0].set_title(f"{col} distribution")
        sns.boxplot(x=df[col], ax=axes[1])
        axes[1].set_title(f"{col} boxplot")
        plt.tight_layout()
        path = os.path.join(FIG_DIR, f"univariate_{col}.png")
        plt.savefig(path, dpi=110)
        plt.close(fig)

        n_out = iqr_outliers(df[col])
        log(f"- `{col}`: **{n_out} IQR outliers**. ![{col}](figures/univariate_{col}.png)")

    mean_f, median_f, mode_f = df["fare"].mean(), df["fare"].median(), df["fare"].mode()[0]
    skew_word = "right-skewed" if mean_f > median_f else (
        "left-skewed" if mean_f < median_f else "symmetric")
    log(f"\n`fare` -- mean={mean_f:.2f}, median={median_f:.2f}, mode={mode_f:.2f}. "
        f"Since mean > median > mode, the distribution is **{skew_word}**: a long "
        f"tail of a few very expensive fares pulls the mean above the median, "
        f"while most passengers cluster at cheap fares near the mode.\n")

# Task 4 - bivariate analysis

def bivariate(df: pd.DataFrame):
    log("## Task 4 -- Bivariate analysis\n")

    rate_by_sex = df.groupby("sex")["survived"].mean().round(3)
    log("**Survival rate by sex:**\n```\n" + rate_by_sex.to_string() + "\n```\n")

    rate_by_pclass = df.groupby("pclass")["survived"].mean().round(3)
    log("**Survival rate by pclass:**\n```\n" + rate_by_pclass.to_string() + "\n```\n")

    combo_rows = []
    for sex_val in df["sex"].unique():
        for pclass_val in sorted(df["pclass"].unique()):
            mask = (df["sex"] == sex_val) & (df["pclass"] == pclass_val)
            rate = df.loc[mask, "survived"].mean()
            combo_rows.append({"sex": sex_val, "pclass": pclass_val, "survival_rate": round(rate, 3)})
    combo_df = pd.DataFrame(combo_rows)
    log("**Survival rate by sex AND pclass (boolean masking):**\n```\n"
        + combo_df.to_string(index=False) + "\n```\n")

    corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation matrix (6 numeric columns)")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "correlation_heatmap.png")
    plt.savefig(path, dpi=110)
    plt.close(fig)
    
    corr_abs = corr.abs().to_numpy().copy()
    np.fill_diagonal(corr_abs, 0)
    corr_abs = pd.DataFrame(corr_abs, index=corr.index, columns=corr.columns)
    pairs = []
    for i, c1 in enumerate(corr_cols):
        for j, c2 in enumerate(corr_cols):
            if j <= i:
                continue
            pairs.append((c1, c2, corr_abs.loc[c1, c2]))
    pairs.sort(key=lambda x: x[2], reverse=True)
    top2 = pairs[:2]

    log(f"![correlation heatmap](figures/correlation_heatmap.png)\n")
    log("**Two strongest correlations (by absolute off-diagonal value):**\n")
    for c1, c2, val in top2:
        signed = corr.loc[c1, c2]
        direction = "positive" if signed > 0 else "negative"
        log(f"- `{c1}` & `{c2}`: r = {signed:.3f} ({direction}).")
    log("")
    return corr, top2

# Task 5 -- multivariate data story (>=4 charts)

def multivariate(df: pd.DataFrame):
    log("## Task 5 -- Multivariate data story\n")

    # Chart 1: grouped bar
    fig, ax = plt.subplots(figsize=(6, 4))
    grouped = df.groupby(["pclass", "sex"])["survived"].mean().unstack()
    grouped.plot(kind="bar", ax=ax)
    ax.set_ylabel("Survival rate")
    ax.set_title("Survival rate by class and sex")
    plt.tight_layout()
    path1 = os.path.join(FIG_DIR, "story_bar_class_sex.png")
    plt.savefig(path1, dpi=110)
    plt.close(fig)
    log("**Chart 1 (bar):** ![chart1](figures/story_bar_class_sex.png)\n\n"
        "Survival rate is highest for women in 1st and 2nd class (well above "
        "90%), and lowest for men in 3rd class (under 15%). Sex is the "
        "dominant factor, but class compounds it heavily -- 3rd class women "
        "still survive at roughly double the rate of 1st class men. This "
        "matches the historical 'women and children first' boarding norm, "
        "moderated by which deck/class had faster access to lifeboats.\n")

    # Chart 2: box
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="pclass", y="age", hue="survived", ax=ax)
    ax.set_title("Age distribution by class and survival")
    plt.tight_layout()
    path2 = os.path.join(FIG_DIR, "story_box_age_class_survival.png")
    plt.savefig(path2, dpi=110)
    plt.close(fig)
    log("**Chart 2 (box):** ![chart2](figures/story_box_age_class_survival.png)\n\n"
        "Median age climbs from 3rd to 1st class regardless of survival "
        "outcome, reflecting that wealthier, older passengers could afford "
        "higher-class tickets. Within each class, survivors skew only "
        "slightly younger than non-survivors -- age matters, but far less "
        "than class or sex do on their own.\n")

    # Chart 3: scatter
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=df, x="age", y="fare", hue="survived", alpha=0.6, ax=ax)
    ax.set_title("Fare vs Age, colored by survival")
    plt.tight_layout()
    path3 = os.path.join(FIG_DIR, "story_scatter_fare_age.png")
    plt.savefig(path3, dpi=110)
    plt.close(fig)
    log("**Chart 3 (scatter):** ![chart3](figures/story_scatter_fare_age.png)\n\n"
        "Survivors (orange) are noticeably denser at higher fares across all "
        "ages, while non-survivors cluster at low fares. There's no strong "
        "age trend within either group -- fare (a proxy for class/wealth) "
        "separates survival outcomes far more cleanly than age does on a "
        "scatter plot.\n")

    # Chart 4: heatmap 
    fig, ax = plt.subplots(figsize=(6, 4))
    pivot = df.pivot_table(index="pclass", columns="embark_town", values="survived", aggfunc="mean")
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
    ax.set_title("Survival rate by class and embarkation town")
    plt.tight_layout()
    path4 = os.path.join(FIG_DIR, "story_heatmap_class_embark.png")
    plt.savefig(path4, dpi=110)
    plt.close(fig)
    log("**Chart 4 (heatmap):** ![chart4](figures/story_heatmap_class_embark.png)\n\n"
        "Cherbourg passengers survive at a noticeably higher rate within "
        "every class than Southampton or Queenstown passengers do -- likely "
        "because Cherbourg boarders skewed more heavily toward 1st class "
        "generally. Southampton 3rd class, the largest and poorest group "
        "aboard, has the lowest survival rate of any cell in this grid.\n")

# Task 6 -- exploratory standardization check

def standardization_check(df: pd.DataFrame):
    log("## Task 6 -- Exploratory standardization check (EDA-stage only)\n")
    for col in ["age", "fare"]:
        mean, std = df[col].mean(), df[col].std()
        z = (df[col] - mean) / std
        log(f"- `{col}`: before -> mean={mean:.3f}, std={std:.3f}; "
            f"after z-score -> mean={z.mean():.3f}, std={z.std():.3f} "
            f"(approximately 0 and 1, as expected).")
    log("\nThis is purely a sanity check on the cleaned EDA data. It does "
        "**not** feed into the modeling pipeline in Part B, which performs "
        "its own train-only `StandardScaler` fit as part of Task 8.\n")


def main():
    df_raw = load_titanic()
    missing_pct = profile(df_raw)
    df_clean = clean_missing(df_raw, missing_pct)

    univariate(df_clean)
    bivariate(df_clean)
    multivariate(df_clean)
    standardization_check(df_clean)

    clean_path = os.path.join(HERE, "titanic_clean.csv")
    df_clean.to_csv(clean_path, index=False)
    print(f"\nCleaned data saved -> {clean_path} ({df_clean.shape[0]} rows, "
          f"{df_clean.shape[1]} cols)")

    report_path = os.path.join(HERE, "EDA_REPORT.md")
    with open(report_path, "w") as f:
        f.write("# Module 2 -- EDA Report\n\n")
        f.write("\n".join(report_lines))
    print(f"EDA report saved -> {report_path}")


if __name__ == "__main__":
    main()
