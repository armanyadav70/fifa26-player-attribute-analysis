"""FC 26 player-attribute analysis for Assessment 2.

Research question: Which player attributes have the strongest relationship
with an outfield player's overall rating in EA Sports FC 26?
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "players.csv"
OUTPUT_DIR = ROOT / "outputs"

DETAILED_ATTRIBUTES = [
    "attacking_crossing", "attacking_finishing", "attacking_heading_accuracy",
    "attacking_short_passing", "attacking_volleys", "skill_dribbling",
    "skill_curve", "skill_fk_accuracy", "skill_long_passing", "skill_ball_control",
    "movement_acceleration", "movement_sprint_speed", "movement_agility",
    "movement_reactions", "movement_balance", "power_shot_power", "power_jumping",
    "power_stamina", "power_strength", "power_long_shots", "mentality_aggression",
    "mentality_interceptions", "mentality_positioning", "mentality_vision",
    "mentality_penalties", "mentality_composure", "defending_marking_awareness",
    "defending_standing_tackle", "defending_sliding_tackle",
]

CORE_ATTRIBUTES = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]


def load_and_clean() -> pd.DataFrame:
    """Load the raw data, remove duplicates and keep complete outfield records."""
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df.drop_duplicates(subset="player_id").copy()
    outfield = df[~df["player_positions"].str.contains("GK", na=False)].copy()
    required = ["overall", *CORE_ATTRIBUTES, *DETAILED_ATTRIBUTES]
    outfield = outfield.dropna(subset=required)
    return outfield


def analyse(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Calculate Pearson correlations and a simple one-variable linear fit."""
    correlations = (
        df[["overall", *DETAILED_ATTRIBUTES]]
        .corr(method="pearson")["overall"]
        .drop("overall")
        .sort_values(ascending=False)
        .rename("pearson_r")
        .rename_axis("attribute")
        .reset_index()
    )
    slope, intercept = np.polyfit(df["movement_reactions"], df["overall"], 1)
    predicted = slope * df["movement_reactions"] + intercept
    r = df["movement_reactions"].corr(df["overall"])
    r_squared = 1 - ((df["overall"] - predicted) ** 2).sum() / (
        (df["overall"] - df["overall"].mean()) ** 2
    ).sum()
    summary = {
        "raw_rows": int(pd.read_csv(DATA_PATH, usecols=["player_id"]).shape[0]),
        "analysis_rows": int(len(df)),
        "attribute_count": len(DETAILED_ATTRIBUTES),
        "overall_mean": float(df["overall"].mean()),
        "overall_median": float(df["overall"].median()),
        "reaction_r": float(r),
        "reaction_r_squared": float(r_squared),
        "slope": float(slope),
        "intercept": float(intercept),
    }
    return correlations, summary


def make_charts(df: pd.DataFrame, correlations: pd.DataFrame) -> None:
    """Create the exact visualisations used in the presentation."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.05)
    navy, cyan, gold = "#142B4A", "#16B8C4", "#FFB547"

    top = correlations.head(10).sort_values("pearson_r")
    labels = top["attribute"].str.replace("_", " ").str.title()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = [cyan if value < 0.85 else gold for value in top["pearson_r"]]
    ax.barh(labels, top["pearson_r"], color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Pearson correlation with overall rating (r)")
    ax.set_ylabel("")
    ax.set_title("Top ten detailed attributes", loc="left", color=navy, weight="bold")
    for i, value in enumerate(top["pearson_r"]):
        ax.text(value + 0.012, i, f"{value:.2f}", va="center", color=navy, weight="bold")
    sns.despine(left=True, bottom=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "top_correlations.png", dpi=180, bbox_inches="tight", transparent=False)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    sample = df.sample(min(5000, len(df)), random_state=26)
    sns.regplot(
        data=sample, x="movement_reactions", y="overall", ax=ax,
        scatter_kws={"alpha": 0.18, "s": 13, "color": cyan},
        line_kws={"color": gold, "linewidth": 2.8},
    )
    ax.set_xlabel("Movement reactions rating")
    ax.set_ylabel("Overall rating")
    ax.set_title("Reactions and overall rating", loc="left", color=navy, weight="bold")
    ax.text(0.03, 0.94, "r = 0.887   R² = 0.786", transform=ax.transAxes,
            fontsize=12, color=navy, weight="bold", va="top")
    sns.despine()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "reactions_scatter.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    core_corr = (
        df[["overall", *CORE_ATTRIBUTES]].corr()["overall"].drop("overall")
        .sort_values(ascending=False)
    )
    core_corr.rename("pearson_r").to_csv(OUTPUT_DIR / "core_correlations.csv")


def main() -> None:
    df = load_and_clean()
    correlations, summary = analyse(df)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    correlations.to_csv(OUTPUT_DIR / "detailed_correlations.csv", index=False)
    pd.Series(summary, name="value").to_csv(OUTPUT_DIR / "analysis_summary.csv")
    df[["player_id", "short_name", "player_positions", "overall", *CORE_ATTRIBUTES,
        *DETAILED_ATTRIBUTES]].to_csv(OUTPUT_DIR / "analysis_sample.csv", index=False)
    make_charts(df, correlations)
    print(pd.Series(summary))
    print("\nTop correlations:\n", correlations.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
