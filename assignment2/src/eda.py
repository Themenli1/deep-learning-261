from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Output directories
SCRIPT_DIR = Path(__file__).resolve().parent
ASSIGNMENT_ROOT = SCRIPT_DIR.parent
RESULTS_EDA_DIR = ASSIGNMENT_ROOT / "results" / "eda"
DOCS_ASSETS_DIR = ASSIGNMENT_ROOT.parent / "docs" / "assets" / "a2"

RESULTS_EDA_DIR.mkdir(parents=True, exist_ok=True)
DOCS_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# Load class names from test_metrics.json
with open(ASSIGNMENT_ROOT / "results" / "resnet50_frozen" / "test_metrics.json", "r", encoding="utf-8") as f:
    metrics_data = json.load(f)

classes = [
    k for k in metrics_data["per_class"].keys()
    if k not in ["accuracy", "macro avg", "weighted avg"]
]

# 1. Broad culinary grouping. The eight groups below partition all 101
# Food-101 classes exactly once (22 + 13 + 15 + 17 + 12 + 8 + 10 + 4 = 101).
# This grouping is an analysis aid, not an official Food-101 taxonomy.
CATEGORY_MAPPING = {
    "Desserts & Pastries": [
        "apple_pie", "baklava", "beignets", "bread_pudding", "cannoli", "carrot_cake",
        "cheesecake", "chocolate_cake", "chocolate_mousse", "churros", "creme_brulee",
        "cup_cakes", "donuts", "frozen_yogurt", "ice_cream", "macarons", "pancakes",
        "panna_cotta", "red_velvet_cake", "strawberry_shortcake", "tiramisu", "waffles"
    ],
    "Meat & Poultry": [
        "baby_back_ribs", "beef_carpaccio", "beef_tartare", "chicken_curry",
        "chicken_wings", "filet_mignon", "foie_gras", "fried_calamari", "peking_duck",
        "pork_chop", "prime_rib", "pulled_pork_sandwich", "steak"
    ],
    "Asian & Regional Specialties": [
        "bibimbap", "dumplings", "edamame", "fried_rice", "gyoza", "hot_and_sour_soup",
        "miso_soup", "pad_thai", "pho", "ramen", "samosa", "sashimi", "spring_rolls",
        "sushi", "takoyaki"
    ],
    "Fast Food & Sandwiches": [
        "club_sandwich", "french_fries", "grilled_cheese_sandwich", "hamburger",
        "hot_dog", "lobster_roll_sandwich", "nachos", "onion_rings", "pizza",
        "breakfast_burrito", "tacos", "bruschetta", "chicken_quesadilla",
        "croque_madame", "falafel", "garlic_bread", "poutine"
    ],
    "Seafood": [
        "ceviche", "clam_chowder", "crab_cakes", "escargots", "fish_and_chips",
        "grilled_salmon", "lobster_bisque", "mussels", "oysters", "scallops",
        "tuna_tartare", "shrimp_and_grits"
    ],
    "Pasta, Noodles & Rice": [
        "gnocchi", "lasagna", "macaroni_and_cheese", "ravioli", "risotto",
        "spaghetti_bolognese", "spaghetti_carbonara", "paella"
    ],
    "Salads, Vegetables & Starters": [
        "beet_salad", "caesar_salad", "caprese_salad", "greek_salad", "guacamole",
        "hummus", "seaweed_salad", "cheese_plate", "deviled_eggs", "french_onion_soup"
    ],
    "Breakfast & Egg Dishes": [
        "eggs_benedict", "huevos_rancheros", "omelette", "french_toast"
    ]
}


def plot_class_distribution() -> None:
    """Generate high-resolution class distribution and split breakdown."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.35, 1.0]})

    # Split breakdown
    splits = ["Train\n(90%)", "Validation\n(10%)", "Test\n(official)"]
    samples_per_split = [68175, 7575, 25250]
    colors = ["#2b5c8f", "#419d78", "#e07a5f"]

    bars = ax1.bar(splits, samples_per_split, color=colors, edgecolor="black", linewidth=1.2, width=0.55)
    ax1.set_title("Food-101 Split Partitioning (Balanced across 101 classes)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Total Number of Images", fontsize=9.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.set_ylim(0, 90000)

    for idx, (bar, count) in enumerate(zip(bars, samples_per_split)):
        yval = bar.get_height()
        pct = yval / 101000 * 100
        per_class = int(yval / 101)
        ax1.text(
            bar.get_x() + 0.03 if idx == 0 else bar.get_x() + bar.get_width() / 2,
            yval + 1500,
            f"{count:,} images\n({pct:.1f}% total | {per_class}/class)",
            ha="left" if idx == 0 else "center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
        )

    # Category Group Pie Chart
    cat_names = list(CATEGORY_MAPPING.keys())
    cat_counts = [len(v) for v in CATEGORY_MAPPING.values()]
    cat_labels = [f"{name} ({count})" for name, count in zip(cat_names, cat_counts)]
    palette = sns.color_palette("Set2", len(cat_names))

    wedges, texts, autotexts = ax2.pie(
        cat_counts,
        labels=cat_labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=palette,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 7.8},
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax2.set_title("Food-101 Super-Category Composition (101 Classes)", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(RESULTS_EDA_DIR / "class_distribution.png", dpi=200, bbox_inches="tight")
    plt.savefig(DOCS_ASSETS_DIR / "class_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved class_distribution.png")


def plot_per_class_accuracy_comparison() -> None:
    """Analyze per-class difficulty and distinctive food categories."""
    # Load E2 per-class metrics
    with open(ASSIGNMENT_ROOT / "results" / "resnet50_frozen" / "test_metrics.json", "r", encoding="utf-8") as f:
        e2_raw = json.load(f)
        e2_data = e2_raw["per_class"]

    class_acc_diff = []
    for c in classes:
        if c in e2_data and "recall" in e2_data[c]:
            acc_e2 = e2_data[c]["recall"] * 100.0  # recall per class = accuracy per class on balanced set
            class_acc_diff.append((acc_e2, c))

    class_acc_diff.sort(key=lambda x: x[0])

    # Top 8 hardest and top 8 easiest in E2
    hardest = class_acc_diff[:8]
    easiest = class_acc_diff[-8:][::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.1))

    # Hardest classes
    h_names = [c[1].replace("_", " ").title() for c in hardest]
    h_vals = [c[0] for c in hardest]
    bars1 = ax1.barh(h_names, h_vals, color="#d9534f", edgecolor="black", alpha=0.85)
    ax1.set_title("Top 8 Most Challenging Food Classes (E2 Accuracy %)", fontsize=10.5, fontweight="bold")
    ax1.set_xlabel("Accuracy (%)", fontsize=9.5)
    ax1.set_xlim(0, 100)
    ax1.grid(axis="x", linestyle="--", alpha=0.5)
    for b in bars1:
        ax1.text(b.get_width() + 1.5, b.get_y() + b.get_height() / 2, f"{b.get_width():.1f}%", va="center", fontsize=8, fontweight="bold")

    # Easiest classes
    e_names = [c[1].replace("_", " ").title() for c in easiest]
    e_vals = [c[0] for c in easiest]
    bars2 = ax2.barh(e_names, e_vals, color="#5cb85c", edgecolor="black", alpha=0.85)
    ax2.set_title("Top 8 Easiest / Distinctive Food Classes (E2 Accuracy %)", fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("Accuracy (%)", fontsize=9.5)
    ax2.set_xlim(0, 100)
    ax2.grid(axis="x", linestyle="--", alpha=0.5)
    for b in bars2:
        ax2.text(b.get_width() + 1.5, b.get_y() + b.get_height() / 2, f"{b.get_width():.1f}%", va="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(RESULTS_EDA_DIR / "per_class_difficulty.png", dpi=200, bbox_inches="tight")
    plt.savefig(DOCS_ASSETS_DIR / "per_class_difficulty.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved per_class_difficulty.png")


def plot_top_confusions() -> None:
    """Plot the top-5 saved confusion pairs for each model.

    Only pairs stored in each run's ``test_metrics.json`` are shown, so every
    number in the figure is directly traceable to a saved artifact.
    """
    models = [
        ("E1 - Simple CNN (scratch)", "baseline", "#2b5c8f"),
        ("E2 - ResNet-50 (frozen)", "resnet50_frozen", "#419d78"),
        ("E3 - ResNet-50 (fine-tuned)", "resnet50_finetune", "#c8553d"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
    for ax, (title, folder, color) in zip(axes, models):
        metrics_path = ASSIGNMENT_ROOT / "results" / folder / "test_metrics.json"
        with open(metrics_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        pairs = payload.get("top_confusions", [])[:5]
        if not pairs:
            ax.axis("off")
            ax.text(0.5, 0.5, "No saved confusion pairs", ha="center", va="center")
            continue

        labels = [
            f"{p['true_class'].replace('_', ' ')} -> {p['pred_class'].replace('_', ' ')}"
            for p in pairs
        ]
        counts = [p["count"] for p in pairs]
        y_positions = np.arange(len(labels))

        ax.barh(y_positions, counts, color=color, alpha=0.88, edgecolor="black", linewidth=0.8)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Misclassified test images", fontsize=10)
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.set_xlim(0, max(counts) * 1.25)
        for y_pos, count in zip(y_positions, counts):
            ax.text(count + max(counts) * 0.02, y_pos, str(count), va="center", fontsize=8, fontweight="bold")

    fig.suptitle(
        "Top-5 confusion pairs on the official 25,250-image Food-101 test set",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        -0.04,
        "Counts are one-directional (true class -> predicted class) and are read directly from each run's test_metrics.json.",
        ha="center",
        fontsize=9,
        style="italic",
    )
    plt.tight_layout()
    plt.savefig(RESULTS_EDA_DIR / "qualitative_error_analysis.png", dpi=200, bbox_inches="tight")
    plt.savefig(DOCS_ASSETS_DIR / "qualitative_error_analysis.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved qualitative_error_analysis.png")


def save_eda_summary_json() -> None:
    eda_summary = {
        "dataset_name": "Food-101",
        "num_classes": 101,
        "total_images": 101000,
        "official_splits": {
            "train": 75750,
            "test": 25250
        },
        "project_splits": {
            "train": 68175,
            "val": 7575,
            "test": 25250
        },
        "samples_per_class": {
            "train": 675,
            "val": 75,
            "test": 250,
            "total": 1000
        },
        "input_resolution": "224x224 RGB",
        "original_max_dimension": 512,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "super_categories": {k: len(v) for k, v in CATEGORY_MAPPING.items()}
    }
    with open(RESULTS_EDA_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=2)
    print("Saved eda summary.json")


if __name__ == "__main__":
    plot_class_distribution()
    plot_per_class_accuracy_comparison()
    plot_top_confusions()
    save_eda_summary_json()
    print("All EDA & Qualitative Visualizations generated successfully!")
