"""Generate result charts from GPU AutoML E2E test data.

Usage:
    python scripts/generate_charts.py --run                    # Run E2E test + generate charts
    python scripts/generate_charts.py --from-json results.json # Generate from saved data
    python scripts/generate_charts.py --run --compare          # Include Naive vs Aware comparison
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# --- Style ---
COLORS = {
    "xgboost": "#3498DB",
    "rf": "#E67E22",
    "glm": "#9B59B6",
    "ensemble": "#2ECC71",
    "default": "#7400B8",
    "naive": "#E74C3C",
    "aware": "#2ECC71",
}

OUTPUT_DIR = Path("assets/results")


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "#FAFAFA",
        "axes.facecolor": "#FAFAFA",
        "axes.edgecolor": "#CCCCCC",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.color": "#CCCCCC",
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "figure.dpi": 150,
    })


def get_color(algorithm: str) -> str:
    algo_lower = algorithm.lower()
    if "ensemble" in algo_lower or "stack" in algo_lower:
        return COLORS["ensemble"]
    for key in COLORS:
        if key in algo_lower:
            return COLORS[key]
    return COLORS["default"]


def run_e2e_test(n_rows=50000, n_features=20, max_models=10, compare=False):
    """Run E2E test and collect results."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from tests.test_e2e_gpu import generate_synthetic_data
    from gpu_automl import GPUAutoML

    print(f"Running E2E test ({n_rows} rows, {n_features} features, {max_models} models)...")

    X_train, y_train, X_test, y_test = generate_synthetic_data(
        n_rows=n_rows, n_features=n_features
    )

    automl = GPUAutoML(
        max_runtime_secs=300,
        max_models=max_models,
        nfolds=5,
        seed=42,
        memory_aware=True,
        memory_profile=True,
        training_strategy="full",
        preprocess=False,
    )
    automl.fit(X_train, y_train)

    lb = automl.leaderboard()
    report = automl.get_memory_report()

    results = {
        "leaderboard": lb.to_dict(orient="records"),
        "stage_profiles": report.stage_summary().to_dict(orient="records"),
        "model_vram": report.model_vram_summary().to_dict(orient="records"),
        "config": {
            "n_rows": n_rows,
            "n_features": n_features,
            "max_models": max_models,
            "gpu": "RTX 4060 8GB",
        },
    }

    # Naive vs Aware comparison
    if compare:
        print("\nRunning Naive vs Aware comparison...")
        comparison = {}
        for mode_name, memory_aware in [("Naive", False), ("Aware", True)]:
            import time
            automl_cmp = GPUAutoML(
                max_runtime_secs=120,
                max_models=max_models,
                memory_aware=memory_aware,
                memory_profile=True,
                training_strategy="baseline",
                preprocess=False,
            )
            t0 = time.perf_counter()
            automl_cmp.fit(X_train, y_train)
            elapsed = time.perf_counter() - t0
            cmp_lb = automl_cmp.leaderboard()

            metric_col = [c for c in cmp_lb.columns if c in ("auc", "rmse")][0]
            comparison[mode_name] = {
                "time": round(elapsed, 2),
                "models": len(cmp_lb),
                "best_metric": round(float(cmp_lb[metric_col].iloc[0]), 6) if len(cmp_lb) > 0 else 0,
            }
            automl_cmp.shutdown()

        results["comparison"] = comparison

    automl.shutdown()

    # Save JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "e2e_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {json_path}")

    return results


def chart_model_performance(results: dict):
    """Horizontal bar chart: AUC per model."""
    lb = results["leaderboard"]
    if not lb:
        return

    metric_key = [k for k in lb[0] if k in ("auc", "rmse")][0]
    models = [r["model_id"] for r in lb]
    scores = [r[metric_key] for r in lb]
    algos = [r["algorithm"] for r in lb]
    colors = [get_color(a) for a in algos]

    fig, ax = plt.subplots(figsize=(10, max(4, len(models) * 0.5)))
    y_pos = np.arange(len(models))

    bars = ax.barh(y_pos, scores, color=colors, height=0.6, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    ax.set_xlabel(metric_key.upper())
    ax.set_title(f"Model Performance ({metric_key.upper()})")

    # Value labels
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{score:.4f}", va="center", fontsize=9, color="#333")

    # Legend
    from matplotlib.patches import Patch
    legend_items = []
    seen = set()
    for algo in algos:
        c = get_color(algo)
        label = "Ensemble" if "ensemble" in algo.lower() or "stack" in algo.lower() else algo
        if label not in seen:
            legend_items.append(Patch(facecolor=c, label=label))
            seen.add(label)
    ax.legend(handles=legend_items, loc="lower right", fontsize=9)

    plt.tight_layout()
    path = OUTPUT_DIR / "model_performance.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def chart_training_time(results: dict):
    """Horizontal bar chart: training time per model."""
    lb = results["leaderboard"]
    if not lb:
        return

    # Exclude ensembles (training_time=0)
    data = [r for r in lb if r.get("training_time_secs", 0) > 0]
    if not data:
        return

    models = [r["model_id"] for r in data]
    times = [r["training_time_secs"] for r in data]
    colors = [get_color(r["algorithm"]) for r in data]

    fig, ax = plt.subplots(figsize=(10, max(3, len(models) * 0.5)))
    y_pos = np.arange(len(models))
    bars = ax.barh(y_pos, times, color=colors, height=0.6, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    ax.set_xlabel("Training Time (seconds)")
    ax.set_title("Training Time per Model (avg per fold)")

    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{t:.2f}s", va="center", fontsize=9, color="#333")

    plt.tight_layout()
    path = OUTPUT_DIR / "training_time.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def chart_memory_per_stage(results: dict):
    """Bar chart: peak VRAM per pipeline stage."""
    stages = results.get("stage_profiles", [])
    if not stages:
        return

    names = [s["stage"] for s in stages]
    peaks = [s["peak_vram_gb"] for s in stages]
    durations = [s["duration_secs"] for s in stages]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, peaks, width, label="Peak VRAM (GB)",
                    color="#7400B8", alpha=0.85, edgecolor="white")
    ax1.set_ylabel("Peak VRAM (GB)", color="#7400B8")
    ax1.tick_params(axis="y", labelcolor="#7400B8")

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, durations, width, label="Duration (s)",
                    color="#3498DB", alpha=0.7, edgecolor="white")
    ax2.set_ylabel("Duration (seconds)", color="#3498DB")
    ax2.tick_params(axis="y", labelcolor="#3498DB")

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=10)
    ax1.set_title("Pipeline Stage: VRAM Usage & Duration")

    # Combined legend
    fig.legend(loc="upper right", bbox_to_anchor=(0.95, 0.95), fontsize=9)

    plt.tight_layout()
    path = OUTPUT_DIR / "memory_per_stage.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def chart_memory_per_model(results: dict):
    """Bar chart: peak VRAM per model."""
    vram_data = results.get("model_vram", [])
    if not vram_data:
        return

    # Filter models with non-zero VRAM
    data = [r for r in vram_data if r.get("peak_vram_gb", 0) > 0]
    if not data:
        return

    # Sort by VRAM descending
    data.sort(key=lambda r: r["peak_vram_gb"], reverse=True)

    models = [r["model_id"] for r in data]
    vrams = [r["peak_vram_gb"] for r in data]
    colors = [get_color(r["algorithm"]) for r in data]

    fig, ax = plt.subplots(figsize=(10, max(3, len(models) * 0.5)))
    y_pos = np.arange(len(models))
    bars = ax.barh(y_pos, vrams, color=colors, height=0.6, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    ax.set_xlabel("Peak VRAM (GB)")
    ax.set_title("Peak VRAM Usage per Model")

    for bar, v in zip(bars, vrams):
        ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f} GB", va="center", fontsize=9, color="#333")

    plt.tight_layout()
    path = OUTPUT_DIR / "memory_per_model.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def chart_naive_vs_aware(results: dict):
    """Grouped bar chart: Naive vs Aware comparison."""
    comparison = results.get("comparison")
    if not comparison:
        return

    modes = list(comparison.keys())
    times = [comparison[m]["time"] for m in modes]
    models_count = [comparison[m]["models"] for m in modes]
    best_metrics = [comparison[m]["best_metric"] for m in modes]
    colors_list = [COLORS.get(m.lower(), COLORS["default"]) for m in modes]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Time
    bars0 = axes[0].bar(modes, times, color=colors_list, edgecolor="white", width=0.5)
    axes[0].set_title("Total Training Time")
    axes[0].set_ylabel("Seconds")
    for bar, v in zip(bars0, times):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{v:.1f}s", ha="center", fontsize=10)

    # Models
    bars1 = axes[1].bar(modes, models_count, color=colors_list, edgecolor="white", width=0.5)
    axes[1].set_title("Models Trained")
    axes[1].set_ylabel("Count")
    for bar, v in zip(bars1, models_count):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     str(v), ha="center", fontsize=10)

    # Best metric
    bars2 = axes[2].bar(modes, best_metrics, color=colors_list, edgecolor="white", width=0.5)
    axes[2].set_title("Best Model AUC")
    axes[2].set_ylabel("AUC")
    axes[2].set_ylim(min(best_metrics) - 0.01, max(best_metrics) + 0.005)
    for bar, v in zip(bars2, best_metrics):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                     f"{v:.4f}", ha="center", fontsize=10)

    fig.suptitle("Memory-Naive vs Memory-Aware Scheduling", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = OUTPUT_DIR / "naive_vs_aware.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def generate_all_charts(results: dict):
    """Generate all charts from results data."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    print("\nGenerating charts...")
    chart_model_performance(results)
    chart_training_time(results)
    chart_memory_per_stage(results)
    chart_memory_per_model(results)

    if "comparison" in results:
        chart_naive_vs_aware(results)

    print(f"\nAll charts saved to {OUTPUT_DIR}/")


def main():
    parser = argparse.ArgumentParser(description="Generate GPU AutoML result charts")
    parser.add_argument("--run", action="store_true", help="Run E2E test and generate charts")
    parser.add_argument("--compare", action="store_true", help="Include Naive vs Aware comparison")
    parser.add_argument("--from-json", type=str, help="Load results from JSON file")
    parser.add_argument("--rows", type=int, default=50000)
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument("--models", type=int, default=10)
    args = parser.parse_args()

    if args.from_json:
        with open(args.from_json) as f:
            results = json.load(f)
    elif args.run:
        results = run_e2e_test(
            n_rows=args.rows,
            n_features=args.features,
            max_models=args.models,
            compare=args.compare,
        )
    else:
        parser.print_help()
        return

    generate_all_charts(results)


if __name__ == "__main__":
    main()
