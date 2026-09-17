"""Recompute descriptive day-zero analysis and a figure from saved raw results.

No model training or rerunning episodes is performed. Existing experimental
records are read only; derived analysis.json and overview.png are written.

    python scripts/analyze_pilot.py --input artifacts/day0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = ["global_physics", "local_identification", "fixed_5", "fixed_10", "fixed_16", "adaptive"]
LABELS = ["Global\nphysics", "Local\nidentification", "Learned\nH=5", "Learned\nH=10", "Learned\nH=16", "Learned\nadaptive"]
CONDITIONS = ["in_distribution", "damping_shift"]


def load_json(path):
    return Path(path).read_text(encoding="utf-8")


def analyze(directory):
    directory = Path(directory)
    rows = json.loads(load_json(directory / "episodes.json"))
    summary = json.loads(load_json(directory / "summary.json"))
    prediction = json.loads(load_json(directory / "prediction.json"))
    results, paired = [], []
    unique_keys = [(r["condition"], r["method"], r["seed"], r["scenario"]) for r in rows]
    if len(unique_keys) != len(set(unique_keys)):
        raise ValueError("Duplicate episode identities in raw data")
    for condition in CONDITIONS:
        by_method = {}
        for method in METHODS:
            selected = [r for r in rows if r["condition"] == condition and r["method"] == method]
            if not selected:
                raise ValueError(f"No observations for {condition}/{method}")
            by_method[method] = {(r["seed"], r["scenario"]): r for r in selected}
            times = np.asarray([v for r in selected for v in r["planning_ms"]])
            calls = np.asarray([v for r in selected for v in r["model_steps"]])
            horizon_values = [v for r in selected for v in r["horizons"]]
            count = len(selected)
            results.append({
                "condition": condition, "method": method, "episodes": count,
                "successes": sum(r["success"] for r in selected),
                "collisions": sum(r["collision"] for r in selected),
                "timeouts": sum(r["timeout"] for r in selected),
                "planning_ms_p50": float(np.median(times)),
                "planning_ms_p95": float(np.quantile(times, .95)),
                "mean_model_steps": float(calls.mean()), "max_model_steps": int(calls.max()),
                "horizon_counts": {str(h): horizon_values.count(h) for h in (5, 10, 16)},
                "horizon_percent": {str(h): 100 * horizon_values.count(h) / len(horizon_values) for h in (5, 10, 16)},
            })
        adaptive = by_method["adaptive"]
        for method in METHODS[:-1]:
            baseline = by_method[method]
            if adaptive.keys() != baseline.keys():
                raise ValueError("Paired comparison requires identical seeds/scenarios")
            only_adaptive, only_baseline, both_success, both_failure = [], [], [], []
            for key, a in adaptive.items():
                b = baseline[key]
                identity = {"seed": key[0], "scenario": key[1]}
                target = both_success if a["success"] and b["success"] else (
                    only_adaptive if a["success"] else only_baseline if b["success"] else both_failure)
                target.append(identity)
            paired.append({"condition": condition, "baseline": method,
                           "only_adaptive_succeeds": only_adaptive, "only_baseline_succeeds": only_baseline,
                           "both_succeed": both_success, "both_fail": both_failure})
    improvement = []
    for h in (1, 5, 10, 16):
        m = {r["model"]: r for r in prediction["metrics"] if r["horizon"] == h}
        physical, learned = m["global_physics"]["rmse"], m["learned_world_model"]["rmse"]
        improvement.append({"horizon": h, "global_physics_rmse": physical, "learned_rmse": learned,
                            "relative_rmse_reduction_percent": 100 * (1 - learned / physical),
                            "windows": m["learned_world_model"]["windows"]})
    output = {"source": "Saved day-zero pilot; no episodes rerun", "episode_count": len(rows),
              "independent_training_runs": summary["training"]["independent_training_runs"],
              "total_seconds": summary["total_seconds"], "training_seconds": summary["training"]["training_seconds"],
              "selected_fixed_method": summary["decision"]["selected_fixed_method"],
              "metrics": results, "paired_outcomes": paired, "prediction_comparison": improvement,
              "limits": ["descriptive results only; no statistical superiority claim",
                         "160 possibly overlapping prediction windows are not 160 independent episodes",
                         "time quantiles pool planning calls and may differ with hardware/load"]}
    (directory / "analysis.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    plot(output, directory / "overview.png")
    print(json.dumps({"episodes": len(rows), "training_runs": output["independent_training_runs"],
                      "figure": str(directory / "overview.png"), "analysis": str(directory / "analysis.json")}, indent=2))


def plot(analysis, path):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 12, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.7), gridspec_kw={"width_ratios": [1.3, 1, 1.2]})
    colors = ["#2563eb", "#e87524"]
    width, locations = .36, np.arange(len(METHODS))
    for j, (condition, label) in enumerate(zip(CONDITIONS, ("Training dynamics", "Damping x1.7"))):
        selected = [next(r for r in analysis["metrics"] if r["condition"] == condition and r["method"] == method) for method in METHODS]
        percentages = [100 * r["successes"] / r["episodes"] for r in selected]
        offset = (j - .5) * width
        bars = axes[0].bar(locations + offset, percentages, width, label=label, color=colors[j])
        for bar, r in zip(bars, selected):
            axes[0].text(bar.get_x() + width / 2, bar.get_height() + 1.5,
                         f'{r["successes"]}/{r["episodes"]}', ha="center", fontsize=8, rotation=90)
        axes[2].plot(locations, [r["planning_ms_p95"] for r in selected], "o-", color=colors[j], label=label)
    axes[0].set(title="A  Navigation success", ylabel="Successful episodes (%)", ylim=(0, 119))
    axes[0].set_xticks(locations, LABELS, fontsize=8)
    axes[0].set_yticks([0, 25, 50, 75, 100])
    axes[0].legend(frameon=True, facecolor="white", framealpha=.95, edgecolor="none", loc="lower left", fontsize=9)
    horizons = [r["horizon"] for r in analysis["prediction_comparison"]]
    axes[1].plot(horizons, [r["global_physics_rmse"] for r in analysis["prediction_comparison"]], "s-", color="#64748b", label="Global physics")
    axes[1].plot(horizons, [r["learned_rmse"] for r in analysis["prediction_comparison"]], "o-", color="#08916c", label="Learned model")
    axes[1].set(title="B  Open-loop prediction", xlabel="Prediction horizon (steps)", ylabel="Robot position RMSE (map units)")
    axes[1].set_xticks(horizons)
    axes[1].legend(frameon=False)
    axes[1].text(.02, .67, "Held-out training dynamics\n160 windows; identical actions\nWindows can overlap", transform=axes[1].transAxes, color="#475569", fontsize=9)
    axes[2].set(title="C  Planning latency", ylabel="95th percentile (ms)", ylim=(0, max(r["planning_ms_p95"] for r in analysis["metrics"]) * 1.2))
    axes[2].set_xticks(locations, LABELS, fontsize=8)
    axes[2].legend(frameon=False, loc="upper left", fontsize=9)
    for axis in axes:
        axis.grid(axis="y", color="#e5e7eb", linewidth=.7)
        axis.set_axisbelow(True)
    fig.suptitle("Foresight Lab | Day-zero feasibility pilot", fontsize=18, fontweight="bold", x=.05, ha="left")
    fig.text(.05, .885, f'{analysis["episode_count"]} navigation episodes  |  one training run, three ensemble members  |  no superiority claim', color="#475569", fontsize=11)
    fig.text(.05, .025, "Budget: <= 4,500 member-transitions per decision, including adaptive probes. CPU timings; predictor families have different per-step costs.", fontsize=9, color="#475569")
    fig.subplots_adjust(left=.05, right=.99, bottom=.16, top=.80, wspace=.30)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="artifacts/day0")
    analyze(parser.parse_args().input)
