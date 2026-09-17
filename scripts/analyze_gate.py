"""Describe the frozen gate study from saved evidence; do not rerun models.

    python scripts/analyze_gate.py --input artifacts/gated-study

Writes only analysis.json, overview.png and overview.pdf in the input directory.
Counts and descriptive plots are not significance tests. Prediction windows
overlap; the same navigation layouts recur across models and conditions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHOD_LABELS = {
    "frozen_learned": "Frozen learned",
    "residual_calibrated": "Always calibrate",
    "gated_calibrated": "Gated calibrate",
    "local_identification": "Local physics",
}
CONDITION_LABELS = {
    "nominal": "Nominal",
    "global_shift": "Global x1.7",
    "local_patch": "Local patch",
    "switch_recover": "Shift / recover",
    "sensor_noise": "Sensor noise",
}
COLORS = ("#4169a1", "#e18a35", "#168a74", "#8592a5")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, message):
    require(bool(np.allclose(a, b, rtol=1e-8, atol=1e-10)), message)


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def delays(values):
    values = sorted(values)
    return {"n": len(values), "observations": values,
            "median": float(np.median(values)) if values else None,
            "p95": float(np.quantile(values, .95)) if values else None,
            "max": max(values) if values else None}


def event_record(row, shift_step, recovery_step):
    """Compute observed event endpoints and distinguish censoring from misses."""
    trace = row["trace"]
    condition = row["condition"]
    start = 0 if condition == "global_shift" else shift_step
    end = len(trace) if condition == "global_shift" else min(recovery_step, len(trace))
    exposed = len(trace) > start
    pre_active = bool(trace[start - 1]["gate_after"]) if start and len(trace) >= start else False
    after = [t["step"] for t in trace if start <= t["step"] < end and t["gate_after"]]
    latency = after[0] - start + 1 if after and not pre_active else None
    if not exposed:
        status = "not_exposed"
    elif pre_active:
        status = "pre_active"
    elif latency is not None:
        status = "detected"
    elif condition == "switch_recover" and len(trace) >= recovery_step:
        status = "missed_full_shift_window"
    else:
        status = "censored_no_detection_before_episode_end"

    recovery_exposed = condition == "switch_recover" and len(trace) > recovery_step
    active_on_return = (bool(trace[recovery_step - 1]["gate_after"])
                        if condition == "switch_recover" and len(trace) >= recovery_step else None)
    off = [t["step"] for t in trace if t["step"] >= recovery_step and not t["gate_after"]]
    recovery_latency = (off[0] - recovery_step + 1
                        if recovery_exposed and active_on_return and off else None)
    if condition != "switch_recover":
        recovery_status = "not_applicable"
    elif not recovery_exposed:
        recovery_status = "not_exposed"
    elif not active_on_return:
        recovery_status = "already_inactive"
    elif recovery_latency is not None:
        recovery_status = "deactivated"
    else:
        recovery_status = "censored_no_deactivation_before_episode_end"

    return {
        "training_seed": row["training_seed"], "condition": condition, "seed": row["seed"],
        "exposed_shift": exposed, "already_active_before_shift": pre_active,
        "detection_observations": latency,
        "shift_observation_count": max(0, end - start),
        "exposed_recovery": recovery_exposed, "active_before_recovery": active_on_return,
        "recovery_observations": recovery_latency,
        "detection_status": status, "recovery_status": recovery_status,
        "recovery_observation_count": max(0, len(trace) - recovery_step) if recovery_exposed else 0,
    }


def event_stats(records):
    detection_counts = Counter(r["detection_status"] for r in records)
    recovery_counts = Counter(r["recovery_status"] for r in records)
    exposed = sum(r["exposed_shift"] for r in records)
    eligible = sum(r["exposed_shift"] and not r["already_active_before_shift"] for r in records)
    recovery_exposed = sum(r["exposed_recovery"] for r in records)
    recovery_eligible = sum(r["exposed_recovery"] and bool(r["active_before_recovery"]) for r in records)
    return {
        "episode_runs": len(records),
        "shift_exposed": exposed, "shift_eligible_not_pre_active": eligible,
        "detection_counts": dict(detection_counts),
        "detected_fraction_of_eligible": fraction(detection_counts["detected"], eligible),
        "detection_delay_conditional_on_observed_detection": delays(
            [r["detection_observations"] for r in records if r["detection_status"] == "detected"]),
        "recovery_exposed": recovery_exposed, "recovery_eligible_active_at_return": recovery_eligible,
        "recovery_counts": dict(recovery_counts),
        "deactivated_fraction_of_eligible": fraction(recovery_counts["deactivated"], recovery_eligible),
        "recovery_delay_conditional_on_observed_deactivation": delays(
            [r["recovery_observations"] for r in records if r["recovery_status"] == "deactivated"]),
    }


def analyze(directory):
    directory = Path(directory)
    protocol = read(directory / "protocol.json")
    summary = read(directory / "summary.json")
    saved_detection = read(directory / "detection.json")
    seeds, conditions, methods = (protocol[k] for k in ("training_seeds", "conditions", "methods"))
    episode_paths = [directory / f"seed{s}-episodes.json" for s in seeds]
    rows = [row for path in episode_paths for row in read(path)]
    n = len(protocol["navigation_seeds"])
    expected = len(seeds) * len(conditions) * len(methods) * n
    identities = [(r["training_seed"], r["condition"], r["method"], r["seed"]) for r in rows]
    require(len(rows) == expected == summary["total_episode_runs"], "Episode count does not match protocol")
    require(len(set(identities)) == len(rows), "Duplicate episode identities")
    for row in rows:
        require(row["steps"] == len(row["trace"]), "Trace length mismatch")
        require([t["step"] for t in row["trace"]] == list(range(row["steps"])), "Non-contiguous trace")
        require(sum(bool(row[k]) for k in ("success", "collision", "timeout")) == 1, "Invalid terminal reason")
        require(all(t["model_steps"] <= protocol["budget"] for t in row["trace"]), "Planner budget exceeded")

    metrics = []
    for condition in conditions:
        for method in methods:
            selected = [r for r in rows if r["condition"] == condition and r["method"] == method]
            per_seed = []
            for seed in seeds:
                panel = [r for r in selected if r["training_seed"] == seed]
                require(sorted(r["seed"] for r in panel) == sorted(protocol["navigation_seeds"]), "Panel layouts mismatch")
                calls = [t for r in panel for t in r["trace"]]
                per_seed.append({
                    "training_seed": seed, "episode_runs": len(panel),
                    "successes": sum(r["success"] for r in panel),
                    "collisions": sum(r["collision"] for r in panel),
                    "timeouts": sum(r["timeout"] for r in panel),
                    "controller_ms_p95": float(np.quantile([t["controller_ms"] for t in calls], .95)),
                    "gate_any_activation_episodes": sum(any(t["gate_after"] for t in r["trace"]) for r in panel)
                    if method == "gated_calibrated" else None,
                })
            calls = [t for r in selected for t in r["trace"]]
            output = {
                "condition": condition, "method": method, "episode_runs": len(selected),
                "unique_layouts": n, "per_seed": per_seed,
                "success_counts_by_seed": [r["successes"] for r in per_seed],
                "mean_success_rate": sum(r["success"] for r in selected) / len(selected),
                "controller_ms_p50": float(np.median([t["controller_ms"] for t in calls])),
                "controller_ms_p95": float(np.quantile([t["controller_ms"] for t in calls], .95)),
                "controller_call_count": len(calls),
                "mean_model_steps": float(np.mean([t["model_steps"] for t in calls])),
                "mean_update_member_evals": float(np.mean([t["update_member_evals"] for t in calls])),
            }
            saved = next(r for r in summary["results"] if r["condition"] == condition and r["method"] == method)
            require(output["success_counts_by_seed"] == saved["success_counts_by_seed"], "Summary success mismatch")
            close(output["controller_ms_p95"], saved["controller_ms_p95"], "Summary timing mismatch")
            if method == "gated_calibrated":
                active_after = sum(t["gate_after"] for t in calls)
                active_before = sum(t["gate_before"] for t in calls)
                any_episode = sum(r["gate_any_activation_episodes"] for r in per_seed)
                output["gate"] = {
                    "any_activation_episodes": any_episode, "episode_run_denominator": len(selected),
                    "any_activation_episode_fraction": any_episode / len(selected),
                    "active_after_observe_steps": active_after, "active_before_action_steps": active_before,
                    "step_denominator": len(calls), "active_after_observe_step_fraction": active_after / len(calls),
                    "active_before_action_step_fraction": active_before / len(calls),
                    "interpretation": "False activation under nominal; noisy-nominal activation under sensor_noise; activation only for other conditions",
                }
                require(any_episode == saved["any_gate_activation_episodes"], "Summary activation mismatch")
                close(active_after / len(calls), saved["gate_active_step_fraction"], "Summary gate step mismatch")
            metrics.append(output)

    # Prediction errors are recomputed from stored windows, never from rounded plot values.
    pred_path = directory / "prediction-windows.json"
    prediction_raw = read(pred_path)
    prediction_keys = [(r["training_seed"], r["condition"], r["method"], r["episode_seed"], r["start_step"], r["horizon"])
                       for r in prediction_raw]
    require(len(prediction_keys) == len(set(prediction_keys)), "Duplicate prediction windows")
    prediction = []
    for condition in conditions:
        reference_keys = None
        for method in methods:
            selected = [r for r in prediction_raw if r["condition"] == condition and r["method"] == method and r["horizon"] == 16]
            values, counts, episode_counts, active_counts = [], [], [], []
            for seed in seeds:
                panel = [r for r in selected if r["training_seed"] == seed]
                keys = {(r["episode_seed"], r["start_step"]) for r in panel}
                if reference_keys is None:
                    reference_keys = keys
                require(keys == reference_keys and bool(keys), "Prediction methods do not share paired windows")
                require(all(r["episode_seed"] in protocol["prediction_seeds"] for r in panel), "Unexpected prediction seed")
                values.append(float(np.sqrt(np.mean([r["position_squared_error"] for r in panel]))))
                counts.append(len(panel))
                episode_counts.append(len({r["episode_seed"] for r in panel}))
                active_counts.append(sum(r["gate_active"] for r in panel))
            saved = next(r for r in summary["prediction"] if r["condition"] == condition and r["method"] == method and r["horizon"] == 16)
            close(values, saved["rmse_by_training_seed"], "Summary H16 RMSE mismatch")
            prediction.append({"condition": condition, "method": method, "horizon": 16,
                               "rmse_by_training_seed": values, "mean_rmse": float(np.mean(values)),
                               "windows_per_seed": counts, "contributing_episodes_per_seed": episode_counts,
                               "generated_episodes_per_condition": len(protocol["prediction_seeds"]),
                               "gate_active_at_forecast_windows_by_seed": active_counts if method == "gated_calibrated" else None,
                               "gate_active_forecast_window_fraction": sum(active_counts) / sum(counts) if method == "gated_calibrated" else None})

    shift_range = re.search(r"transitions\s*(\d+)\.\.(\d+)", protocol["condition_definitions"]["switch_recover"])
    require(shift_range is not None, "Cannot infer frozen transition schedule")
    shift_step, recovery_step = int(shift_range[1]), int(shift_range[2]) + 1
    events = [event_record(r, shift_step, recovery_step) for r in rows
              if r["method"] == "gated_calibrated" and r["condition"] in ("global_shift", "switch_recover")]
    event_key = lambda r: (r["training_seed"], r["condition"], r["seed"])
    saved_by_key = {event_key(r): r for r in saved_detection}
    require(len(saved_by_key) == len(saved_detection) == len(events), "Detection record count mismatch")
    for event in events:
        saved = saved_by_key[event_key(event)]
        require(all(event[k] == v for k, v in saved.items()), "Detection record disagrees with trace")
    event_summary = [{"condition": condition, **event_stats([r for r in events if r["condition"] == condition]),
                      "per_seed": [{"training_seed": seed, **event_stats([r for r in events if r["condition"] == condition and r["training_seed"] == seed])}
                                   for seed in seeds]}
                     for condition in ("global_shift", "switch_recover")]

    # The physics method has no trained seed. Check repeat panels without timings.
    def physics_signature(row):
        values = {k: v for k, v in row.items() if k not in ("training_seed", "trace")}
        values["trace"] = [{k: v for k, v in t.items() if not k.endswith("_ms")} for t in row["trace"]]
        return json.dumps(values, sort_keys=True)

    physics = {(r["training_seed"], r["condition"], r["seed"]): physics_signature(r)
               for r in rows if r["method"] == "local_identification"}
    physics_identical = all(physics[(seed, c, s)] == physics[(seeds[0], c, s)]
                            for seed in seeds for c in conditions for s in protocol["navigation_seeds"])
    sources = [directory / name for name in ("protocol.json", "summary.json", "detection.json", "calibration.json")]
    sources += episode_paths + [pred_path]
    analysis = {
        "study": "Frozen gate calibration study; descriptive analysis only",
        "training_seeds": seeds, "conditions": conditions, "methods": methods,
        "total_episode_runs": len(rows), "unique_navigation_layouts": n,
        "independent_learning_runs": len(seeds),
        "physics_panels_identical_except_timing": physics_identical,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sanity_checks": {"expected_episode_runs": expected, "unique_episode_keys": len(set(identities)),
                          "paired_prediction_windows": True, "summary_matches_raw": True,
                          "detection_matches_raw": True, "terminal_reasons_valid": True,
                          "within_planning_budget": True},
        "metrics": metrics, "prediction_h16": prediction, "events": event_summary,
        "event_records": events,
        "event_timing": {"shift_transition_zero_based": shift_step, "recovery_transition_zero_based": recovery_step,
                         "unit": "Completed observations since event; detection after observe affects the next decision",
                         "conditional_delay_warning": "Delay summaries include only observed endpoints. See exposed, pre-active, missed and censored counts."},
        "limits": [
            "Three trained-model panels reuse the same layouts; pooled runs are not independent layouts.",
            "Local physics has no learned seed; its three deterministic repeat panels add no independent performance evidence.",
            "Prediction windows overlap; RMSE averages windows, then the three training-seed RMSE values.",
            "Only episodes long enough for a 16-step rollout contribute prediction windows.",
            "Prediction uses separate behavior-policy episodes; their gate activation differs from MPC navigation.",
            "Open-loop forecasts do not know scheduled future changes or receive intermediate observations.",
            "P95 controller latency pools control calls and includes calibration updates; it depends on hardware/load.",
            "No threshold selection, significance test or general safety claim is made by this analysis.",
            "Recovery is observed deactivation, not proof of permanently correct calibration thereafter.",
            "Sensor noise occurs under nominal dynamics; activation is not evidence of an actual dynamics change.",
        ],
    }
    (directory / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plot(analysis, directory)
    print(json.dumps({"episode_runs": len(rows), "unique_layouts": n,
                      "physics_repeat_panels_identical": physics_identical,
                      "outputs": [str(directory / p) for p in ("analysis.json", "overview.png", "overview.pdf")]}, indent=2))
    return analysis


def plot(analysis, directory):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 12, "axes.titleweight": "bold",
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.7))
    conditions, methods = analysis["conditions"], analysis["methods"]
    x = np.arange(len(conditions))
    metric = {(r["condition"], r["method"]): r for r in analysis["metrics"]}
    prediction = {(r["condition"], r["method"]): r for r in analysis["prediction_h16"]}
    for index, method in enumerate(methods):
        offset = (index - 1.5) * .18
        xs = x + offset
        pred = [prediction[(c, method)] for c in conditions]
        axes[0, 0].bar(xs, [r["mean_rmse"] for r in pred], .17, color=COLORS[index], alpha=.9, label=METHOD_LABELS[method])
        for pos, row in zip(xs, pred):
            axes[0, 0].scatter(pos + np.linspace(-.035, .035, 3), row["rmse_by_training_seed"], s=13, c="#1e293b", linewidths=.5, edgecolors="white", zorder=3)
        rates = [metric[(c, method)]["mean_success_rate"] * 100 for c in conditions]
        axes[0, 1].plot(xs, rates, "o-", color=COLORS[index], markersize=4, linewidth=1.1)
        for pos, condition in zip(xs, conditions):
            values = [100 * r["successes"] / r["episode_runs"] for r in metric[(condition, method)]["per_seed"]]
            axes[0, 1].scatter(pos + np.linspace(-.025, .025, 3), values, s=14, color=COLORS[index], edgecolors="white", linewidths=.4, zorder=3)
    axes[0, 0].set(title="A  16-step open-loop prediction", ylabel="Position RMSE (map units; log scale)", yscale="log")
    axes[0, 1].set(title="B  Navigation success", ylabel="Success (%)", ylim=(0, 105))
    axes[0, 1].set_yticks([0, 25, 50, 75, 100])
    for ax in axes[0]:
        ax.set_xticks(x, [CONDITION_LABELS[c] for c in conditions], fontsize=9)

    gates = [metric[(c, "gated_calibrated")]["gate"] for c in conditions]
    episode_rates = [100 * g["any_activation_episode_fraction"] for g in gates]
    used_rates = [100 * g["active_before_action_step_fraction"] for g in gates]
    bars = axes[1, 0].bar(x - .17, episode_rates, .32, color="#168a74", label="Any activation in episode")
    axes[1, 0].bar(x + .17, used_rates, .32, color="#94c9bd", label="Active when choosing action")
    for bar, gate in zip(bars, gates):
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                        f'{gate["any_activation_episodes"]}/{gate["episode_run_denominator"]}', ha="center", fontsize=8)
    axes[1, 0].set(title="C  Gate use (episode and step denominators)", ylabel="Observed fraction (%)", ylim=(0, 133))
    axes[1, 0].set_yticks([0, 25, 50, 75, 100])
    axes[1, 0].set_xticks(x, [CONDITION_LABELS[c] for c in conditions], fontsize=9)
    axes[1, 0].legend(frameon=False, fontsize=9, loc="upper left")

    # A text panel preserves exposure denominators and censored outcomes.
    ax = axes[1, 1]
    ax.axis("off")
    ax.set_title("D  Observed change response", loc="left", pad=14)
    lines = []
    for event in analysis["events"]:
        counts = event["detection_counts"]
        delay = event["detection_delay_conditional_on_observed_detection"]
        title = "Global x1.7 from start" if event["condition"] == "global_shift" else "Within-episode shift (step 8)"
        median = f'{delay["median"]:g}' if delay["median"] is not None else "n/a"
        lines.extend([
            (f'{title}: {counts.get("detected", 0)}/{event["shift_eligible_not_pre_active"]} eligible detected', True),
            (f'Exposed {event["shift_exposed"]}/{event["episode_runs"]}; pre-active {counts.get("pre_active", 0)}; median delay {median} observations', False),
            (f'Full-window misses {counts.get("missed_full_shift_window", 0)}; censored {counts.get("censored_no_detection_before_episode_end", 0)}', False),
        ])
        if event["condition"] == "switch_recover":
            recovery = event["recovery_counts"]
            d = event["recovery_delay_conditional_on_observed_deactivation"]
            median = f'{d["median"]:g}' if d["median"] is not None else "n/a"
            lines.extend([
                (f'Return to nominal (step 24): {recovery.get("deactivated", 0)}/{event["recovery_eligible_active_at_return"]} eligible deactivated', True),
                (f'Exposed {event["recovery_exposed"]}/{event["episode_runs"]}; already inactive {recovery.get("already_inactive", 0)}; median {median} observations', False),
                (f'No observed deactivation before episode end: {recovery.get("censored_no_deactivation_before_episode_end", 0)}', False),
            ])
    y = .99
    for text, bold in lines:
        if bold and y < .99:
            y -= .065
        ax.text(0, y, text, transform=ax.transAxes, va="top", fontsize=10 if bold else 9,
                color="#17283d" if bold else "#475569", fontweight="bold" if bold else "normal")
        y -= .078
    ax.text(0, .015, "Delays condition on observed endpoints. Terminal censoring is retained.\nP95 controller timings and every seed's counts are in analysis.json.",
            transform=ax.transAxes, fontsize=8.5, color="#475569", va="bottom")
    for ax in axes.flat:
        if ax.axison:
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#e2e8f0", linewidth=.7)
    fig.suptitle("Foresight Lab | Selective calibration under changing dynamics", x=.065, ha="left", fontsize=18, fontweight="bold", y=.985)
    fig.text(.065, .948, f'{analysis["total_episode_runs"]:,} method episode runs | {analysis["unique_navigation_layouts"]} paired layouts per condition | 3 frozen training seeds | descriptive results', fontsize=10.5, color="#475569")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(.06, .932), ncol=4, frameon=False, fontsize=10)
    fig.text(.065, .031, "Dots show individual training seeds, not confidence intervals. Physics panels repeat the same deterministic method; prediction windows overlap.", fontsize=9, color="#475569")
    fig.text(.065, .012, "MPC: H=10, 4,500 member-transitions per decision; calibration work counted separately. Gate activation under sensor noise is not a true dynamics event.", fontsize=9, color="#475569")
    fig.subplots_adjust(left=.065, right=.985, top=.87, bottom=.1, hspace=.35, wspace=.23)
    fig.savefig(directory / "overview.png", dpi=180)
    fig.savefig(directory / "overview.pdf", metadata={"Title": "Foresight Lab: gate study descriptive results", "Author": "Foresight Lab", "CreationDate": None, "ModDate": None})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(ROOT / "artifacts/gated-study"))
    analyze(parser.parse_args().input)
