"""Create comparison figures from saved run logs (requires matplotlib)."""

import csv
import gzip
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent / "results"
GROUPS = (("regular", "coverage", "Regular A*"),
          ("regular", "random", "Regular random"),
          ("complex", "coverage", "Complex A*"),
          ("complex", "random", "Complex random"))
COLORS = ("#2463eb", "#e99734", "#188657", "#bd5a86")


def main():
    summary = {(row["scene"], row["mode"]): row for row in
               json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))}
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    labels = [label for _, _, label in GROUPS]
    x = range(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for ax, field, title, unit in (
            (axes[0], "coverage_percent", "Final coverage", "%"),
            (axes[1], "repeat_rate_percent", "Repeated cleaning path", "%")):
        values = [summary[s, m][field + "_mean"] for s, m, _ in GROUPS]
        errors = [summary[s, m][field + "_std"] for s, m, _ in GROUPS]
        ax.bar(x, values, yerr=errors, color=COLORS, capsize=4)
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.set_xticks(list(x), labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=.25)
    axes[0].axhline(95, color="#b63a3a", linestyle="--", label="95% target")
    axes[0].legend()
    axes[1].axhline(30, color="#b63a3a", linestyle="--", label="30% limit")
    axes[1].legend()
    successes = [summary[s, m]["successes"] for s, m, _ in GROUPS]
    axes[2].bar(x, successes, color=COLORS)
    axes[2].set_title("Successful runs")
    axes[2].set_ylabel("runs out of %d" % len(manifest["seeds"]))
    axes[2].set_ylim(0, len(manifest["seeds"]) + 1)
    axes[2].set_xticks(list(x), labels, rotation=25, ha="right")
    axes[2].grid(axis="y", alpha=.25)
    fig.savefig(figures / "comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    for ax, field, title, unit in (
            (axes[0], "cleaning_time_s", "Cleaning time", "s"),
            (axes[1], "cleaning_path_m", "Cleaning path length", "m")):
        values = [summary[s, m][field + "_mean"] for s, m, _ in GROUPS]
        errors = [summary[s, m][field + "_std"] for s, m, _ in GROUPS]
        ax.bar(x, values, yerr=errors, color=COLORS, capsize=4)
        ax.set(title=title, ylabel=unit)
        ax.set_xticks(list(x), labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=.25)
    axes[0].axhline(1200, color="#b63a3a", linestyle="--", label="1200 s limit")
    axes[0].legend()
    fig.savefig(figures / "time_path.png", dpi=160)
    plt.close(fig)

    times = list(range(0, 1201, 60))
    fig, ax = plt.subplots(figsize=(10, 5.3), layout="constrained")
    for (scene, mode, label), color in zip(GROUPS, COLORS):
        curves = []
        for seed in manifest["seeds"]:
            filename = ROOT / "logs" / ("%s_%s_seed%02d.csv.gz" % (scene, mode, seed))
            values = [0.0] * len(times)
            index = 1
            last = 0.0
            with gzip.open(str(filename), "rt", encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    t = float(row["time"])
                    last = float(row["coverage_percent"])
                    while index < len(times) and t >= times[index] - 1e-8:
                        values[index] = last
                        index += 1
            while index < len(times):
                values[index] = last
                index += 1
            curves.append(values)
        means = [statistics.mean(curve[k] for curve in curves) for k in range(len(times))]
        stds = [statistics.stdev(curve[k] for curve in curves) for k in range(len(times))]
        ax.plot(times, means, label=label, color=color, linewidth=2)
        ax.fill_between(times, [a - b for a, b in zip(means, stds)],
                        [a + b for a, b in zip(means, stds)], color=color, alpha=.12)
    ax.axhline(95, color="#b63a3a", linestyle="--", label="95% target")
    ax.set(xlabel="Simulation time (s)", ylabel="Coverage (%)", title="Coverage across 20 seeds")
    ax.set_xlim(0, 1200)
    ax.set_ylim(0, 100)
    ax.grid(alpha=.25)
    ax.legend(loc="upper left")
    fig.savefig(figures / "coverage_curves.png", dpi=160)
    plt.close(fig)
    print(figures / "comparison.png")
    print(figures / "time_path.png")
    print(figures / "coverage_curves.png")


if __name__ == "__main__":
    main()
