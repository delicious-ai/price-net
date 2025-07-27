from argparse import ArgumentParser
from pathlib import Path

import polars as pl
import yaml
from sklearn.base import defaultdict

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--threshold", default=0.5)
    args = parser.parse_args()
    threshold = float(args.threshold)

    results_dir = Path("results/association")
    all_results = {
        "method": [],
        "trial": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "auroc": [],
        "aupr": [],
    }
    precisions = defaultdict(list)
    recalls = defaultdict(list)
    f1s = defaultdict(list)
    for method_subdir in results_dir.iterdir():
        if not method_subdir.name.startswith("pricenet"):
            continue

        method_name, trial = method_subdir.name.rsplit("-", 1)

        metrics_path = method_subdir / "association_metrics.yaml"
        with open(metrics_path, "r") as f:
            metrics = yaml.safe_load(f)

        precision_key = f"precision@{threshold}"
        recall_key = f"recall@{threshold}"
        f1_key = f"f1@{threshold}"
        all_results["method"].append(method_name)
        all_results["trial"].append(trial)
        all_results["precision"].append(metrics[precision_key])
        all_results["recall"].append(metrics[recall_key])
        all_results["f1"].append(metrics[f1_key])
        all_results["auroc"].append(metrics["roc-auc"])
        all_results["aupr"].append(metrics["aupr"])

    all_results = pl.DataFrame(all_results)
    all_results = all_results.group_by(["method"]).agg(
        pl.col("precision").mean().alias("precision_mean"),
        pl.col("precision").std().alias("precision_std"),
        pl.col("recall").mean().alias("recall_mean"),
        pl.col("recall").std().alias("recall_std"),
        pl.col("f1").mean().alias("f1_mean"),
        pl.col("f1").std().alias("f1_std"),
        pl.col("auroc").mean().alias("auroc_mean"),
        pl.col("auroc").std().alias("auroc_std"),
        pl.col("aupr").mean().alias("aupr_mean"),
        pl.col("aupr").std().alias("aupr_std"),
    )
    print(all_results)
    all_results.write_csv("combined_pricenet_results.csv")
