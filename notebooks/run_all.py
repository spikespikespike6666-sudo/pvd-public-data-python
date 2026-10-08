"""Run the three course notebooks without a Jupyter UI and record elapsed time."""

from __future__ import annotations

import csv
import json
import os
import platform
import time
from pathlib import Path


PROJECT = Path(__file__).resolve().parent.parent
NOTEBOOKS = [
    "01_data_quality_check.ipynb",
    "02_descriptive_statistics_visualization.ipynb",
    "03_pt_regression_validation.ipynb",
]
RESULTS = PROJECT / "outputs" / "pvd_public_data_v3"


def run_notebook(path: Path) -> float:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__", "__file__": str(path)}
    start = time.perf_counter()
    for cell_index, cell in enumerate(notebook["cells"], start=1):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        try:
            exec(compile(source, f"{path.name}:cell-{cell_index}", "exec"), namespace)
        except Exception as error:
            raise RuntimeError(f"{path.name}, cell {cell_index}: {error}") from error
    return time.perf_counter() - start


def main() -> None:
    os.chdir(PROJECT)
    start = time.perf_counter()
    import matplotlib

    matplotlib.use("Agg")
    timings = {name: run_notebook(PROJECT / "notebooks" / name) for name in NOTEBOOKS}
    total = time.perf_counter() - start

    with (RESULTS / "quality_checks" / "data_quality_summary.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        quality = list(csv.DictReader(handle))
    with (RESULTS / "pt_model" / "pt_loo_metrics.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        metrics = list(csv.DictReader(handle))
    failed = [row["check"] for row in quality if row["result"] != "PASS"]
    if failed:
        raise RuntimeError(f"Quality checks failed: {failed}")

    report = {
        "measurement": "Python process main() wall time, excluding interpreter/WSL startup",
        "python_version": platform.python_version(),
        "notebook_seconds": {key: round(value, 3) for key, value in timings.items()},
        "total_seconds": round(total, 3),
        "quality_passed": len(quality),
        "quality_failed": len(failed),
        "pt_loo_mae_nm": float(metrics[0]["MAE_nm"]),
        "pt_loo_r2": float(metrics[0]["R2"]),
    }
    destination = RESULTS / "runtime_validation.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Full notebook pipeline: {total:.3f} s")
    print(f"Quality checks: {len(quality)} passed, {len(failed)} failed")
    print(f"Pt LOO: MAE {report['pt_loo_mae_nm']:.3f} nm, R2 {report['pt_loo_r2']:.3f}")
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
