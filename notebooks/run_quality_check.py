from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


DATA_FILENAME = "pvd_experiments_verified_v3.csv"
EXPECTED_ROWS = 30

REQUIRED_COLUMNS = [
    "experiment_id",
    "source_id",
    "source_locator",
    "target_material",
    "deposition_method",
    "deposition_time_min",
    "pressure_Pa",
    "film_thickness_nm",
    "substrate",
    "source_doi",
    "source_url",
    "thickness_status",
    "verification_status",
]

NUMERIC_COLUMNS = [
    "deposition_time_min",
    "pressure_Pa",
    "film_thickness_nm",
    "power_W",
    "discharge_current_mA",
    "substrate_temperature_C",
    "deposition_rate_nm_min",
    "original_deposition_time_s",
]

OPTIONAL_COLUMNS = [
    "power_W",
    "discharge_current_mA",
    "substrate_temperature_C",
    "substrate_temperature_label",
    "deposition_rate_nm_min",
    "original_deposition_time_s",
    "comments",
]


def locate_data_file() -> Path:
    """Locate the verified CSV without relying on a machine-specific absolute path."""
    start = Path.cwd().resolve()
    roots = [start, *start.parents]
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            [
                root / DATA_FILENAME,
                root / "outputs" / "pvd_public_data_v3" / DATA_FILENAME,
                root / "项目名" / "outputs" / "pvd_public_data_v3" / DATA_FILENAME,
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    checked = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"未找到 {DATA_FILENAME}。已检查：\n{checked}")


def build_quality_report(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    checks: list[dict[str, object]] = []
    issues: list[dict[str, object]] = []

    def add_check(name: str, passed: bool, observed: object, expected: str, details: str) -> None:
        checks.append(
            {
                "check": name,
                "result": "PASS" if passed else "FAIL",
                "observed": observed,
                "expected": expected,
                "details": details,
            }
        )

    add_check(
        "row_count",
        len(data) == EXPECTED_ROWS,
        len(data),
        str(EXPECTED_ROWS),
        "主数据表应包含30条已核实实验记录。",
    )

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    add_check(
        "required_columns_present",
        not missing_columns,
        len(REQUIRED_COLUMNS) - len(missing_columns),
        f"{len(REQUIRED_COLUMNS)} required columns",
        "缺失列：" + (", ".join(missing_columns) if missing_columns else "无"),
    )

    if missing_columns:
        return pd.DataFrame(checks), pd.DataFrame(issues), pd.DataFrame()

    id_text = data["experiment_id"].astype("string").str.strip()
    missing_id_mask = id_text.isna() | id_text.eq("")
    duplicate_id_mask = id_text.duplicated(keep=False) & ~missing_id_mask
    add_check(
        "experiment_id_complete",
        not missing_id_mask.any(),
        int(missing_id_mask.sum()),
        "0 missing IDs",
        "每条记录必须有实验编号。",
    )
    add_check(
        "experiment_id_unique",
        not duplicate_id_mask.any(),
        int(duplicate_id_mask.sum()),
        "0 duplicated rows",
        "实验编号必须唯一。",
    )

    for index in data.index[missing_id_mask]:
        issues.append({"row": int(index) + 2, "experiment_id": "", "check": "missing_experiment_id", "details": "实验编号为空"})
    for index in data.index[duplicate_id_mask]:
        issues.append(
            {
                "row": int(index) + 2,
                "experiment_id": data.at[index, "experiment_id"],
                "check": "duplicate_experiment_id",
                "details": "实验编号重复",
            }
        )

    required_missing = data[REQUIRED_COLUMNS].isna() | data[REQUIRED_COLUMNS].astype("string").apply(lambda column: column.str.strip().eq(""))
    required_missing_count = int(required_missing.to_numpy().sum())
    add_check(
        "required_values_complete",
        required_missing_count == 0,
        required_missing_count,
        "0 missing required values",
        "功率和电流按单独规则检查，其余必填字段不得为空。",
    )
    for index, column in zip(*required_missing.to_numpy().nonzero()):
        issues.append(
            {
                "row": int(index) + 2,
                "experiment_id": data.iloc[index]["experiment_id"],
                "check": "missing_required_value",
                "details": f"{REQUIRED_COLUMNS[column]} 为空",
            }
        )

    numeric_failures: dict[str, int] = {}
    numeric_data: dict[str, pd.Series] = {}
    for column in NUMERIC_COLUMNS:
        if column not in data.columns:
            continue
        converted = pd.to_numeric(data[column], errors="coerce")
        numeric_data[column] = converted
        invalid_mask = data[column].notna() & converted.isna()
        numeric_failures[column] = int(invalid_mask.sum())
        for index in data.index[invalid_mask]:
            issues.append(
                {
                    "row": int(index) + 2,
                    "experiment_id": data.at[index, "experiment_id"],
                    "check": "non_numeric_value",
                    "details": f"{column} 不是有效数值：{data.at[index, column]}",
                }
            )
    numeric_failure_count = sum(numeric_failures.values())
    add_check(
        "numeric_fields_parseable",
        numeric_failure_count == 0,
        numeric_failure_count,
        "0 invalid numeric values",
        "; ".join(f"{key}: {value}" for key, value in numeric_failures.items()),
    )

    positive_columns = ["deposition_time_min", "pressure_Pa", "film_thickness_nm"]
    nonpositive_count = 0
    for column in positive_columns:
        converted = numeric_data[column]
        bad_mask = converted.notna() & converted.le(0)
        nonpositive_count += int(bad_mask.sum())
        for index in data.index[bad_mask]:
            issues.append(
                {
                    "row": int(index) + 2,
                    "experiment_id": data.at[index, "experiment_id"],
                    "check": "nonpositive_physical_value",
                    "details": f"{column} 必须大于0，当前值为 {data.at[index, column]}",
                }
            )
    add_check(
        "core_physical_values_positive",
        nonpositive_count == 0,
        nonpositive_count,
        "0 non-positive values",
        "沉积时间、工作压力和膜厚必须大于0。",
    )

    power = numeric_data.get("power_W", pd.Series(index=data.index, dtype=float))
    current = numeric_data.get("discharge_current_mA", pd.Series(index=data.index, dtype=float))
    no_electrical_parameter = power.isna() & current.isna()
    invalid_power = power.notna() & power.le(0)
    invalid_current = current.notna() & current.le(0)
    electrical_bad = no_electrical_parameter | invalid_power | invalid_current
    add_check(
        "power_or_current_available",
        not electrical_bad.any(),
        int(electrical_bad.sum()),
        "0 invalid rows",
        "每条记录至少需要一个正的功率或放电电流值。",
    )
    for index in data.index[electrical_bad]:
        issues.append(
            {
                "row": int(index) + 2,
                "experiment_id": data.at[index, "experiment_id"],
                "check": "electrical_parameter",
                "details": "功率和电流均缺失，或存在非正值",
            }
        )

    doi_pattern = re.compile(r"^10\.\d{4,9}/\S+$")
    doi_text = data["source_doi"].astype("string").str.strip()
    url_text = data["source_url"].astype("string").str.strip()
    bad_doi = ~doi_text.fillna("").map(lambda value: bool(doi_pattern.match(value)))
    bad_url = ~url_text.fillna("").str.startswith(("https://", "http://"))
    source_bad = bad_doi | bad_url
    add_check(
        "source_identifiers_valid",
        not source_bad.any(),
        int(source_bad.sum()),
        "0 invalid rows",
        "DOI格式和来源URL必须可识别。",
    )
    for index in data.index[source_bad]:
        issues.append(
            {
                "row": int(index) + 2,
                "experiment_id": data.at[index, "experiment_id"],
                "check": "source_identifier",
                "details": "DOI或URL格式无效",
            }
        )

    bad_verification = data["verification_status"].astype("string").str.strip().ne("verified")
    bad_thickness = data["thickness_status"].astype("string").str.strip().ne("reported_measured")
    status_bad = bad_verification | bad_thickness
    add_check(
        "verification_status_consistent",
        not status_bad.any(),
        int(status_bad.sum()),
        "0 rows outside verified/measured status",
        "主表仅允许 verified 且 reported_measured 的记录。",
    )
    for index in data.index[status_bad]:
        issues.append(
            {
                "row": int(index) + 2,
                "experiment_id": data.at[index, "experiment_id"],
                "check": "verification_status",
                "details": "记录状态不符合主表规则",
            }
        )

    optional_summary = pd.DataFrame(
        [
            {
                "column": column,
                "missing_count": int(data[column].isna().sum()),
                "available_count": int(data[column].notna().sum()),
                "note": "允许缺失，不计为质量错误",
            }
            for column in OPTIONAL_COLUMNS
            if column in data.columns
        ]
    )

    issue_columns = ["row", "experiment_id", "check", "details"]
    return pd.DataFrame(checks), pd.DataFrame(issues, columns=issue_columns), optional_summary


def main() -> None:
    data_path = locate_data_file()
    output_dir = data_path.parent / "quality_checks"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(data_path)
    summary, issues, optional_summary = build_quality_report(data)

    summary.to_csv(output_dir / "data_quality_summary.csv", index=False, encoding="utf-8-sig")
    issues.to_csv(output_dir / "data_quality_issues.csv", index=False, encoding="utf-8-sig")
    optional_summary.to_csv(output_dir / "optional_missingness.csv", index=False, encoding="utf-8-sig")
    issues.to_csv(data_path.parent / "error_log.csv", index=False, encoding="utf-8-sig")
    if summary["result"].eq("PASS").all():
        data.to_csv(data_path.parent / "cleaned_pvd_data.csv", index=False, encoding="utf-8-sig")
    else:
        print("Quality checks failed; cleaned_pvd_data.csv was not exported.")

    report = {
        "data_file": data_path.name,
        "row_count": int(len(data)),
        "column_count": int(len(data.columns)),
        "passed_checks": int(summary["result"].eq("PASS").sum()),
        "failed_checks": int(summary["result"].eq("FAIL").sum()),
        "issue_count": int(len(issues)),
        "sources": {str(key): int(value) for key, value in data["source_id"].value_counts().sort_index().items()},
        "materials": {str(key): int(value) for key, value in data["target_material"].value_counts().sort_index().items()},
    }
    (output_dir / "data_quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Data file: {data_path}")
    print(f"Rows: {len(data)} | Columns: {len(data.columns)}")
    print(summary.to_string(index=False))
    print(f"\nIssues: {len(issues)}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
