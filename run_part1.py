from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from music_recsys import (
    S3Config,
    build_items,
    inspect_raw_data,
    load_raw_data,
    metrics_to_frame,
    run_offline_pipeline_from_files,
    save_stage2_outputs,
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    s3_config = S3Config.from_env()

    print("Loading raw parquet files...")
    tracks, catalog_names, interactions = load_raw_data(base_dir)

    print("Inspecting data quality...")
    summary, details = inspect_raw_data(tracks, catalog_names, interactions)
    items, missing_genres = build_items(tracks, catalog_names)

    print("Saving stage 2 outputs...")
    stage2_paths = save_stage2_outputs(items, interactions, base_dir, s3_config)

    del tracks, catalog_names, interactions

    print("Running offline recommendation pipeline...")
    offline_result = run_offline_pipeline_from_files(
        base_dir / "items.parquet",
        base_dir / "events.parquet",
        base_dir,
        s3_config,
    )
    metrics_frame = metrics_to_frame(offline_result["metrics"])

    report = {
        "summary": summary.to_dict(orient="records"),
        "details": details,
        "missing_genres": missing_genres,
        "stage2_paths": stage2_paths,
        "offline_paths": offline_result["saved_paths"],
        "metrics": offline_result["metrics"],
    }

    report_path = base_dir / "part1_report.json"
    report_path.write_text(
        json.dumps(_json_ready(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nData summary:")
    print(summary.to_string(index=False))
    print("\nMetrics:")
    print(metrics_frame.to_string(index=False))
    print(f"\nSaved report to {report_path}")


if __name__ == "__main__":
    main()
