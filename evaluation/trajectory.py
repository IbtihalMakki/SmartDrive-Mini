"""ADE/FDE in pixels over processed observations, not fixed-time metric motion."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def evaluate(records, output_dir, horizon=5, warmup=5):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for before, after in zip(records, records[1:]):
        if after["frame"] <= before["frame"] or after["timestamp"] <= before["timestamp"]:
            raise ValueError("Frames and simulation timestamps must strictly increase")
    observations = [{obj["track_id"]: obj for obj in rec["objects"]} for rec in records]
    for index in range(warmup-1, len(records)-horizon):
        for track, origin in observations[index].items():
            window = observations[index-warmup+1:index+horizon+1]
            if not all(track in frame for frame in window):
                continue  # No bridging missing tracks or evaluating truncated futures.
            if len({frame[track]["class_name"] for frame in window}) != 1:
                continue
            truth = np.asarray([frame[track]["center"] for frame in observations[index+1:index+horizon+1]], dtype=float)
            current = np.asarray(origin["predicted_centers"], dtype=float)
            baseline = np.tile(origin["center"], (horizon, 1)).astype(float)
            if current.shape != (horizon, 2) or not all(np.isfinite(a).all() for a in (truth, current, baseline)):
                continue
            for name, prediction in (("constant_position", baseline), ("current_predictor", current)):
                errors = np.linalg.norm(prediction-truth, axis=1)
                rows.append(dict(scenario=records[index]["scenario"], frame=records[index]["frame"],
                                 track_id=track, model=name, horizon_steps=horizon,
                                 horizon_seconds=records[index+horizon]["timestamp"]-records[index]["timestamp"],
                                 ade_px=float(errors.mean()), fde_px=float(errors[-1])))
    fields = ["scenario", "frame", "track_id", "model", "horizon_steps", "horizon_seconds", "ade_px", "fde_px"]
    with (output_dir / "trajectory_prediction_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(status="EVALUATED" if len(rows)//2 >= 20 else "INSUFFICIENT_DATA",
                   units="pixels", horizon="5 processed observations; variable elapsed time",
                   eligible_windows=len(rows)//2, recorded_frames=len(records),
                   ground_truth="future tracked image centers, not actor world coordinates",
                   models={})
    for name in ("constant_position", "current_predictor"):
        values = [r for r in rows if r["model"] == name]
        if values:
            summary["models"][name] = {
                "mean_ade_px": float(np.mean([r["ade_px"] for r in values])),
                "mean_fde_px": float(np.mean([r["fde_px"] for r in values])),
                "p95_ade_px": float(np.percentile([r["ade_px"] for r in values], 95)),
                "p95_fde_px": float(np.percentile([r["fde_px"] for r in values], 95)),
            }
    (output_dir / "trajectory_prediction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(json.dumps(evaluate(records, args.output), indent=2))
