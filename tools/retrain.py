#!/usr/bin/env python3
"""Stage C — wake-word retrain trigger (STUB).

Runs daily via ``calisto-retrain.timer``. The real retrain pipeline (Nabu
Casa openWakeWord training on accumulated negatives → new ``alexa.tflite``)
lands in a later stage once we have enough labelled samples. For now this
just counts and logs so we can see the dataset growing in journalctl + HA
without burning Colab/GPU cycles.

Exit codes
----------
0 — counted successfully (regardless of whether retrain would have run).
1 — capture dir unreadable / config error.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict

LOG = logging.getLogger("calisto-retrain")


def _summarise(capture_dir: Path) -> Dict[str, int]:
    counts = {
        "total": 0,
        "positive": 0,
        "negative": 0,
        "ambiguous": 0,
        "gate2_reject": 0,
        "unlabelled": 0,
        "since_last_run": 0,
    }
    marker = capture_dir / ".retrain-last-run"
    last_run_ts = 0.0
    if marker.exists():
        try:
            last_run_ts = float(marker.read_text().strip())
        except Exception:
            last_run_ts = 0.0
    for sidecar in capture_dir.glob("*.wav.json"):
        try:
            with open(sidecar) as f:
                data = json.load(f)
        except Exception:
            continue
        counts["total"] += 1
        label = data.get("label")
        if label is None:
            counts["unlabelled"] += 1
        elif label in counts:
            counts[label] += 1
        ts_epoch = data.get("ts_epoch") or 0
        if isinstance(ts_epoch, (int, float)) and ts_epoch >= last_run_ts:
            counts["since_last_run"] += 1
    return counts


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Wake-word retrain trigger (stub).")
    parser.add_argument(
        "--capture-dir",
        default=str(Path.home() / "wake_captures"),
        help="Directory holding *.wav.json sidecars.",
    )
    parser.add_argument(
        "--min-new-negatives",
        type=int,
        default=200,
        help="Threshold of new negatives at which a real retrain would fire.",
    )
    parser.add_argument(
        "--mark-run",
        action="store_true",
        help="Update .retrain-last-run on success.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    capture_dir = Path(args.capture_dir)
    if not capture_dir.is_dir():
        LOG.error("Capture dir does not exist: %s", capture_dir)
        return 1

    counts = _summarise(capture_dir)
    LOG.info(
        "Wake-capture dataset: total=%d positive=%d negative=%d ambiguous=%d "
        "gate2_reject=%d unlabelled=%d since_last_run=%d",
        counts["total"], counts["positive"], counts["negative"],
        counts["ambiguous"], counts["gate2_reject"], counts["unlabelled"],
        counts["since_last_run"],
    )

    if counts["negative"] >= args.min_new_negatives:
        LOG.info(
            "Would retrain: %d negatives ≥ threshold %d. "
            "(Stub — real pipeline lands in a later stage.)",
            counts["negative"], args.min_new_negatives,
        )
    else:
        LOG.info(
            "Below retrain threshold (%d / %d negatives). No-op.",
            counts["negative"], args.min_new_negatives,
        )

    if args.mark_run:
        marker = capture_dir / ".retrain-last-run"
        marker.write_text(f"{time.time():.3f}")
        LOG.info("Marked retrain run at %s", marker)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
