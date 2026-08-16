from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .config import ROOT
from .models import MetricsSummary


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    arr = np.sort(np.asarray(values, dtype=np.float64))
    if p >= 100:
        return float(arr[-1])
    idx = (p / 100.0) * (len(arr) - 1)
    lo = int(np.floor(idx))
    hi = int(np.ceil(idx))
    if lo == hi:
        return float(arr[lo])
    w = idx - lo
    return float(arr[lo] * (1 - w) + arr[hi] * w)


def summarize_latencies(
    totals: List[float],
    embed: List[float],
    retrieve: List[float],
    generate: List[float],
    source: str,
) -> MetricsSummary:
    n = len(totals)
    under = sum(1 for t in totals if t < 200.0)
    return MetricsSummary(
        n=n,
        p50_ms=percentile(totals, 50),
        p70_ms=percentile(totals, 70),
        p100_ms=percentile(totals, 100),
        mean_ms=float(np.mean(totals)) if totals else 0.0,
        under_200ms_pct=(100.0 * under / n) if n else 0.0,
        breakdown={
            "embed_ms_p50": percentile(embed, 50),
            "retrieve_ms_p50": percentile(retrieve, 50),
            "generate_ms_p50": percentile(generate, 50),
        },
        source=source,
    )


def metrics_path() -> Path:
    return ROOT / "artifacts" / "latency_metrics.json"


def load_metrics() -> Dict:
    path = metrics_path()
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_metrics(data: Dict) -> Path:
    path = metrics_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path
