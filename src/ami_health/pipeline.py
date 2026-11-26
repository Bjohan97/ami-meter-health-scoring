"""End-to-end scoring of a fleet."""

from __future__ import annotations

import pandas as pd

from . import scores
from .config import Settings
from .diagnosis import diagnose
from .events import event_summary
from .features import meter_features
from .inputs import Inputs

HEALTH_COLUMNS = ["priority", "meter_id", "group", "diagnosis", "diagnosis_reason", "action", "health_score",
                  "coverage_of_score", "modules_used", "availability", "peer", "stability", "voltage", "events",
                  "days", "pct_gap", "day_concentration", "hour_entropy", "full_day_share", "trend_slope",
                  "trend_rise", "ewma_alarms", "group_median_gap", "group_size", "voltage_samples",
                  "voltage_median", "voltage_share_outside", "tamper_unresolved", "n_power_failure", "n_comm_loss"]


def score_fleet(inp: Inputs, s: Settings, group_col: str | None = None) -> pd.DataFrame:
    feats = meter_features(inp)
    t = feats.set_index("meter_id")
    t = t.join(scores.availability(feats))
    t = t.join(scores.peer(feats, inp.groups, group_col, s))
    t = t.join(scores.stability(inp, s))
    t = t.join(scores.voltage(inp, s))
    ev = event_summary(inp.events, s)
    t = t.join(ev)
    if "events" not in t.columns:
        t["events"] = float("nan")
    elif inp.events is not None:
        # an events file was given: a meter without events has no penalty (score 1), not "no data"
        t["events"] = t["events"].astype("float64").fillna(1.0)
    t["tamper_unresolved"] = t.get("tamper_unresolved", False)
    t["tamper_unresolved"] = t["tamper_unresolved"].fillna(False).astype(bool)
    for c in ("n_power_failure", "n_comm_loss"):
        t[c] = t[c].fillna(0).astype(int) if c in t.columns else 0
    if "power_failure_days" in t.columns:
        t["power_failure_days"] = t["power_failure_days"].map(lambda x: x if isinstance(x, frozenset) else frozenset())
    t = scores.fuse(t.astype({"events": "float64"}), s)
    t = diagnose(t, inp, s).reset_index()
    return t[[c for c in HEALTH_COLUMNS if c in t.columns]]


def group_summary(health: pd.DataFrame) -> pd.DataFrame:
    h = health.assign(group=health["group"].fillna("UNGROUPED"))
    out = h.groupby("group").agg(meters=("meter_id", "size"), median_health=("health_score", "median"),
                                 median_gap=("pct_gap", "median"))
    counts = h.pivot_table(index="group", columns="diagnosis", values="meter_id", aggfunc="count", fill_value=0)
    return out.join(counts).reset_index()
