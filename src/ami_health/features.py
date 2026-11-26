"""Per-meter features of the communication gaps.

Definitions (n = analysed days, g_d = missing slots on day d, E_d = expected slots):

pct_gap            sum(g_d) / sum(E_d)
day_concentration  normalised Herfindahl index of the daily gaps:
                   H = sum((g_d / G)^2), H* = (H - 1/n) / (1 - 1/n)  -> 0 even spread, 1 one single day
hour_entropy       Shannon entropy of the missing slots by hour of day, normalised by log2(24):
                   1 = gaps spread over every hour, 0 = always the same hour
full_day_share     share of missing slots that belong to days with no data at all
trend_slope        least-squares slope of the daily gap fraction (per day). It describes the past
                   trend; it is not a forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .inputs import Inputs

FEATURE_COLUMNS = ["meter_id", "days", "expected", "missing", "pct_gap", "day_concentration", "hour_entropy",
                   "full_day_share", "trend_slope", "trend_rise"]


def daily_gaps(inp: Inputs) -> pd.DataFrame:
    md = inp.meter_days.copy()
    if inp.excluded_days:
        md = md[~md["local_date"].isin(inp.excluded_days)]
    md["missing"] = (md["expected_slots"] - md["received"]).clip(lower=0)
    md["gap_frac"] = md["missing"] / md["expected_slots"]
    return md.sort_values(["meter_id", "local_date"]).reset_index(drop=True)


def _hhi_norm(g: np.ndarray) -> float:
    n, total = len(g), g.sum()
    if total == 0 or n < 2:
        return 0.0
    h = float(((g / total) ** 2).sum())
    return (h - 1 / n) / (1 - 1 / n)


def _slope(y: np.ndarray) -> float:
    if len(y) < 3:
        return 0.0
    x = np.arange(len(y), dtype=float)
    x -= x.mean()
    den = float((x ** 2).sum())
    return float((x * (y - y.mean())).sum() / den) if den else 0.0


def hourly_missing(inp: Inputs, md: pd.DataFrame) -> pd.DataFrame:
    """Missing slots per (meter, hour of day), from the received slots. Hour = slot * 24 // E(d)."""
    if inp.slots is None:
        return pd.DataFrame(columns=["meter_id", "hour", "missing"])
    rec = inp.slots.merge(md[["meter_id", "local_date", "expected_slots"]], on=["meter_id", "local_date"])
    rec_hour = (rec["slot_index"] * 24 // rec["expected_slots"]).astype(int)
    received = rec.assign(hour=rec_hour).groupby(["meter_id", "hour"]).size().rename("received")
    # expected slots per hour, summed over the analysed days of each meter
    md = md.assign(per_hour=md["expected_slots"] / 24)
    exp = md.groupby("meter_id")["per_hour"].sum()
    idx = pd.MultiIndex.from_product([sorted(md["meter_id"].unique()), range(24)], names=["meter_id", "hour"])
    out = pd.DataFrame(index=idx).join(received).fillna(0).reset_index()
    out["expected"] = out["meter_id"].map(exp)
    out["missing"] = (out["expected"] - out["received"]).clip(lower=0)
    return out[["meter_id", "hour", "missing"]]


def meter_features(inp: Inputs) -> pd.DataFrame:
    md = daily_gaps(inp)
    hours = hourly_missing(inp, md)
    ent = {}
    for m, h in hours.groupby("meter_id"):
        p = h["missing"].to_numpy(dtype=float)
        tot = p.sum()
        if tot <= 0:
            ent[m] = np.nan
            continue
        p = p[p > 0] / tot
        ent[m] = float(-(p * np.log2(p)).sum() / np.log2(24))
    rows = []
    for m, g in md.groupby("meter_id", sort=True):
        miss = g["missing"].to_numpy(dtype=float)
        frac = g["gap_frac"].to_numpy(dtype=float)
        total_missing = miss.sum()
        full = g.loc[g["received"] == 0, "missing"].sum()
        k = max(1, min(3, len(frac) // 3))
        rows.append({
            "meter_id": m, "days": len(g), "expected": int(g["expected_slots"].sum()), "missing": int(total_missing),
            "pct_gap": total_missing / g["expected_slots"].sum(),
            "day_concentration": _hhi_norm(miss),
            "hour_entropy": ent.get(m, np.nan) if total_missing > 0 else np.nan,
            "full_day_share": full / total_missing if total_missing > 0 else 0.0,
            "trend_slope": _slope(frac),
            "trend_rise": float(frac[-k:].mean() - frac[:k].mean()) if len(frac) else 0.0,
        })
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)
