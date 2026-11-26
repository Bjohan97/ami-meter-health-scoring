"""The five health scores, each in [0, 1] (1 = healthy) or missing when it cannot be computed.

availability  1 - pct_gap
peer          robust z of pct_gap against the meter's group: z = (x - median) / max(1.4826 MAD, scale_min);
              score = 1 - clip(z / z_max, 0, 1). Needs at least ``peer_min_meters`` meters in the group.
stability     EWMA control chart of the daily gap fraction (lambda, L): centre = median, sigma = max(1.4826 MAD,
              sigma_min), UCL = centre + L sigma sqrt(lambda / (2 - lambda)); score = 1 - min(alarms / (0.2 n), 1).
voltage       usable voltage samples only: share outside the +/-band and median deviation d;
              score = 1 - max(min(share / share_zero, 1), min((d / dev_zero)^2, 1))
events        1 - min(sum(weight_c * count_c), 1); an unresolved tamper gives 0 (see events.py)

The formulas are transparent heuristics for ranking meters; they are not calibrated against field
inspection outcomes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Settings
from .features import daily_gaps
from .inputs import USABLE, Inputs


def availability(features: pd.DataFrame) -> pd.Series:
    return (1 - features.set_index("meter_id")["pct_gap"]).clip(0, 1).rename("availability")


def peer(features: pd.DataFrame, groups: pd.DataFrame | None, group_col: str | None, s: Settings) -> pd.DataFrame:
    """Peer score plus the group median gap (used by the GROUP_ISSUE diagnosis)."""
    f = features[["meter_id", "pct_gap"]].copy()
    if groups is None or group_col is None or group_col not in groups.columns:
        f["group"], f["peer"], f["group_median_gap"], f["group_size"] = pd.NA, np.nan, np.nan, 0
        return f.set_index("meter_id")[["group", "peer", "group_median_gap", "group_size"]]
    f["group"] = f["meter_id"].map(groups.set_index("meter_id")[group_col])
    g = f.groupby("group")["pct_gap"]
    f["group_size"] = g.transform("size")
    med = g.transform("median")
    mad = (f["pct_gap"] - med).abs().groupby(f["group"]).transform("median")
    scale = np.maximum(1.4826 * mad, s.scores.peer_scale_min)
    z = (f["pct_gap"] - med) / scale
    score = 1 - (z / s.scores.peer_z_max).clip(0, 1)
    f["peer"] = score.where(f["group_size"] >= s.scores.peer_min_meters)
    f["group_median_gap"] = med
    return f.set_index("meter_id")[["group", "peer", "group_median_gap", "group_size"]]


def stability(inp: Inputs, s: Settings) -> pd.DataFrame:
    md = daily_gaps(inp)
    lam, L = s.scores.ewma_lambda, s.scores.ewma_l
    rows = []
    for m, g in md.groupby("meter_id", sort=True):
        f = g["gap_frac"].to_numpy(dtype=float)
        if len(f) < s.scores.stability_min_days:
            rows.append((m, np.nan, 0))
            continue
        centre = float(np.median(f))
        sigma = max(1.4826 * float(np.median(np.abs(f - centre))), s.scores.ewma_sigma_min)
        ucl = centre + L * sigma * np.sqrt(lam / (2 - lam))
        z, alarms = centre, 0
        for x in f:
            z = lam * x + (1 - lam) * z
            alarms += int(z > ucl)
        rows.append((m, 1 - min(alarms / (0.2 * len(f)), 1.0), alarms))
    return pd.DataFrame(rows, columns=["meter_id", "stability", "ewma_alarms"]).set_index("meter_id")


def voltage(inp: Inputs, s: Settings) -> pd.DataFrame:
    cols = ["voltage", "voltage_samples", "voltage_median", "voltage_share_outside"]
    if inp.voltage is None or inp.voltage.empty:
        return pd.DataFrame(columns=["meter_id", *cols]).set_index("meter_id")
    v = inp.voltage
    if "quality" in v.columns:
        v = v[v["quality"].isin(USABLE)]
    v = v.dropna(subset=["voltage"])
    v = v[v["voltage"] > 0]                      # 0 V = outage, not a voltage-quality sample
    un, band = s.scores.nominal_voltage_v, s.scores.voltage_band
    rows = []
    for m, g in v.groupby("meter_id", sort=True):
        x = g["voltage"].to_numpy(dtype=float)
        med = float(np.median(x))
        share = float((np.abs(x - un) > band * un).mean())
        if len(x) < s.scores.voltage_min_samples:
            rows.append((m, np.nan, len(x), med, share))
            continue
        d = abs(med - un) / un
        pen = max(min(share / s.scores.voltage_share_zero_score, 1.0), min((d / s.scores.voltage_dev_zero_score) ** 2, 1.0))
        rows.append((m, 1 - pen, len(x), med, share))
    return pd.DataFrame(rows, columns=["meter_id", *cols]).set_index("meter_id")


def fuse(table: pd.DataFrame, s: Settings) -> pd.DataFrame:
    """Weighted mean over the available scores (weights renormalised), and the share of the total
    weight that was available (``coverage_of_score``; not a probability)."""
    w = {"availability": s.weights.availability, "peer": s.weights.peer, "stability": s.weights.stability,
         "voltage": s.weights.voltage, "events": s.weights.events}
    total = sum(w.values())
    num = pd.Series(0.0, index=table.index)
    den = pd.Series(0.0, index=table.index)
    for k, wk in w.items():
        if k not in table.columns or wk == 0:
            continue
        ok = table[k].notna()
        num[ok] += wk * table.loc[ok, k]
        den[ok] += wk
    out = table.copy()
    out["health_score"] = (num / den).where(den > 0)
    out["coverage_of_score"] = den / total
    out["modules_used"] = [";".join(k for k in w if k in table.columns and pd.notna(table.at[i, k]) and w[k] > 0)
                           for i in table.index]
    return out
