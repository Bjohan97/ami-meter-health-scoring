"""Meter events: categorisation, tamper state in time order, and the events score.

Event texts and codes are manufacturer-specific, so categories come from configurable regular
expressions (``[event_patterns]``) unless the input already has a ``category`` column.
The tamper state of a meter is decided by its *last* tamper event in time: open -> unresolved.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import Settings

CATEGORIES = ("TAMPER_OPEN", "TAMPER_CLOSE", "POWER_FAILURE", "COMM_LOSS", "VOLTAGE_EVENT", "CLOCK_SYNC")


def categorise(events: pd.DataFrame, s: Settings) -> pd.DataFrame:
    ev = events.copy()
    if "category" not in ev.columns:
        text = ev["description"].astype(str)
        cat = pd.Series(pd.NA, index=ev.index, dtype=object)
        for name in CATEGORIES:            # first matching category wins, in this order
            rx = re.compile(s.event_patterns[name])
            hit = cat.isna() & text.map(lambda t: rx.search(t) is not None)
            cat[hit] = name
        ev["category"] = cat
    ev["category"] = ev["category"].where(ev["category"].isin(CATEGORIES), "OTHER")
    return ev.sort_values(["meter_id", "ts_local"], kind="stable").reset_index(drop=True)


def event_summary(events: pd.DataFrame | None, s: Settings) -> pd.DataFrame:
    cols = ["events", "tamper_unresolved", "power_failure_days", *[f"n_{c.lower()}" for c in CATEGORIES]]
    if events is None or events.empty:
        return pd.DataFrame(columns=["meter_id", *cols]).set_index("meter_id")
    ev = categorise(events, s)
    w = s.event_weights
    rows = []
    for m, g in ev.groupby("meter_id", sort=True):
        counts = g["category"].value_counts()
        tamper = g[g["category"].isin(["TAMPER_OPEN", "TAMPER_CLOSE"])]
        unresolved = bool(len(tamper)) and tamper["category"].iloc[-1] == "TAMPER_OPEN"
        pen = (w.comm_loss * counts.get("COMM_LOSS", 0) + w.voltage_event * counts.get("VOLTAGE_EVENT", 0)
               + w.clock_sync * counts.get("CLOCK_SYNC", 0) + w.power_failure * counts.get("POWER_FAILURE", 0))
        score = 0.0 if (unresolved and w.tamper_open > 0) else 1 - min(pen, 1.0)
        pf_days = frozenset(g.loc[g["category"] == "POWER_FAILURE", "ts_local"].dt.normalize())
        rows.append({"meter_id": m, "events": score, "tamper_unresolved": unresolved, "power_failure_days": pf_days,
                     **{f"n_{c.lower()}": int(counts.get(c, 0)) for c in CATEGORIES}})
    return pd.DataFrame(rows).set_index("meter_id")[cols]
