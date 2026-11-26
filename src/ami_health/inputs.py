"""Input tables (canonical, minimal) and an adapter for ami-data-quality outputs.

meter_days  meter_id, local_date, expected_slots, received
slots       meter_id, local_date, slot_index            (received slots; optional)
voltage     meter_id, ts_local, voltage, quality         (optional; only VALID / VALID_WITH_WARNING used)
groups      meter_id, <group column>                     (optional)
events      meter_id, ts_local, description | category   (optional)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import ConfigError

USABLE = ("VALID", "VALID_WITH_WARNING")


@dataclass
class Inputs:
    meter_days: pd.DataFrame
    slots: pd.DataFrame | None = None
    voltage: pd.DataFrame | None = None
    groups: pd.DataFrame | None = None
    events: pd.DataFrame | None = None
    excluded_days: frozenset = frozenset()


def _require(df: pd.DataFrame, cols, name: str) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ConfigError(f"{name}: missing columns {missing}")
    return df


def _read(path, name, cols, dates=("local_date",), times=()) -> pd.DataFrame:
    df = _require(pd.read_csv(path, dtype={"meter_id": str}), cols, name)
    for c in dates:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], format="%Y-%m-%d")
    for c in times:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], format="mixed")
    return df


def load_inputs(meter_days, slots=None, voltage=None, groups=None, events=None, excluded_days=()) -> Inputs:
    md = _read(meter_days, "meter_days", ["meter_id", "local_date", "expected_slots", "received"])
    sl = _read(slots, "slots", ["meter_id", "local_date", "slot_index"]) if slots else None
    vo = _read(voltage, "voltage", ["meter_id", "ts_local", "voltage"], dates=(), times=("ts_local",)) if voltage else None
    gr = _require(pd.read_csv(groups, dtype=str), ["meter_id"], "groups") if groups else None
    ev = _read(events, "events", ["meter_id", "ts_local"], dates=(), times=("ts_local",)) if events else None
    if ev is not None and not ({"description", "category"} & set(ev.columns)):
        raise ConfigError("events: needs a 'description' or a 'category' column")
    return Inputs(md, sl, vo, gr, ev, frozenset(pd.to_datetime(list(excluded_days))))


def from_ami_data_quality(out_dir: str | Path, groups=None, events=None, profile: str = "interval",
                          voltage_column: str = "voltage_l1") -> Inputs:
    """Build the inputs from an ``ami-data-quality`` output folder.

    Uses meter_days_<profile>.csv, readings_<profile>.csv (ACCEPTED slots) and, when present,
    readings_instrumentation.csv (voltage) and day_class_<profile>.csv (BORDER_CONFIRMED days
    are excluded).
    """
    d = Path(out_dir)
    md = _read(d / f"meter_days_{profile}.csv", "meter_days", ["meter_id", "local_date", "expected_slots", "received"])
    rd = pd.read_csv(d / f"readings_{profile}.csv", dtype={"meter_id": str},
                     usecols=["meter_id", "local_date", "slot_index", "row_status"])
    rd = rd[(rd["row_status"] == "ACCEPTED") & rd["slot_index"].notna()]
    slots = pd.DataFrame({"meter_id": rd["meter_id"], "local_date": pd.to_datetime(rd["local_date"]),
                          "slot_index": rd["slot_index"].astype(int)})
    voltage = None
    inst = d / "readings_instrumentation.csv"
    if inst.exists():
        cols = ["meter_id", "ts_local", voltage_column, f"{voltage_column}_q", "row_status"]
        vi = pd.read_csv(inst, dtype={"meter_id": str}, usecols=lambda c: c in cols)
        if voltage_column in vi.columns:
            vi = vi[vi["row_status"] == "ACCEPTED"]
            voltage = pd.DataFrame({"meter_id": vi["meter_id"], "ts_local": pd.to_datetime(vi["ts_local"]),
                                    "voltage": vi[voltage_column], "quality": vi[f"{voltage_column}_q"]})
    excluded = frozenset()
    dc = d / f"day_class_{profile}.csv"
    if dc.exists():
        t = pd.read_csv(dc)
        excluded = frozenset(pd.to_datetime(t.loc[t["day_class"] == "BORDER_CONFIRMED", "local_date"]))
    gr = pd.read_csv(groups, dtype=str) if groups else None
    ev = _read(events, "events", ["meter_id", "ts_local"], dates=(), times=("ts_local",)) if events else None
    return Inputs(md, slots, voltage, gr, ev, excluded)
