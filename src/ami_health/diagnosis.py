"""Rule-based diagnosis and field-inspection priority.

Rules are evaluated in this order; the first that applies is the diagnosis (all thresholds in
``[diagnosis]``):

TAMPER_OPEN                 last tamper event is "open"                       -> inspect the meter
GROUP_ISSUE                 the group's median gap is high and the meter has gaps -> check the DCU / network
POWER_OUTAGE                power-failure events cover most of the gap days   -> supply problem, not the meter
NO_DATA                     almost no data in the period                      -> inspect
DAY_LEVEL_LOSS              gaps are mostly whole days without data           -> meter unreachable on some days
DETERIORATING               rising trend of the daily gap fraction            -> watch
HARDWARE_SUSPECT            gaps concentrated in a few hours of the day       -> meter-side behaviour
INTERMITTENT_COMMUNICATION  gaps spread over the day                          -> communication quality
VOLTAGE_QUALITY             low voltage score                                 -> supply quality
OK
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Settings
from .features import daily_gaps
from .inputs import Inputs

ORDER = ["TAMPER_OPEN", "NO_DATA", "HARDWARE_SUSPECT", "DAY_LEVEL_LOSS", "INTERMITTENT_COMMUNICATION",
         "VOLTAGE_QUALITY", "DETERIORATING", "GROUP_ISSUE", "POWER_OUTAGE", "OK"]   # field priority (1 = first)
ACTION = {
    "TAMPER_OPEN": "field inspection: terminal cover / tamper",
    "GROUP_ISSUE": "check the data concentrator or network of the group, not the individual meter",
    "POWER_OUTAGE": "gaps explained by supply interruptions; no meter visit needed",
    "NO_DATA": "field inspection: meter not communicating",
    "DAY_LEVEL_LOSS": "check reachability (whole days missing)",
    "HARDWARE_SUSPECT": "field inspection: gaps at the same hours (clock, battery, local interference)",
    "INTERMITTENT_COMMUNICATION": "review communication path / signal quality",
    "VOLTAGE_QUALITY": "review supply voltage at the service point",
    "DETERIORATING": "monitor: communication worsening",
    "OK": "",
}


def diagnose(table: pd.DataFrame, inp: Inputs, s: Settings) -> pd.DataFrame:
    d = s.diagnosis
    gap_days = daily_gaps(inp)
    gap_days = gap_days[gap_days["missing"] > 0].groupby("meter_id")["local_date"].agg(frozenset)
    out = table.copy()
    diag, why = [], []
    for m, r in out.iterrows():
        gap = r.get("pct_gap", 0.0) or 0.0
        if bool(r.get("tamper_unresolved", False)):
            diag.append("TAMPER_OPEN"); why.append("last tamper event is 'open'")
            continue
        gmed = r.get("group_median_gap", np.nan)
        if pd.notna(gmed) and gmed >= d.group_issue_median_gap and gap >= d.meter_gap_min:
            diag.append("GROUP_ISSUE"); why.append(f"group median gap {gmed:.2f}")
            continue
        pf = r.get("power_failure_days", frozenset())
        gdays = gap_days.get(m, frozenset())
        if isinstance(pf, frozenset) and gdays and gap >= d.meter_gap_min and \
                len(gdays & pf) / len(gdays) >= d.power_outage_cover:
            diag.append("POWER_OUTAGE"); why.append(f"{len(gdays & pf)}/{len(gdays)} gap days with power failure")
            continue
        if gap >= d.no_data_gap:
            diag.append("NO_DATA"); why.append(f"gap {gap:.2f}")
            continue
        if gap >= d.meter_gap_min and r.get("full_day_share", 0) >= d.day_level_share:
            diag.append("DAY_LEVEL_LOSS"); why.append(f"{r['full_day_share']:.2f} of the gaps are whole days")
            continue
        # a clearly rising trend is checked before the static gap patterns: a worsening meter has gaps
        if r.get("trend_slope", 0) >= d.trend_slope_min and r.get("trend_rise", 0) >= d.trend_rise_min:
            diag.append("DETERIORATING"); why.append(f"slope {r['trend_slope']:.3f}/day, rise {r['trend_rise']:.2f}")
            continue
        ent = r.get("hour_entropy", np.nan)
        if gap >= d.meter_gap_min and pd.notna(ent) and ent <= d.hardware_entropy_max:
            diag.append("HARDWARE_SUSPECT"); why.append(f"hour entropy {ent:.2f}")
            continue
        if gap >= d.meter_gap_min:
            diag.append("INTERMITTENT_COMMUNICATION"); why.append(f"gap {gap:.2f}, hour entropy {ent:.2f}")
            continue
        vs = r.get("voltage", np.nan)
        if pd.notna(vs) and vs <= d.voltage_score_max:
            diag.append("VOLTAGE_QUALITY"); why.append(f"voltage score {vs:.2f}")
            continue
        diag.append("OK"); why.append("")
    out["diagnosis"], out["diagnosis_reason"] = diag, why
    out["action"] = out["diagnosis"].map(ACTION)
    rank = {k: i for i, k in enumerate(ORDER)}
    out["_rank"] = out["diagnosis"].map(rank)
    out = out.sort_values(["_rank", "health_score", "meter_id"], na_position="first", kind="stable")
    out["priority"] = range(1, len(out) + 1)
    return out.drop(columns="_rank")
