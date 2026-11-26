"""ami-meter-health-scoring tests. Every dataset is synthetic."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ami_health import scores
from ami_health.cli import main
from ami_health.config import ConfigError, Settings, load_settings
from ami_health.events import categorise, event_summary
from ami_health.features import _hhi_norm, _slope, meter_features
from ami_health.inputs import Inputs, from_ami_data_quality, load_inputs
from ami_health.pipeline import score_fleet
from ami_health.synthetic import ARCHETYPES, generate

ROOT = Path(__file__).resolve().parents[1]
S = Settings()


# --- helpers ---------------------------------------------------------------------------------

def md(rows):
    """rows: (meter, day index, received) with 96 expected slots."""
    return pd.DataFrame([(m, pd.Timestamp("2026-01-01") + pd.Timedelta(days=d), 96, r) for m, d, r in rows],
                        columns=["meter_id", "local_date", "expected_slots", "received"])


def slots_for(meter, day, slots):
    return pd.DataFrame({"meter_id": meter, "local_date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=day),
                         "slot_index": list(slots)})


@pytest.fixture(scope="module")
def fleet(tmp_path_factory):
    d = tmp_path_factory.mktemp("fleet")
    generate(d)
    inp = load_inputs(d / "meter_days.csv", d / "slots.csv", d / "voltage.csv", d / "groups.csv", d / "events.csv")
    h = score_fleet(inp, load_settings(ROOT / "config" / "health.example.toml"), "dcu_id")
    return h.merge(pd.read_csv(d / "archetypes.csv"), on="meter_id"), d


# --- features ---------------------------------------------------------------------------------

@pytest.mark.parametrize("g, expected", [([5, 5, 5, 5], 0.0), ([20, 0, 0, 0], 1.0), ([0, 0, 0], 0.0), ([3], 0.0)])
def test_normalised_herfindahl(g, expected):
    assert _hhi_norm(np.array(g, dtype=float)) == pytest.approx(expected)


def test_slope_of_a_line_and_of_short_series():
    assert _slope(np.array([0.0, 0.1, 0.2, 0.3])) == pytest.approx(0.1)
    assert _slope(np.array([0.5, 0.1])) == 0.0


def test_hour_entropy_concentrated_vs_spread():
    m = md([("A", d, 92) for d in range(3)] + [("B", d, 72) for d in range(3)])
    sl = pd.concat([slots_for("A", d, [s for s in range(96) if s not in (8, 9, 10, 11)]) for d in range(3)]
                   + [slots_for("B", d, [s for s in range(96) if s % 4]) for d in range(3)])
    f = meter_features(Inputs(m, sl)).set_index("meter_id")
    assert f.loc["A", "hour_entropy"] == pytest.approx(0.0)              # always 02:00-02:45
    assert f.loc["B", "hour_entropy"] == pytest.approx(1.0)              # one slot every hour
    assert f.loc["A", "pct_gap"] == pytest.approx(4 / 96)


def test_full_day_share_and_excluded_days():
    m = md([("A", 0, 0), ("A", 1, 96), ("A", 2, 48)])
    f = meter_features(Inputs(m)).set_index("meter_id")
    assert f.loc["A", "full_day_share"] == pytest.approx(96 / 144)
    assert np.isnan(f.loc["A", "hour_entropy"])                           # no slot table given
    f2 = meter_features(Inputs(m, excluded_days=frozenset([pd.Timestamp("2026-01-03")]))).set_index("meter_id")
    assert f2.loc["A", "days"] == 2


# --- scores -----------------------------------------------------------------------------------

def test_peer_needs_enough_meters_and_uses_a_minimum_scale():
    feats = pd.DataFrame({"meter_id": list("ABCDEF"), "pct_gap": [0, 0, 0, 0, 0, 0.03]})
    groups = pd.DataFrame({"meter_id": list("ABCDEF"), "dcu": ["X"] * 6})
    p = scores.peer(feats, groups, "dcu", S)
    assert p.loc["A", "peer"] == 1.0
    assert p.loc["F", "peer"] == pytest.approx(1 - 3 / 5)                 # z = 0.03 / 0.01
    small = scores.peer(feats.head(3), groups, "dcu", S)
    assert small["peer"].isna().all()
    assert scores.peer(feats, None, None, S)["peer"].isna().all()


def test_stability_detects_a_step_and_needs_enough_days():
    m = md([("A", d, 96) for d in range(10)] + [("B", d, 96 if d < 6 else 30) for d in range(10)]
           + [("C", d, 96) for d in range(3)])
    st = scores.stability(Inputs(m), S)
    assert st.loc["A", "stability"] == 1.0
    assert st.loc["B", "ewma_alarms"] > 0 and st.loc["B", "stability"] < 1
    assert np.isnan(st.loc["C", "stability"])


def test_voltage_score():
    t = pd.date_range("2026-01-01", periods=48, freq="h")
    v = pd.concat([pd.DataFrame({"meter_id": "OK", "ts_local": t, "voltage": 222.0, "quality": "VALID"}),
                   pd.DataFrame({"meter_id": "HIGH", "ts_local": t, "voltage": 246.0, "quality": "VALID"}),
                   pd.DataFrame({"meter_id": "CODE", "ts_local": t, "voltage": 1000.0, "quality": "INVALID"}),
                   pd.DataFrame({"meter_id": "FEW", "ts_local": t[:5], "voltage": 222.0, "quality": "VALID"})])
    vs = scores.voltage(Inputs(md([]), voltage=v), S)
    assert vs.loc["OK", "voltage"] == pytest.approx(1 - (2 / 220 / 0.10) ** 2)
    assert vs.loc["HIGH", "voltage"] == 0.0                               # 11.8 % above nominal, all outside band
    assert "CODE" not in vs.index                                         # invalid samples are ignored
    assert np.isnan(vs.loc["FEW", "voltage"])
    assert scores.voltage(Inputs(md([])), S).empty


def test_fusion_renormalises_and_reports_coverage():
    t = pd.DataFrame({"availability": [0.8, 0.8], "peer": [np.nan, 1.0], "stability": [1.0, 1.0],
                      "voltage": [np.nan, np.nan], "events": [np.nan, np.nan]}, index=["A", "B"])
    f = scores.fuse(t, S)
    assert f.loc["A", "health_score"] == pytest.approx((0.35 * 0.8 + 0.15 * 1) / 0.5)
    assert f.loc["A", "coverage_of_score"] == pytest.approx(0.5)
    assert f.loc["A", "modules_used"] == "availability;stability"
    assert f.loc["B", "coverage_of_score"] == pytest.approx(0.7)


# --- events -----------------------------------------------------------------------------------

def test_event_categories_and_tamper_order():
    ev = pd.DataFrame({"meter_id": ["A", "A", "B", "B", "C"],
                       "ts_local": pd.to_datetime(["2026-01-02 10:20", "2026-01-02 10:00", "2026-01-02 10:00",
                                                   "2026-01-03 09:00", "2026-01-02 08:00"]),
                       "description": ["Terminal cover closed", "Terminal cover opened", "Terminal cover closed",
                                       "Terminal cover opened", "Long power failure"]})
    s = event_summary(ev, S)
    assert not s.loc["A", "tamper_unresolved"]           # opened then closed (input order does not matter)
    assert s.loc["B", "tamper_unresolved"]               # closed then opened again -> open
    assert s.loc["B", "events"] == 0.0
    assert s.loc["C", "n_power_failure"] == 1 and s.loc["C", "events"] == 1.0
    assert s.loc["C", "power_failure_days"] == frozenset([pd.Timestamp("2026-01-02")])
    assert event_summary(None, S).empty


def test_precategorised_events_and_unknown_text():
    ev = pd.DataFrame({"meter_id": ["A", "A"], "ts_local": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                       "category": ["COMM_LOSS", "SOMETHING_ELSE"]})
    c = categorise(ev, S)
    assert c["category"].tolist() == ["COMM_LOSS", "OTHER"]
    assert event_summary(ev, S).loc["A", "events"] == pytest.approx(1 - 0.02)


# --- diagnosis on the synthetic archetypes -----------------------------------------------------

@pytest.mark.parametrize("archetype", sorted(ARCHETYPES))
def test_each_archetype_gets_its_diagnosis(fleet, archetype):
    h, _ = fleet
    rows = h[h.archetype == archetype]
    assert (rows.diagnosis == rows.expected_diagnosis).all(), rows[["meter_id", "diagnosis", "diagnosis_reason"]]


def test_priority_order_and_field_list(fleet):
    h, _ = fleet
    h = h.sort_values("priority")
    assert h.iloc[0].diagnosis == "TAMPER_OPEN"
    assert h["priority"].tolist() == list(range(1, len(h) + 1))
    assert (h[h.diagnosis == "OK"].priority > h[h.diagnosis != "OK"].priority.max()).all()


def test_scores_are_in_range_and_use_every_module(fleet):
    h, _ = fleet
    for c in ("health_score", "availability", "peer", "stability", "voltage", "events"):
        assert h[c].between(0, 1).all(), c
    assert (h.coverage_of_score == 1.0).all()


def test_power_outage_needs_events_covering_the_gap_days():
    m = md([("A", d, 96 if d not in (2, 3) else 40) for d in range(6)])
    ev = pd.DataFrame({"meter_id": ["A"], "ts_local": pd.to_datetime(["2026-01-03 10:00"]),
                       "description": ["Long power failure"]})
    h = score_fleet(Inputs(m, events=ev), S).set_index("meter_id")
    assert h.loc["A", "diagnosis"] == "POWER_OUTAGE"                      # 1 of 2 gap days = 0.5 cover
    strict = replace(S, diagnosis=replace(S.diagnosis, power_outage_cover=0.9))
    assert score_fleet(Inputs(m, events=ev), strict).set_index("meter_id").loc["A", "diagnosis"] != "POWER_OUTAGE"


def test_without_optional_inputs_the_score_uses_fewer_modules():
    h = score_fleet(Inputs(md([("A", d, 96) for d in range(6)])), S).iloc[0]
    assert h.modules_used == "availability;stability"
    assert h.coverage_of_score == pytest.approx(0.5) and h.diagnosis == "OK"


# --- configuration and inputs -------------------------------------------------------------------

def test_example_settings_load():
    s = load_settings(ROOT / "config" / "health.example.toml")
    assert s.weights.availability == 0.35 and "TAMPER_OPEN" in s.event_patterns


@pytest.mark.parametrize("text", [
    "[weights]\nfoo = 1\n", "[nonsense]\na = 1\n", "[weights]\navailability = -1\n",
    "[weights]\navailability = 0\npeer = 0\nstability = 0\nvoltage = 0\nevents = 0\n",
    "[scores]\newma_lambda = 0\n", "[event_patterns]\nTAMPER_OPEN = \"(\"\n", "[event_patterns]\nNEW = \"x\"\n",
], ids=["unknown-key", "unknown-section", "negative", "zero-weights", "lambda", "bad-regex", "unknown-category"])
def test_invalid_settings(tmp_path, text):
    p = tmp_path / "h.toml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(p)


def test_input_validation(tmp_path):
    bad = tmp_path / "md.csv"
    bad.write_text("meter_id,local_date\nA,2026-01-01\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_inputs(bad)
    good = tmp_path / "ok.csv"
    good.write_text("meter_id,local_date,expected_slots,received\nA,2026-01-01,96,96\n", encoding="utf-8")
    ev = tmp_path / "ev.csv"
    ev.write_text("meter_id,ts_local\nA,2026-01-01 10:00\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_inputs(good, events=ev)


def test_adapter_for_ami_data_quality_outputs(tmp_path):
    d = tmp_path / "dq"
    d.mkdir()
    (d / "meter_days_interval.csv").write_text(
        "meter_id,local_date,expected_slots,received,non_null,valid\nA,2026-01-01,96,95,95,95\n"
        "A,2026-01-02,96,1,1,1\n", encoding="utf-8")
    rows = ["meter_id,local_date,slot_index,row_status,other"]
    rows += [f"A,2026-01-01,{s},ACCEPTED,x" for s in range(95)] + ["A,2026-01-01,,REJECTED,x", "A,2026-01-02,0,ACCEPTED,x"]
    (d / "readings_interval.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (d / "readings_instrumentation.csv").write_text(
        "meter_id,ts_local,voltage_l1,voltage_l1_q,row_status\nA,2026-01-01 00:00,225,VALID,ACCEPTED\n"
        "A,2026-01-01 01:00,1000,INVALID,ACCEPTED\n", encoding="utf-8")
    (d / "day_class_interval.csv").write_text(
        "local_date,day_class\n2026-01-01,NORMAL\n2026-01-02,BORDER_CONFIRMED\n", encoding="utf-8")
    inp = from_ami_data_quality(d)
    assert len(inp.slots) == 96 and inp.excluded_days == frozenset([pd.Timestamp("2026-01-02")])
    assert inp.voltage["quality"].tolist() == ["VALID", "INVALID"]
    h = score_fleet(inp, S).iloc[0]
    assert h.days == 1 and h.pct_gap == pytest.approx(1 / 96)            # border day excluded


# --- command line -------------------------------------------------------------------------------

def test_cli_score_outputs_and_determinism(fleet, tmp_path):
    _, d = fleet
    args = ["score", "--meter-days", str(d / "meter_days.csv"), "--slots", str(d / "slots.csv"),
            "--voltage", str(d / "voltage.csv"), "--events", str(d / "events.csv"),
            "--groups", str(d / "groups.csv"), "--group-col", "dcu_id"]
    assert main([*args, "--out", str(tmp_path / "a")]) == 0
    assert main([*args, "--out", str(tmp_path / "b")]) == 0
    for name in ("health.csv", "field_list.csv", "group_summary.csv"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    m = json.loads((tmp_path / "a" / "run_manifest.json").read_text(encoding="utf-8"))
    assert m["meters"] == 26 and len(m["inputs"]["slots"]["sha256"]) == 64
    fl = pd.read_csv(tmp_path / "a" / "field_list.csv")
    assert "OK" not in set(fl.diagnosis) and fl.iloc[0].diagnosis == "TAMPER_OPEN"


def test_cli_from_ami_dq_and_errors(tmp_path, capsys):
    assert main(["score", "--meter-days", str(tmp_path / "missing.csv"), "--out", str(tmp_path / "o")]) == 2
    assert main(["from-ami-dq", "--dir", str(tmp_path), "--out", str(tmp_path / "o")]) == 2
    d = tmp_path / "syn"
    assert main(["synthetic", "--out", str(d), "--days", "8"]) == 2
    assert main(["synthetic", "--out", str(d), "--days", "12"]) == 0
    assert main(["score", "--meter-days", str(d / "meter_days.csv"), "--groups", str(d / "groups.csv"),
                 "--group-col", "nope", "--out", str(tmp_path / "o")]) == 2
    assert "error:" in capsys.readouterr().err


def test_cli_from_ami_dq_success(tmp_path):
    d = tmp_path / "dq"
    d.mkdir()
    (d / "meter_days_interval.csv").write_text("meter_id,local_date,expected_slots,received\nA,2026-01-01,96,96\n",
                                               encoding="utf-8")
    (d / "readings_interval.csv").write_text(
        "meter_id,local_date,slot_index,row_status\n" + "".join(f"A,2026-01-01,{s},ACCEPTED\n" for s in range(96)),
        encoding="utf-8")
    assert main(["from-ami-dq", "--dir", str(d), "--out", str(tmp_path / "o")]) == 0
    assert pd.read_csv(tmp_path / "o" / "health.csv").iloc[0].diagnosis == "OK"


def test_committed_example_is_reproducible(tmp_path):
    generate(tmp_path)
    ex = ROOT / "examples" / "synthetic"
    for name in ("meter_days.csv", "slots.csv", "voltage.csv", "events.csv", "groups.csv", "archetypes.csv"):
        assert (tmp_path / name).read_bytes() == (ex / name).read_bytes(), name


def test_python_dash_m(tmp_path):
    import os
    import subprocess
    import sys

    r = subprocess.run([sys.executable, "-m", "ami_health", "synthetic", "--out", str(tmp_path), "--days", "12"],
                       capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert r.returncode == 0, r.stderr
