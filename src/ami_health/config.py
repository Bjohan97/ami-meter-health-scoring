"""Settings (``health.toml``). Every threshold of the scores and of the diagnosis is here.

Unknown sections or keys are rejected so that a typo never falls back silently to a default.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path


class ConfigError(ValueError):
    def __init__(self, message: str):
        super().__init__(f"CONFIG_INVALID: {message}")


@dataclass(frozen=True)
class Weights:
    availability: float = 0.35
    peer: float = 0.20
    stability: float = 0.15
    voltage: float = 0.15
    events: float = 0.15


@dataclass(frozen=True)
class Scores:
    peer_min_meters: int = 5          # peers needed in a group for the peer score
    peer_z_max: float = 5.0           # robust z at which the peer score reaches 0
    peer_scale_min: float = 0.01      # minimum robust scale (1 percentage point of gaps)
    ewma_lambda: float = 0.3
    ewma_l: float = 3.0
    ewma_sigma_min: float = 0.02
    stability_min_days: int = 5
    nominal_voltage_v: float = 220.0
    voltage_band: float = 0.10        # quality band +/-10 %
    voltage_min_samples: int = 24
    voltage_share_zero_score: float = 0.20   # share of samples outside the band giving score 0
    voltage_dev_zero_score: float = 0.10     # median deviation giving score 0 (quadratic)


@dataclass(frozen=True)
class EventWeights:
    tamper_open: float = 1.0
    comm_loss: float = 0.02
    voltage_event: float = 0.05
    clock_sync: float = 0.01
    power_failure: float = 0.0        # explains gaps; does not penalise the meter itself


@dataclass(frozen=True)
class Diagnosis:
    group_issue_median_gap: float = 0.30
    meter_gap_min: float = 0.05
    no_data_gap: float = 0.95
    day_level_share: float = 0.80
    hardware_entropy_max: float = 0.60
    power_outage_cover: float = 0.50
    voltage_score_max: float = 0.50
    trend_slope_min: float = 0.02     # gap fraction per day
    trend_rise_min: float = 0.10


@dataclass(frozen=True)
class Settings:
    weights: Weights = field(default_factory=Weights)
    scores: Scores = field(default_factory=Scores)
    event_weights: EventWeights = field(default_factory=EventWeights)
    diagnosis: Diagnosis = field(default_factory=Diagnosis)
    event_patterns: dict = field(default_factory=lambda: {
        "TAMPER_OPEN": r"(?i)cover.*open",
        "TAMPER_CLOSE": r"(?i)cover.*clos",
        "POWER_FAILURE": r"(?i)power (failure|down|outage)",
        "COMM_LOSS": r"(?i)(communication|plc).*(lost|end|fail)",
        "VOLTAGE_EVENT": r"(?i)(sag|swell|under ?voltage|over ?voltage)",
        "CLOCK_SYNC": r"(?i)clock|time sync",
    })


_SECTIONS = {"weights": Weights, "scores": Scores, "event_weights": EventWeights, "diagnosis": Diagnosis}


def _build(cls, raw: dict, name: str):
    allowed = {f.name for f in fields(cls)}
    bad = sorted(set(raw) - allowed)
    if bad:
        raise ConfigError(f"[{name}] unknown keys {bad}")
    obj = cls(**raw)
    for f in fields(cls):
        v = getattr(obj, f.name)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0:
            raise ConfigError(f"{name}.{f.name} must be a non-negative number, got {v!r}")
    return obj


def validate(s: Settings) -> Settings:
    w = s.weights
    if sum(getattr(w, f.name) for f in fields(w)) <= 0:
        raise ConfigError("weights must not all be zero")
    if not 0 < s.scores.ewma_lambda <= 1:
        raise ConfigError("scores.ewma_lambda must be in (0, 1]")
    import re

    required = {"TAMPER_OPEN", "TAMPER_CLOSE", "POWER_FAILURE", "COMM_LOSS", "VOLTAGE_EVENT", "CLOCK_SYNC"}
    unknown = set(s.event_patterns) - required
    if unknown:
        raise ConfigError(f"[event_patterns] unknown categories {sorted(unknown)}")
    for k, p in s.event_patterns.items():
        try:
            re.compile(p)
        except re.error as exc:
            raise ConfigError(f"event_patterns.{k}: invalid regex ({exc})") from None
    return s


def load_settings(path: str | Path | None = None) -> Settings:
    if path is None:
        return validate(Settings())
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    bad = sorted(set(raw) - {*_SECTIONS, "event_patterns"})
    if bad:
        raise ConfigError(f"unknown sections {bad}")
    kwargs = {k: _build(cls, raw.get(k, {}), k) for k, cls in _SECTIONS.items()}
    patterns = {**Settings().event_patterns, **raw.get("event_patterns", {})}
    return validate(Settings(**kwargs, event_patterns=patterns))
