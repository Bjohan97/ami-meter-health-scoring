# ami-meter-health-scoring

Health scores and **field-inspection priorities** for AMI smart meters. The inputs are communication
gaps, voltage quality and meter events.

When thousands of meters stop reporting now and then, an operator needs to know **which ones need a visit
and why**: a meter fault, a data concentrator (DCU) or network problem, a supply interruption, tampering,
or poor supply voltage. This tool scores every meter with transparent indicators and ranks the field list.

> Status: first version (0.0.1). 43 automated tests on synthetic data, line and branch coverage 93–100 %
> per module. Companion of [ami-data-quality](https://github.com/Bjohan97/ami-data-quality), whose outputs
> it can read directly.

## How it works

**1. Features of the gaps (per meter)**

| Feature | Definition | What it tells |
|---|---|---|
| `pct_gap` | missing slots / expected slots | how much data is missing |
| `day_concentration` | normalised Herfindahl index of the daily gaps, `H* = (H − 1/n)/(1 − 1/n)` | 0 = gaps spread over the days, 1 = a single day |
| `hour_entropy` | Shannon entropy of the missing slots by hour of day / log2(24) | 0 = always the same hours (meter-side behaviour), 1 = spread over the day (communication) |
| `full_day_share` | share of the gaps that are whole days without data | meter unreachable on some days |
| `trend_slope`, `trend_rise` | least-squares slope of the daily gap fraction, last-vs-first days | worsening or improving; a description of the past, **not a forecast** |

**2. Five scores in [0, 1]** (1 = healthy; missing when they cannot be computed)

| Score | Method |
|---|---|
| availability | 1 − pct_gap |
| peer | robust z-score of pct_gap against the meter's group: `(x − median) / max(1.4826·MAD, 0.01)`; needs ≥ 5 meters in the group |
| stability | EWMA control chart of the daily gap fraction (λ = 0.3, L = 3, robust centre and sigma); penalises alarms |
| voltage | share of **valid** voltage samples outside ±10 % and median deviation (quadratic); invalid values such as relay-open codes are ignored |
| events | penalties per event category; an unresolved tamper (last tamper event = "open", by time order) gives 0 |

**3. Fusion.** A weighted mean over the scores that could be computed. The weights are renormalised, so a
missing module never counts as "perfect". `coverage_of_score` reports the share of the total weight that
had data; it is not a probability.

**4. Diagnosis**, the first rule that applies, with configurable thresholds:

`TAMPER_OPEN` → `GROUP_ISSUE` (high median gap in the meter's group: check the DCU, not the meter) →
`POWER_OUTAGE` (power-failure events cover the gap days) → `NO_DATA` → `DAY_LEVEL_LOSS` → `DETERIORATING`
→ `HARDWARE_SUSPECT` (gaps at the same hours) → `INTERMITTENT_COMMUNICATION` → `VOLTAGE_QUALITY` → `OK`.

Every meter gets a diagnosis, a reason (the numbers that triggered the rule), an action text and a field
priority.

All formulas are **transparent heuristics for ranking**. They are not calibrated against the outcome of
real field inspections; tune the thresholds in `config/health.example.toml` to your fleet.

## Quick start

```bash
pip install -e ".[dev]"                 # or run from the source tree with PYTHONPATH=src

ami-health synthetic --out examples/synthetic          # synthetic fleet (already committed)
ami-health score \
  --meter-days examples/synthetic/meter_days.csv --slots examples/synthetic/slots.csv \
  --voltage examples/synthetic/voltage.csv --events examples/synthetic/events.csv \
  --groups examples/synthetic/groups.csv --group-col dcu_id \
  --config config/health.example.toml --out examples/synthetic/output

# directly from an ami-data-quality output folder (truncated border days are excluded automatically)
ami-health from-ami-dq --dir path/to/ami-dq-output --groups meter_groups.csv --group-col dcu_id --out out/
```

Outputs: `health.csv` (every meter, every score and feature), `field_list.csv` (meters needing action, in
priority order), `group_summary.csv`, and `run_manifest.json` with the SHA-256 of the inputs. CSVs are
byte-identical between runs.

### Inputs

| Table | Columns | Required |
|---|---|---|
| meter days | `meter_id, local_date, expected_slots, received` | yes |
| slots | `meter_id, local_date, slot_index` (received slots) | for `hour_entropy` |
| voltage | `meter_id, ts_local, voltage, quality` | for the voltage score |
| groups | `meter_id, <group column>` | for the peer score and `GROUP_ISSUE` |
| events | `meter_id, ts_local, description` (or `category`) | for the events score, tamper and outages |

Event texts are manufacturer-specific, so they are mapped to categories with configurable regular
expressions (`[event_patterns]`).

## Testing

```bash
python -m pytest --cov=ami_health --cov-branch
```

The synthetic fleet (`examples/synthetic`) contains 26 meters with known archetypes: healthy, gaps at fixed
hours, random gaps, whole days lost, no data, open tamper, closed tamper, supply outage, high voltage,
worsening, and a whole DCU down. A test checks that every archetype gets its expected diagnosis. Others
check each feature and score formula, the time order of tamper events, the renormalised fusion, the
configuration validation, the `ami-data-quality` adapter and the command line.

During development the tool was also run locally on the outputs of `ami-data-quality` for a non-public
extract, as a plausibility check. Those data and results are not part of this repository.

## Limitations

- The thresholds and weights are illustrative defaults, **not calibrated** against field outcomes.
- The power factor is not used: which export column carries the meaningful PF depends on the meter convention.
- On daylight-saving days the hour of a slot is approximated as `slot × 24 / expected slots`.
- The peer score and `GROUP_ISSUE` need a meter → group mapping; they were tested on synthetic groups only.
- No machine-learning anomaly detector yet (an Isolation Forest is on the roadmap); `stability` is a classical EWMA chart.
- A diagnosis is a prioritisation aid, not a fault certificate.

## Origin of the project

This is an independent reimplementation, redesigned and written from scratch, of a meter-diagnosis tool I
originally built while working with AMI data at an engineering firm in Peru. It contains no code, data,
customer names or internal information from that work. Every dataset in the repository is synthetic.
Developed with AI assistance (Claude), which is credited as co-author in the commits.

Spanish notes: [docs/README_es.md](docs/README_es.md).

## Author and license

Brayam Martinez · [MIT](LICENSE)
