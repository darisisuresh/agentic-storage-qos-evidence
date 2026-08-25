# PAQO — Agentic Storage QoS Evidence

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-live-73e0ba)](https://darisisuresh.github.io/agentic-storage-qos-evidence/)
[![AEGIS](https://img.shields.io/badge/AEGIS-2.4.0-ff6b3d)](https://github.com/sunilgentyala/aegis-integrity)
[![Privacy](https://img.shields.io/badge/manuscript-private-071b1c)](#privacy-controls)
![Framework tests](https://img.shields.io/badge/framework%20tests-15%20passed-73e0ba)
![Validation](https://img.shields.io/badge/validation-100%20seeds-73e0ba)

This repository contains sanitized, reproducible evidence supporting an IEEE-style manuscript about predictive, safety-constrained storage QoS optimization.

The manuscript and extracted text are intentionally excluded. Public artifacts contain checksums, methodology, aggregate findings, tool versions, and limitations only.

## Evidence snapshot

- Manuscript identity: SHA-256 only; no paper content is tracked.
- AEGIS Integrity: v2.4.0, source commit `dfd6d15bb7adf6ca0e23ec70ac3dadaa2ae5304e`.
- Tool validation: 133 upstream tests passed with AEGIS v2.4.0 at source commit `dfd6d15bb7adf6ca0e23ec70ac3dadaa2ae5304e`.
- Plagiarism: inconclusive because no independent comparison corpus was supplied.
- AI analysis: probabilistic score 0.138; this is a supporting signal, not proof of authorship.
- Watermark analysis: experimental `NO_STATISTICAL_ANOMALY`; it did not affect overall risk.
- Citation assessment: 12 of 15 references verified (80%); the remaining three have no DOI. The final rerun reported no citation flags or metadata mismatches.
- Formatting: US Letter; one-column title/abstract; two-column main body; duplex mirrored margins enabled in the proofread copy.

Explore the public evidence experience at **[darisisuresh.github.io/agentic-storage-qos-evidence](https://darisisuresh.github.io/agentic-storage-qos-evidence/)**.

## Architecture

```text
Telemetry → multi-horizon forecast → constrained policy proposal
          → deterministic safety supervisor → platform adapter → outcome feedback
```

PAQO keeps learned recommendation separate from execution authority. The public repository documents the evidence boundary; it does not expose the manuscript.

## Results summary

| Evidence | Result | Interpretation |
|---|---:|---|
| Mean P99 latency | 29.47% lower | Predictive vs. static QoS; nominal scenario, 100 paired seeds |
| P99 SLO violations | 93.87% fewer | Predictive vs. static QoS; nominal scenario, 100 paired seeds |
| Mean synthetic backlog index | 89.76% lower | Dimensionless simulator state; predictive vs. static QoS, 100 paired seeds |
| Predictive vs. MPC P99 | 7.85% lower | Lightweight forecast controller vs. implemented linear-horizon MPC |
| Unsafe proposals | 684/684 blocked | Deterministic mocked fault injection |
| Injected rollbacks | 200/200 issued | Mocked runner; not a production rollback rate |
| Trace-quality audit | 600/600 passed | Zero missing/nonfinite, nonpositive, malformed, or duplicate traces |
| Decision time | 19.79 μs | Mean predictive software decision time; excludes array actuation |
| Framework tests | 15 passed | Unit, rollback, safety, data-quality, model-specification, and benchmark checks |
| AEGIS upstream tests | 133 passed | Tool suite validated locally |
| Public manuscript files | 0 | Privacy control enforced |

All performance values are from a transparent synthetic shared-capacity model, not a production storage array. The stored `queue_depth` field is retained for compatibility but represents a dimensionless calibrated backlog index—not outstanding I/Os. P95/P99 values are deterministic tail-latency surrogates rather than empirical order statistics. Independent seeds—not the 720 interval samples within each seed—are the experimental units.

## Reproduce the validation

```sh
python -m pip install -e '.[dev,validation]'
pytest -q
python benchmarks/run_validation.py --seeds 100
python benchmarks/run_safety_validation.py
python benchmarks/run_data_quality_validation.py
```

The runs write per-seed observations and aggregate reports to `evidence/results/`. The performance report includes two-sided 95% t confidence intervals, paired one-sided Wilcoxon tests, Cohen's dz, an MPC baseline, a no-forecast ablation, three stress levels, and three 15% platform-demand shifts. The trace-quality audit checks 600 generated traces, each containing 720 intervals and four workloads. On the final recorded test host run, all 3,000 controller-scenario runs completed in 52.67 seconds. The fault-injection report uses mocked runners and must not be interpreted as production-array safety or rollback evidence.

## Privacy controls

The `.gitignore` blocks common manuscript formats and raw/extracted content. Before every publication, run:

```sh
./scripts/privacy-check.sh
```

Results from AI detectors are probabilistic signals and cannot establish authorship or misconduct. AEGIS itself describes its results as supporting signals for human review.

## Citation

The companion manuscript remains private while under preparation. Use the repository URL as the public artifact reference:

```bibtex
@misc{darisi2026paqo,
  title        = {{PAQO}: Agentic Storage QoS Optimization Using Predictive I/O Intelligence},
  author       = {Darisi, Suresh Kumar and Gentyala, Sunil},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/darisisuresh/agentic-storage-qos-evidence}
}
```

## Authors

**Suresh Kumar Darisi** — Rocket Software Inc., Boston, Massachusetts, USA<br>
**Sunil Gentyala** — Independent Researcher; HCLTech, America Inc., Dallas, Texas, USA; IEEE Senior Member

[GitHub](https://github.com/darisisuresh)
