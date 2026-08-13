# PAQO — Agentic Storage QoS Evidence

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-live-73e0ba)](https://darisisuresh.github.io/agentic-storage-qos-evidence/)
[![AEGIS](https://img.shields.io/badge/AEGIS-2.4.0-ff6b3d)](https://github.com/sunilgentyala/aegis-integrity)
[![Privacy](https://img.shields.io/badge/manuscript-private-071b1c)](#privacy-controls)
[![Tests](https://img.shields.io/badge/tests-129%20passed-73e0ba)](reports/aegis-integrity-report.html)

This repository contains sanitized, reproducible evidence supporting an IEEE-style manuscript about predictive, safety-constrained storage QoS optimization.

The manuscript and extracted text are intentionally excluded. Public artifacts contain checksums, methodology, aggregate findings, tool versions, and limitations only.

## Evidence snapshot

- Manuscript identity: SHA-256 only; no paper content is tracked.
- AEGIS Integrity: v2.4.0, source commit `dfd6d15bb7adf6ca0e23ec70ac3dadaa2ae5304e`.
- Tool validation: 129 upstream tests passed after installing the test-only `httpx2` dependency omitted from the package metadata.
- Plagiarism: inconclusive because no independent comparison corpus was supplied.
- AI/watermark analysis: unavailable; AEGIS terminated in its experimental watermark path after parsing the manuscript.
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
| Mean P99 latency | 23.7% lower | Predictive vs. static QoS in the stated synthetic model |
| P99 SLO violations | 46.3% fewer | Predictive vs. static QoS in the stated synthetic model |
| AEGIS upstream tests | 129 passed | Tool suite validated locally |
| Public manuscript files | 0 | Privacy control enforced |

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
  author       = {Darisi, Suresh Kumar},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/darisisuresh/agentic-storage-qos-evidence}
}
```

## Author

**Suresh kumar Darisi** — Independent Researcher

[GitHub](https://github.com/darisisuresh)
