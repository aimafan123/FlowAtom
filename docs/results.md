# Results and reproducibility status

## Measured seed-2025 run

Using seed 2025, the run completed closed-world and target-present open-world
experiments for Direct HTTPS, Trojan and VMess. It reused a frozen externally
pretrained encoder and rebuilt each scenario's Atom vocabulary, trace responses
and predictor. Aggregate measurements and input fingerprints are in
[seed2025.json](results/seed2025.json).

### Matched evaluation windows

Micro-F1 is shown as a percentage. Closed world uses the 480 saved paper test
windows. Open world uses the same 12,500 windows, with `m=1..5`, `b=1..5` and
500 windows per cell.

| Scenario | Closed world | Open world |
| --- | ---: | ---: |
| Direct HTTPS | 97.86 | 92.52 |
| Trojan | 94.31 | 93.34 |
| VMess | 94.28 | 89.60 |

### Default generated windows and paper subset

| Scenario | Closed, generated windows | Open, b=5 and m=2..5 | Original seed, same open subset | Atoms | Training flows |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct HTTPS | 97.50 | 91.02 | 91.42 | 1190 | 387318 |
| Trojan | 94.66 | 92.75 | 91.71 | 1176 | 356139 |
| VMess | 94.55 | 88.13 | 88.07 | 1191 | 429516 |

Each mainline run used 15,000 training windows, 1,500 validation windows and
480 generated test windows. The complete three-scenario campaign took about
135 minutes on an RTX A6000.
