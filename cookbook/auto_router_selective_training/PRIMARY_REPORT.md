# Held-out Luna/Sol routing results

All 125 held-out tasks and 250 Luna/Sol attempts. Frozen task-level policy replay; additional Anthropic controls remain in the full report. No refitting or reselection.

Retention and quality denote development-selection objectives, not guarantees. Costs include classifier/reviewer inference and discarded Luna attempts on escalation. These are independent paired attempts; Sol starts from the original task, not from Luna's partial work.

Costs use recorded gateway bills. A ≥ cost and ≤ savings mark unmetered transport requests: cost is a lower bound and savings versus fully metered Sol is an upper bound. The JSON counts these requests per policy and task. They are not assumed free.

## all

| Policy | Solved | Inference cost | Savings vs Sol | Gained / lost vs Sol | Sol-only rescues captured |
|---|---:|---:|---:|---:|---:|
| Luna only | 84/125 | ≥$2.6531 | ≤87.3% | +8 / -11 | 0/11 |
| Sol only | 87/125 | $20.8964 | 0.0% | +0 / -0 | 11/11 |
| Capability upfront, retention objective | 86/125 | $13.8156 | 33.9% | +1 / -2 | 9/11 |
| Capability upfront, quality objective | 88/125 | $18.5237 | 11.4% | +5 / -4 | 7/11 |
| V2 upfront, retention objective | 85/125 | $14.0036 | 33.0% | +2 / -4 | 7/11 |
| V2 upfront, quality objective | 86/125 | $18.1999 | 12.9% | +4 / -5 | 6/11 |
| Capability baseline | 85/125 | $6.7642 | 67.6% | +8 / -10 | 1/11 |
| Capability with learned cards only | 84/125 | $7.2496 | 65.3% | +7 / -10 | 1/11 |
| V2 baseline | 84/125 | ≥$17.0029 | ≤18.6% | +3 / -6 | 5/11 |
| V2 with learned cards only | 85/125 | ≥$14.6487 | ≤29.9% | +6 / -8 | 3/11 |
| Capability review, retention objective | 81/125 | ≥$7.5830 | ≤63.7% | +0 / -6 | 5/11 |
| Capability review, quality objective | 87/125 | ≥$10.1853 | ≤51.3% | +7 / -7 | 4/11 |
| V2 review, retention objective | 88/125 | ≥$20.9754 | ≤-0.4% | +1 / -0 | 11/11 |
| V2 review, quality objective | 88/125 | ≥$8.6493 | ≤58.6% | +8 / -7 | 4/11 |
| Hindsight direct routing (not deployable) | 95/125 | ≥$4.5190 | ≤78.4% | +8 / -0 | 11/11 |
| Hindsight Luna-first routing (not deployable) | 95/125 | ≥$5.2653 | ≤74.8% | +8 / -0 | 11/11 |

## mbpp

| Policy | Solved | Inference cost | Savings vs Sol | Gained / lost vs Sol | Sol-only rescues captured |
|---|---:|---:|---:|---:|---:|
| Luna only | 38/50 | $0.0174 | 91.6% | +3 / -2 | 0/2 |
| Sol only | 37/50 | $0.2064 | 0.0% | +0 / -0 | 2/2 |
| Capability upfront, retention objective | 37/50 | $0.2180 | -5.6% | +0 / -0 | 2/2 |
| Capability upfront, quality objective | 38/50 | $0.0444 | 78.5% | +3 / -2 | 0/2 |
| V2 upfront, retention objective | 37/50 | $0.2298 | -11.3% | +0 / -0 | 2/2 |
| V2 upfront, quality objective | 38/50 | $0.0465 | 77.4% | +3 / -2 | 0/2 |
| Capability baseline | 38/50 | $0.0290 | 85.9% | +3 / -2 | 0/2 |
| Capability with learned cards only | 38/50 | $0.0284 | 86.2% | +3 / -2 | 0/2 |
| V2 baseline | 38/50 | $0.0562 | 72.8% | +3 / -2 | 0/2 |
| V2 with learned cards only | 38/50 | $0.0396 | 80.8% | +3 / -2 | 0/2 |
| Capability review, retention objective | 37/50 | $0.2503 | -21.3% | +0 / -0 | 2/2 |
| Capability review, quality objective | 38/50 | $0.0434 | 79.0% | +3 / -2 | 0/2 |
| V2 review, retention objective | 37/50 | $0.2441 | -18.3% | +0 / -0 | 2/2 |
| V2 review, quality objective | 38/50 | $0.0488 | 76.4% | +3 / -2 | 0/2 |
| Luna with public-test escalation | 38/50 | $0.0240 | 88.4% | +2 / -1 | 1/2 |
| Hindsight direct routing (not deployable) | 40/50 | $0.0353 | 82.9% | +3 / -0 | 2/2 |
| Hindsight Luna-first routing (not deployable) | 40/50 | $0.0361 | 82.5% | +3 / -0 | 2/2 |

## lcb

| Policy | Solved | Inference cost | Savings vs Sol | Gained / lost vs Sol | Sol-only rescues captured |
|---|---:|---:|---:|---:|---:|
| Luna only | 13/25 | $0.1165 | 92.4% | +2 / -3 | 0/3 |
| Sol only | 14/25 | $1.5299 | 0.0% | +0 / -0 | 3/3 |
| Capability upfront, retention objective | 14/25 | $1.5476 | -1.2% | +0 / -0 | 3/3 |
| Capability upfront, quality objective | 14/25 | $0.9200 | 39.9% | +1 / -1 | 2/3 |
| V2 upfront, retention objective | 14/25 | $1.3923 | 9.0% | +0 / -0 | 3/3 |
| V2 upfront, quality objective | 12/25 | $1.1104 | 27.4% | +0 / -2 | 1/3 |
| Capability baseline | 13/25 | $0.2918 | 80.9% | +2 / -3 | 0/3 |
| Capability with learned cards only | 12/25 | $0.4860 | 68.2% | +1 / -3 | 0/3 |
| V2 baseline | 12/25 | $0.9924 | 35.1% | +0 / -2 | 1/3 |
| V2 with learned cards only | 13/25 | $0.4799 | 68.6% | +2 / -3 | 0/3 |
| Capability review, retention objective | 14/25 | $1.6846 | -10.1% | +0 / -0 | 3/3 |
| Capability review, quality objective | 14/25 | $1.3257 | 13.4% | +2 / -2 | 1/3 |
| V2 review, retention objective | 15/25 | $1.6076 | -5.1% | +1 / -0 | 3/3 |
| V2 review, quality objective | 14/25 | $1.0584 | 30.8% | +2 / -2 | 1/3 |
| Luna with public-test escalation | 15/25 | $1.3645 | 10.8% | +2 / -1 | 2/3 |
| Hindsight direct routing (not deployable) | 16/25 | $0.3436 | 77.5% | +2 / -0 | 3/3 |
| Hindsight Luna-first routing (not deployable) | 16/25 | $0.3657 | 76.1% | +2 / -0 | 3/3 |

## swe

| Policy | Solved | Inference cost | Savings vs Sol | Gained / lost vs Sol | Sol-only rescues captured |
|---|---:|---:|---:|---:|---:|
| Luna only | 16/25 | $0.6371 | 94.1% | +0 / -2 | 0/2 |
| Sol only | 18/25 | $10.7132 | 0.0% | +0 / -0 | 2/2 |
| Capability upfront, retention objective | 18/25 | $6.2493 | 41.7% | +0 / -0 | 2/2 |
| Capability upfront, quality objective | 18/25 | $9.7546 | 8.9% | +0 / -0 | 2/2 |
| V2 upfront, retention objective | 16/25 | $5.3537 | 50.0% | +0 / -2 | 0/2 |
| V2 upfront, quality objective | 17/25 | $8.8718 | 17.2% | +0 / -1 | 1/2 |
| Capability baseline | 16/25 | $1.3893 | 87.0% | +0 / -2 | 0/2 |
| Capability with learned cards only | 16/25 | $1.6826 | 84.3% | +0 / -2 | 0/2 |
| V2 baseline | 17/25 | $9.1304 | 14.8% | +0 / -1 | 1/2 |
| V2 with learned cards only | 17/25 | $7.4181 | 30.8% | +0 / -1 | 1/2 |
| Capability review, retention objective | 16/25 | $2.4752 | 76.9% | +0 / -2 | 0/2 |
| Capability review, quality objective | 16/25 | $0.9753 | 90.9% | +0 / -2 | 0/2 |
| V2 review, retention objective | 18/25 | $9.1263 | 14.8% | +0 / -0 | 2/2 |
| V2 review, quality objective | 16/25 | $0.9775 | 90.9% | +0 / -2 | 0/2 |
| Hindsight direct routing (not deployable) | 18/25 | $1.4342 | 86.6% | +0 / -0 | 2/2 |
| Hindsight Luna-first routing (not deployable) | 18/25 | $1.5478 | 85.6% | +0 / -0 | 2/2 |

## terminal

| Policy | Solved | Inference cost | Savings vs Sol | Gained / lost vs Sol | Sol-only rescues captured |
|---|---:|---:|---:|---:|---:|
| Luna only | 17/25 | ≥$1.8822 | ≤77.7% | +3 / -4 | 0/4 |
| Sol only | 18/25 | $8.4468 | 0.0% | +0 / -0 | 4/4 |
| Capability upfront, retention objective | 17/25 | $5.8008 | 31.3% | +1 / -2 | 2/4 |
| Capability upfront, quality objective | 18/25 | $7.8047 | 7.6% | +1 / -1 | 3/4 |
| V2 upfront, retention objective | 18/25 | $7.0279 | 16.8% | +2 / -2 | 2/4 |
| V2 upfront, quality objective | 19/25 | $8.1712 | 3.3% | +1 / -0 | 4/4 |
| Capability baseline | 18/25 | $5.0541 | 40.2% | +3 / -3 | 1/4 |
| Capability with learned cards only | 18/25 | $5.0526 | 40.2% | +3 / -3 | 1/4 |
| V2 baseline | 17/25 | ≥$6.8238 | ≤19.2% | +0 / -1 | 3/4 |
| V2 with learned cards only | 17/25 | ≥$6.7112 | ≤20.5% | +1 / -2 | 2/4 |
| Capability review, retention objective | 14/25 | ≥$3.1729 | ≤62.4% | +0 / -4 | 0/4 |
| Capability review, quality objective | 19/25 | ≥$7.8410 | ≤7.2% | +2 / -1 | 3/4 |
| V2 review, retention objective | 18/25 | ≥$9.9975 | ≤-18.4% | +0 / -0 | 4/4 |
| V2 review, quality objective | 20/25 | ≥$6.5646 | ≤22.3% | +3 / -1 | 3/4 |
| Hindsight direct routing (not deployable) | 21/25 | ≥$2.7059 | ≤68.0% | +3 / -0 | 4/4 |
| Hindsight Luna-first routing (not deployable) | 21/25 | ≥$3.3157 | ≤60.7% | +3 / -0 | 4/4 |

The hindsight rows use hidden outcomes and are unattainable routing references. One attempt per model and small per-benchmark samples do not establish future reliability. JSON records exploratory paired uncertainty, every frozen ablation and per-task decisions. Study training and discarded harness-run spending is separate from deployed-policy cost.
