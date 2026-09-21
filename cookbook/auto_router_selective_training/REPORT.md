# Clean selective-router evaluation

Policies were frozen before final solver outcomes. Upfront routing uses one classifier response; the post-attempt variants also use Luna output, optional public-example checks and a Luna review. These are task-level replays over matched independent attempts, not live per-turn routing or continuation from a cheaper model's patch

The configured original-card baselines receive the same model identity and execution conditions as the trained variants. Capability uses base_threshold=0.5 and threshold_step=0.1; V2 uses max_quality_gap=0.05. The empirical-card baselines change the card supplement while retaining those boundaries

Retention profiles were selected to preserve Sol successes at lower validation cost. Quality profiles were selected to maximize validation solves within the Sol budget. These names describe selection objectives, not guarantees about the held-out results. Exact policy IDs remain in final_results.json

Costs use recorded gateway bills. A ≥ cost and ≤ savings mark unmetered transport requests: cost is a lower bound and savings versus fully metered Sol is an upper bound. The JSON counts these requests per policy and task. They are not assumed free.

## Observed opportunity for routing

The Sol-only column counts the rescues a perfect decision would capture. The hindsight cascade pays for Luna on every task and adds Sol only for those rescues. Tasks where both attempts failed cannot be recovered by choosing between these two saved outputs. This is a bound on this replay, not a guarantee about future attempts

| Benchmark | Both solve | Luna only | Sol only | Both fail | Hindsight cascade cost |
|---|---:|---:|---:|---:|---:|
| all | 76 | 8 | 11 | 30 | ≥$5.2653 |
| lcb | 11 | 2 | 3 | 9 | $0.3657 |
| mbpp | 35 | 3 | 2 | 10 | $0.0361 |
| swe | 16 | 0 | 2 | 7 | $1.5478 |
| terminal | 14 | 3 | 4 | 4 | ≥$3.3157 |

## all

| Policy | Solved | Inference cost | Savings vs Sol | Gained / lost vs Sol |
|---|---:|---:|---:|---:|
| Luna only | 84/125 | ≥$2.6531 | ≤87.3% | +8 / -11 |
| Sol only | 87/125 | $20.8964 | 0.0% | +0 / -0 |
| Sonnet 5 only | 81/125 | ≥$37.9961 | ≤-81.8% | +8 / -14 |
| Opus 5 only | 99/125 | ≥$46.0550 | ≤-120.4% | +19 / -7 |
| Capability upfront, retention objective | 86/125 | $13.8156 | 33.9% | +1 / -2 |
| Capability upfront, quality objective | 88/125 | $18.5237 | 11.4% | +5 / -4 |
| V2 upfront, retention objective | 85/125 | $14.0036 | 33.0% | +2 / -4 |
| V2 upfront, quality objective | 86/125 | $18.1999 | 12.9% | +4 / -5 |
| Capability baseline | 85/125 | $6.7642 | 67.6% | +8 / -10 |
| Capability with learned cards only | 84/125 | $7.2496 | 65.3% | +7 / -10 |
| V2 baseline | 84/125 | ≥$17.0029 | ≤18.6% | +3 / -6 |
| V2 with learned cards only | 85/125 | ≥$14.6487 | ≤29.9% | +6 / -8 |
| Capability review, retention objective | 81/125 | ≥$7.5830 | ≤63.7% | +0 / -6 |
| Capability review, quality objective | 87/125 | ≥$10.1853 | ≤51.3% | +7 / -7 |
| V2 review, retention objective | 88/125 | ≥$20.9754 | ≤-0.4% | +1 / -0 |
| V2 review, quality objective | 88/125 | ≥$8.6493 | ≤58.6% | +8 / -7 |
| Hindsight direct routing (not deployable) | 95/125 | ≥$4.5190 | ≤78.4% | +8 / -0 |
| Hindsight Luna-first routing (not deployable) | 95/125 | ≥$5.2653 | ≤74.8% | +8 / -0 |

## lcb

| Policy | Solved | Inference cost | Savings vs Sol | Gained / lost vs Sol |
|---|---:|---:|---:|---:|
| Luna only | 13/25 | $0.1165 | 92.4% | +2 / -3 |
| Sol only | 14/25 | $1.5299 | 0.0% | +0 / -0 |
| Sonnet 5 only | 9/25 | $1.3790 | 9.9% | +1 / -6 |
| Opus 5 only | 14/25 | $2.3849 | -55.9% | +2 / -2 |
| Capability upfront, retention objective | 14/25 | $1.5476 | -1.2% | +0 / -0 |
| Capability upfront, quality objective | 14/25 | $0.9200 | 39.9% | +1 / -1 |
| V2 upfront, retention objective | 14/25 | $1.3923 | 9.0% | +0 / -0 |
| V2 upfront, quality objective | 12/25 | $1.1104 | 27.4% | +0 / -2 |
| Capability baseline | 13/25 | $0.2918 | 80.9% | +2 / -3 |
| Capability with learned cards only | 12/25 | $0.4860 | 68.2% | +1 / -3 |
| V2 baseline | 12/25 | $0.9924 | 35.1% | +0 / -2 |
| V2 with learned cards only | 13/25 | $0.4799 | 68.6% | +2 / -3 |
| Capability review, retention objective | 14/25 | $1.6846 | -10.1% | +0 / -0 |
| Capability review, quality objective | 14/25 | $1.3257 | 13.4% | +2 / -2 |
| V2 review, retention objective | 15/25 | $1.6076 | -5.1% | +1 / -0 |
| V2 review, quality objective | 14/25 | $1.0584 | 30.8% | +2 / -2 |
| Luna with public-test escalation | 15/25 | $1.3645 | 10.8% | +2 / -1 |
| Hindsight direct routing (not deployable) | 16/25 | $0.3436 | 77.5% | +2 / -0 |
| Hindsight Luna-first routing (not deployable) | 16/25 | $0.3657 | 76.1% | +2 / -0 |

## mbpp

| Policy | Solved | Inference cost | Savings vs Sol | Gained / lost vs Sol |
|---|---:|---:|---:|---:|
| Luna only | 38/50 | $0.0174 | 91.6% | +3 / -2 |
| Sol only | 37/50 | $0.2064 | 0.0% | +0 / -0 |
| Sonnet 5 only | 36/50 | $0.0988 | 52.1% | +2 / -3 |
| Opus 5 only | 42/50 | $0.2249 | -9.0% | +7 / -2 |
| Capability upfront, retention objective | 37/50 | $0.2180 | -5.6% | +0 / -0 |
| Capability upfront, quality objective | 38/50 | $0.0444 | 78.5% | +3 / -2 |
| V2 upfront, retention objective | 37/50 | $0.2298 | -11.3% | +0 / -0 |
| V2 upfront, quality objective | 38/50 | $0.0465 | 77.4% | +3 / -2 |
| Capability baseline | 38/50 | $0.0290 | 85.9% | +3 / -2 |
| Capability with learned cards only | 38/50 | $0.0284 | 86.2% | +3 / -2 |
| V2 baseline | 38/50 | $0.0562 | 72.8% | +3 / -2 |
| V2 with learned cards only | 38/50 | $0.0396 | 80.8% | +3 / -2 |
| Capability review, retention objective | 37/50 | $0.2503 | -21.3% | +0 / -0 |
| Capability review, quality objective | 38/50 | $0.0434 | 79.0% | +3 / -2 |
| V2 review, retention objective | 37/50 | $0.2441 | -18.3% | +0 / -0 |
| V2 review, quality objective | 38/50 | $0.0488 | 76.4% | +3 / -2 |
| Luna with public-test escalation | 38/50 | $0.0240 | 88.4% | +2 / -1 |
| Hindsight direct routing (not deployable) | 40/50 | $0.0353 | 82.9% | +3 / -0 |
| Hindsight Luna-first routing (not deployable) | 40/50 | $0.0361 | 82.5% | +3 / -0 |

## swe

| Policy | Solved | Inference cost | Savings vs Sol | Gained / lost vs Sol |
|---|---:|---:|---:|---:|
| Luna only | 16/25 | $0.6371 | 94.1% | +0 / -2 |
| Sol only | 18/25 | $10.7132 | 0.0% | +0 / -0 |
| Sonnet 5 only | 19/25 | $11.0537 | -3.2% | +2 / -1 |
| Opus 5 only | 22/25 | $10.5296 | 1.7% | +4 / -0 |
| Capability upfront, retention objective | 18/25 | $6.2493 | 41.7% | +0 / -0 |
| Capability upfront, quality objective | 18/25 | $9.7546 | 8.9% | +0 / -0 |
| V2 upfront, retention objective | 16/25 | $5.3537 | 50.0% | +0 / -2 |
| V2 upfront, quality objective | 17/25 | $8.8718 | 17.2% | +0 / -1 |
| Capability baseline | 16/25 | $1.3893 | 87.0% | +0 / -2 |
| Capability with learned cards only | 16/25 | $1.6826 | 84.3% | +0 / -2 |
| V2 baseline | 17/25 | $9.1304 | 14.8% | +0 / -1 |
| V2 with learned cards only | 17/25 | $7.4181 | 30.8% | +0 / -1 |
| Capability review, retention objective | 16/25 | $2.4752 | 76.9% | +0 / -2 |
| Capability review, quality objective | 16/25 | $0.9753 | 90.9% | +0 / -2 |
| V2 review, retention objective | 18/25 | $9.1263 | 14.8% | +0 / -0 |
| V2 review, quality objective | 16/25 | $0.9775 | 90.9% | +0 / -2 |
| Hindsight direct routing (not deployable) | 18/25 | $1.4342 | 86.6% | +0 / -0 |
| Hindsight Luna-first routing (not deployable) | 18/25 | $1.5478 | 85.6% | +0 / -0 |

## terminal

| Policy | Solved | Inference cost | Savings vs Sol | Gained / lost vs Sol |
|---|---:|---:|---:|---:|
| Luna only | 17/25 | ≥$1.8822 | ≤77.7% | +3 / -4 |
| Sol only | 18/25 | $8.4468 | 0.0% | +0 / -0 |
| Sonnet 5 only | 17/25 | ≥$25.4647 | ≤-201.5% | +3 / -4 |
| Opus 5 only | 21/25 | ≥$32.9157 | ≤-289.7% | +6 / -3 |
| Capability upfront, retention objective | 17/25 | $5.8008 | 31.3% | +1 / -2 |
| Capability upfront, quality objective | 18/25 | $7.8047 | 7.6% | +1 / -1 |
| V2 upfront, retention objective | 18/25 | $7.0279 | 16.8% | +2 / -2 |
| V2 upfront, quality objective | 19/25 | $8.1712 | 3.3% | +1 / -0 |
| Capability baseline | 18/25 | $5.0541 | 40.2% | +3 / -3 |
| Capability with learned cards only | 18/25 | $5.0526 | 40.2% | +3 / -3 |
| V2 baseline | 17/25 | ≥$6.8238 | ≤19.2% | +0 / -1 |
| V2 with learned cards only | 17/25 | ≥$6.7112 | ≤20.5% | +1 / -2 |
| Capability review, retention objective | 14/25 | ≥$3.1729 | ≤62.4% | +0 / -4 |
| Capability review, quality objective | 19/25 | ≥$7.8410 | ≤7.2% | +2 / -1 |
| V2 review, retention objective | 18/25 | ≥$9.9975 | ≤-18.4% | +0 / -0 |
| V2 review, quality objective | 20/25 | ≥$6.5646 | ≤22.3% | +3 / -1 |
| Hindsight direct routing (not deployable) | 21/25 | ≥$2.7059 | ≤68.0% | +3 / -0 |
| Hindsight Luna-first routing (not deployable) | 21/25 | ≥$3.3157 | ≤60.7% | +3 / -0 |

## Frozen family comparisons

Each row was selected on development data before final inference. These are diagnostic comparisons, not a new winner selection on the held-out tasks. An always-Sol fallback means that no candidate in that requested family met the validation criterion; it bypasses the classifier, Luna attempt and reviewer

| Classifier | Requested stage | Cards | Fitted target | Boundary | Solved | Inference cost | Savings vs Sol | Gained / lost vs Sol |
|---|---|---|---|---:|---:|---:|---:|---:|
| cap | upfront | empirical | benefit_per_dollar | always Sol fallback | 87/125 | $20.8964 | 0.0% | +0 / -0 |
| cap | upfront | empirical | paired | always Sol fallback | 87/125 | $20.8964 | 0.0% | +0 / -0 |
| cap | upfront | empirical | per_model | always Sol fallback | 87/125 | $20.8964 | 0.0% | +0 / -0 |
| cap | upfront | empirical | raw | always Sol fallback | 87/125 | $20.8964 | 0.0% | +0 / -0 |
| cap | upfront | empirical | rescue | always Sol fallback | 87/125 | $20.8964 | 0.0% | +0 / -0 |
| cap | upfront | empirical | scalar_calibration | always Sol fallback | 87/125 | $20.8964 | 0.0% | +0 / -0 |
| cap | review | empirical | paired | always Sol fallback | 87/125 | $20.8964 | 0.0% | +0 / -0 |
| cap | review | empirical | per_model | always Sol fallback | 87/125 | $20.8964 | 0.0% | +0 / -0 |
| cap | review | empirical | rescue | always Sol fallback | 87/125 | $20.8964 | 0.0% | +0 / -0 |
| cap | upfront | original | benefit_per_dollar | 1 | 86/125 | $13.8156 | 33.9% | +1 / -2 |
| cap | upfront | original | paired | -0.02 | 88/125 | $20.8563 | 0.2% | +1 / -0 |
| cap | upfront | original | per_model | -0.05 | 88/125 | $20.3797 | 2.5% | +1 / -0 |
| cap | upfront | original | raw | always Sol fallback | 87/125 | $20.8964 | 0.0% | +0 / -0 |
| cap | upfront | original | rescue | 0.03 | 88/125 | $20.7799 | 0.6% | +1 / -0 |
| cap | upfront | original | scalar_calibration | always Sol fallback | 87/125 | $20.8964 | 0.0% | +0 / -0 |
| cap | review | original | paired | always Sol fallback | 87/125 | $20.8964 | 0.0% | +0 / -0 |
| cap | review | original | per_model | always Sol fallback | 87/125 | $20.8964 | 0.0% | +0 / -0 |
| cap | review | original | rescue | always Sol fallback | 87/125 | $20.8964 | 0.0% | +0 / -0 |
| v2 | upfront | empirical | benefit_per_dollar | 0.5 | 87/125 | $19.2534 | 7.9% | +2 / -2 |
| v2 | upfront | empirical | paired | 0.01 | 85/125 | $19.7030 | 5.7% | +1 / -3 |
| v2 | upfront | empirical | per_model | 0.01 | 86/125 | $19.7351 | 5.6% | +1 / -2 |
| v2 | upfront | empirical | raw | always Sol fallback | 87/125 | $20.8964 | 0.0% | +0 / -0 |
| v2 | upfront | empirical | rescue | 0.05 | 86/125 | $18.0048 | 13.8% | +2 / -3 |
| v2 | upfront | empirical | scalar_calibration | always Sol fallback | 87/125 | $20.8964 | 0.0% | +0 / -0 |
| v2 | review | empirical | paired | always Sol fallback | 87/125 | $20.8964 | 0.0% | +0 / -0 |
| v2 | review | empirical | per_model | always Sol fallback | 87/125 | $20.8964 | 0.0% | +0 / -0 |
| v2 | review | empirical | rescue | always Sol fallback | 87/125 | $20.8964 | 0.0% | +0 / -0 |
| v2 | upfront | original | benefit_per_dollar | 1 | 85/125 | $14.0036 | 33.0% | +2 / -4 |
| v2 | upfront | original | paired | always Sol fallback | 87/125 | $20.8964 | 0.0% | +0 / -0 |
| v2 | upfront | original | per_model | 0 | 86/125 | $20.1971 | 3.3% | +0 / -1 |
| v2 | upfront | original | raw | always Sol fallback | 87/125 | $20.8964 | 0.0% | +0 / -0 |
| v2 | upfront | original | rescue | always Sol fallback | 87/125 | $20.8964 | 0.0% | +0 / -0 |
| v2 | upfront | original | scalar_calibration | always Sol fallback | 87/125 | $20.8964 | 0.0% | +0 / -0 |
| v2 | review | original | paired | always Sol fallback | 87/125 | $20.8964 | 0.0% | +0 / -0 |
| cap | review | empirical | benefit_per_dollar | 0.1 | 82/125 | ≥$16.8075 | ≤19.6% | +0 / -5 |
| cap | review | original | benefit_per_dollar | 0.25 | 81/125 | ≥$7.5830 | ≤63.7% | +0 / -6 |
| v2 | review | empirical | benefit_per_dollar | 0.25 | 83/125 | ≥$16.3243 | ≤21.9% | +0 / -4 |
| v2 | review | original | benefit_per_dollar | 0.1 | 85/125 | ≥$19.8241 | ≤5.1% | +0 / -2 |
| v2 | review | original | per_model | -0.02 | 88/125 | ≥$20.9754 | ≤-0.4% | +1 / -0 |
| v2 | review | original | rescue | 0.02 | 86/125 | ≥$23.4088 | ≤-12.0% | +0 / -1 |

## Response budget effects

Every solver uses the same 8,192-token response cap and high reasoning effort. Length stops that produce neither a tool call nor visible text still consume inference cost. They are retained as model behavior under this budget, not retried as infrastructure failures. These counts describe completed final attempts across all four benchmarks

| Model | Responses | Length stops | Length stops without action or text | Cost of those empty length stops |
|---|---:|---:|---:|---:|
| gpt-5.6-luna | 1341 | 50 | 42 | $0.4217 |
| gpt-5.6-sol | 819 | 7 | 4 | $0.6676 |
| claude-sonnet-5 | 1703 | 124 | 116 | $10.3656 |
| claude-opus-5 | 870 | 87 | 83 | $17.9127 |

## Experiment spending

Recorded study inference cost is $225.5557. This includes training, validation, final solver/judge/reviewer calls, superseded development forecasts, archived infrastructure-interrupted attempts, superseded Anthropic agentic controls, controls with mismatched Terminal images, and adapter round-trip probes. It is separate from the deployed-policy costs above. Unknown transport billing, setup, live-proxy QA, local Docker and CPU expenses are not included. See research_costs.json for the breakdown

## Interpretation limits

The oracle is a hindsight reference using known costs. Cost intervals use recorded bills and exclude unmetered requests. A small sample with one attempt per model cannot establish the same routing advantage across workloads or repeated runs. Paired bootstrap intervals and exact discordant-pair tests in the JSON are exploratory; policy selection searched many development candidates. Inference costs include classifier/reviewer calls and discarded Luna attempts on escalation. Local checker CPU time and experiment training/transport overhead are separate. Terminal-Bench and SWE-bench use documented offline/native-image eligibility adaptations
