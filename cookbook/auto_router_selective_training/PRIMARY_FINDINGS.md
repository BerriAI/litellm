# Primary findings: quality retained versus money saved

Training improved some comparisons, but it did not approach the ideal Luna-first policy across all 125 held-out tasks. The profiles selected on validation do not combine preservation of every Sol success with lower total cost across all four benchmarks

Sol solved 87/125 at $20.8964. Luna solved 84/125 at at least $2.6531. There were 11 Sol-only successes, eight Luna-only successes, 76 successes shared by both, and 30 tasks both attempts failed. A perfect Luna-first decision would therefore reach 95/125 at at least $5.2653. That is an unattainable hindsight reference, with at most 74.8% savings using the recorded bills

| Frozen comparison | Result | Cost | Sol successes lost |
|---|---|---:|---:|
| Capability retention profile, SWE-bench | 18/25, matching Sol | $6.2493, 41.7% savings | 0 |
| V2 quality profile, Terminal-Bench | 19/25 versus Sol's 18/25 | $8.1712, 3.3% savings | 0 |
| V2 retention profile, all tasks | 85/125 versus Sol's 87/125 | $14.0036, 33.0% savings | 4 |
| V2 review retention profile, all tasks | 88/125 versus Sol's 87/125 | At least $20.9754, higher than Sol | 0 |
| V2 review quality profile, all tasks | 88/125 versus Sol's 87/125 | At least $8.6493, at most 58.6% savings | 7 |

These benchmark-specific successes are separate frozen policies. Combining the winning profile for each benchmark would be a new policy requiring fresh evaluation

Relative to the original V2 baseline, the trained V2 retention profile improves aggregate quality from 84 to 85 solves and reduces cost from at least $17.0029 to $14.0036. That is at least a 17.6% cost reduction against that baseline. It still misses four Sol successes, so it does not meet the stronger preservation goal

Among the 40 prespecified family controls, three capability variants retained every Sol success and gained one additional solve. The per-model adjustment control achieved 88/125 at $20.3797, a 2.5% saving. The paired and rescue controls saved 0.2% and 0.6%. These are exploratory comparisons across many variants, not a new winner selection. All twelve upfront family configurations per classifier are provided for reproduction and fresh benchmarking; none was refitted on these outcomes

The expensive V2 review policy captured all 11 Sol-only successes by escalating 115 of 125 tasks. Its cheaper review policy escalated 15 tasks but captured only four of those 11 rescues. This is the remaining problem: identify the rare tasks where Sol changes failure into success without also sending most other tasks to Sol

Calibration improved more than routing precision. V2's selected scalar calibration reduced the Luna Brier score from 0.223 to 0.205 and the Sol Brier score from 0.237 to 0.211. However, probability-gap MSE only moved from 0.153 to 0.151. Ranking Sol-only rescues by the estimated gap improved from ROC AUC 0.618 to 0.689, still far from perfect separation. The empirical review variant reached 0.742 gap-ranking AUC, but its frozen economical threshold missed seven rescues. A more accurate average success probability does not by itself isolate the rescue tasks

The selected upfront profiles retain the original cards. Their improvements come from fitted adjustments and decision boundaries. Card rewrites alone gave mixed results. Training contained only 13 Sol-only successes, including two agentic examples, which limits evidence for agentic rescue decisions. Additional paired agentic training data and stronger verification signals are plausible next directions; this study does not establish their effect

All policies and family controls were frozen before final outcomes. These are task-level replays of independent attempts, not per-turn routing or Sol continuation from a Luna patch. Small samples and the search across development candidates limit generalization. One attempt per model does not establish a stable solve probability for an individual task

Two Luna solver requests interrupted by host sleep have unknown charges. An earlier unavailable DNA-assembly review has one additional unmetered connection-timeout request. Thus Luna-only comparisons contain two unmetered requests, and review cascades contain three. The report marks affected costs as lower bounds and savings as upper bounds. The three upfront comparisons quoted with exact costs above are fully metered

The additional Sonnet/Opus controls are still running. Their completion will extend the comparison; it cannot change these frozen Luna/Sol results
