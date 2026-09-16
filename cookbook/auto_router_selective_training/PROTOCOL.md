# Selective routing experiment, 15 September 2026

The target is preserving stronger-model task quality while reducing total inference spending through selective use of the cheaper model. No contaminated outcomes from earlier experiments may be used

The primary model pair is GPT-5.6 Luna and GPT-5.6 Sol at high effort. Sonnet-5 and Opus-5 are additional final-evaluation controls. Model identities and gateway billing must match each response. The original capability classifier and fused V2 are baselines, both using the same Luna judge

The final allocation below supersedes the initial targets and intermediate preparation counts recorded in the amendment history. Training uses 160 tasks, validation uses 64 eligible tasks, and final evaluation uses 125 tasks. Each final task has one independent attempt from each of the four models, for 500 attempts

| Benchmark | Training | Validation | Final |
|---|---:|---:|---:|
| MBPP+ | 80 | 29 | 50 |
| LiveCodeBench | 50 | 20 | 25 |
| SWE-bench Verified | 25 | 10 | 25 |
| Terminal-Bench | 5 | 5 | 25 |

The initial final runner used three concurrent SWE task workers and two concurrent Terminal trials. The corrected Anthropic-control rerun uses one SWE worker and two Terminal trials, prioritizing remaining Luna/Sol Terminal attempts. These limits share the machine with other work and do not change individual model budgets. The benchmark selection and offline environment controls make this an adapted subset, not an official leaderboard submission

The following paragraphs preserve the initial plan and its amendments, including why some preparation targets changed

Development uses 25 SWE-bench Verified training tasks and 10 validation tasks, plus 80 MBPP+ training tasks and 30 validation tasks. Terminal-Bench task counts will be fixed after environment-only eligibility checks, targeting at least 15 training and 10 validation tasks. Final evaluation uses 25 SWE-bench, 25 Terminal-Bench and 50 MBPP+ tasks. SWE-bench repositories are disjoint across splits; previously inspected tasks are excluded. Terminal-Bench variants of the same mechanism must remain in the same split

Solver outcomes are new independent paired attempts. No gold fixes, future Git objects, hidden tests, prior trajectories or benchmark results are exposed to solvers. Docker hosts, credentials and task sources are not mounted into solver environments. SWE-bench uses a one-commit Git database with no unreachable objects. Terminal-Bench tests and reference solution are uploaded only in isolated control or grading phases. MBPP solutions are generated from prompts only and executed in a sandbox by EvalPlus

Candidate methods, fixed before final evaluation: original classifiers; adjusted per-model probabilities; regularized task-feature predictors of paired outcomes (both fail, cheap-only, strong-only, both succeed); empirical capability cards using training evidence; and cheap-first verification based only on the prompt and cheap solver trajectory. Training and threshold selection must use development splits only. Cost includes classifiers, verifiers and failed attempts. Shared solver attempts enable task-level policy replay, not claims about independent live router arms or stateful continuation

Hyperparameter selection minimizes validation cost among candidates with zero stronger-only losses and no net quality loss within each benchmark. Also report the validation cost-quality frontier. Always-strong is an allowed fallback, not evidence of training progress. Final policies and route decisions must be frozen before final labels are generated. Hindsight oracle routing is labeled an unattainable reference, never a trained result. Final reports include paired lost/gained solves, escalation precision/recall, per-benchmark performance, cost and small-sample uncertainty

MBPP uses EvalPlus 0.3.1, MBPP+ v0.2.0. SWE-bench uses official grader 4.1.0 and mini-swe-agent 2.0.0. Terminal-Bench uses Harbor 0.23.0 with a metered mini-swe-agent adapter. No default production routing policy is changed based on an exploratory benchmark

Control-only amendment: MBPP/255 is excluded from scoring before reading any model grades because its own reference implementation fails an extreme combination-output case under the sandbox memory limit. All attempts and costs are retained. MBPP scoring has 80 training, 29 validation and 50 final tasks. SWE-bench native-image eligibility yielded a repository-disjoint allocation of Django training, SymPy validation, and the remaining eligible repositories for final evaluation. This is a platform-limited subset and must be labeled accordingly

Development selection clarification, before any final outcomes: report two frozen profiles for each classifier. The primary profile maximizes validation solves under the Sol-only total-cost budget, then minimizes cost, while meeting Sol aggregate quality within each benchmark. The secondary profile minimizes cost with zero Sol-only successes lost in validation. The first can gain cheap-only successes while missing a different Sol-only task; report these separately rather than hiding task swaps behind aggregate quality. This distinction follows the requested quality-and-savings objective and does not promise per-task dominance

LiveCodeBench v6 medium/hard problems are added because the MBPP development split contains too few Sol-only rescue examples. Selection is chronological and contest-disjoint:50train/20validation/25test. Prompts alone go to the solver. The official pinned harness runs public and private tests in a network-disabled Docker grader, after synthetic positive/negative grader controls. This addition was chosen using development evidence, before final outcomes

Forecast-condition correction: MBPP and LiveCodeBench original forecasts mistakenly described an agent with shell/test feedback. Those development forecasts and provisional fits are archived under development_harness_correction. Replacement forecasts specify their actual single-response/no-tools/8192-token setting. Solver outputs, grades, splits and final-task quarantine are unchanged. The earlier development savings diagnostic is superseded pending refit

Runtime feature contract: upfront learned heads use only the existing classifier verdict, including probabilities, rule/demand/verification categories and bounded features of its crux text. They do not use benchmark identity, task ID, repository identity or hidden labels. This permits deterministic inference from one classifier response. Post-attempt review policies additionally need the solver output/review and are evaluated as a separate cascade

Transport amendment before final evaluation: difficult LiveCodeBench responses exceeded the initial180-second HTTP read timeout. Increase the transport allowance to600seconds without changing model effort or token budgets; retain every completed response and resume only missing outputs. Early unresolved transport calls have unknown billing and are reported separately as research overhead, never treated as free calls or wrong-code grades. Subsequent requests have durable start/response/error records

Control scheduling: Terminal-Bench environment validation uses two disjoint workers. The original worker handles the first26candidates; a second worker reserves and processes the final26. This changes only preparation throughput, not solver budgets, task criteria or selection

Terminal-Bench development can start on five eligible tasks while remaining environment controls complete. These tasks are assigned permanently to training before inspecting their paired outcomes, including the known pilot task. The final25-task reserve and validation set will be fixed using only remaining environment eligibility and task identity. Any additional training allocation must respect that reserve

Agentic forecast conditions are also normalized before fitting: both classifier prompts now state the actual disconnected shell, token budget and command timeout; capability names Luna explicitly, and Terminal-Bench includes each task wall-clock budget. The previous development-only agentic forecasts are archived; solver attempts and grades are unchanged.

The empirical card variant uses one prespecified Sol synthesis of training-only paired outcomes and fallible classifier/reviewer descriptions, plus measured aggregate counts by execution family. It must contain general mechanisms, no task identities or solutions, and is not rewritten against validation outcomes. Deterministic learned heads additionally compare raw threshold tuning, scalar probability calibration, task-dependent calibration, paired outcomes, Sol-only rescue risk, and expected incremental quality per predicted dollar. Cost prediction is fitted on training costs only. Fixed threshold grids replace validation-specific score cutoffs.

Final-label quarantine: freeze fitted policies before launching final attempts and upfront routes before final solver runs. A post-attempt cascade necessarily waits for Luna output; its deterministic route freezer loads only classifier forecasts, Luna attempts and reviews, never grade files. Harbor may write automatic grades on disk, but aggregate final analysis and model comparisons remain quarantined until all routes are frozen. No policy tuning or card revision occurs after final inference begins.

Terminal-Bench validation begins on five hash-ordered currently control-eligible tasks, fixed before their model attempts. The existing five training tasks stay fixed; at least 25 final tasks will be allocated from other eligible groups. Additional eligible tasks may enlarge development only after this final reserve is secured. This rolling preparation allocation is a convenience sample, not the full Terminal-Bench distribution.

Post-attempt variants can use deterministic checks of task-supplied public examples on MBPP+ and LiveCodeBench. The checker input excludes private test cases. The resulting available/passed signal is an additional post-attempt feature, and public-check-only escalation is reported as a separate baseline where examples exist. Checker CPU time is reported separately from inference dollars. The original solver attempts still receive no repair feedback.

Three original SWE final candidates failed their gold-reference controls. They are replaced, before final inference, by the first three passing tasks in a preregistered hash-ordered unused Sphinx pool. Training and validation remain unchanged and final evaluation retains 25 tasks. Exclusion records and original allocation are retained.

Before final Terminal-Bench inference, agent isolation expands to every nested .git path in the container, verified by a synthetic future-commit probe. Existing development task Dockerfiles were checked: their only source Git checkout is the already-sanitized /app Bottle repository. New trial network audits also record filesystem layers and an image-configuration digest, and retain an image tag for reproduction. Final model comparisons must verify matching task filesystem layers.

Final solver scheduling uses up to four concurrent tasks/trials per agentic benchmark, while retaining every individual task/model resource and token limit. The USD 5 stopping rule is checked before the next query and can exceed that amount by the last billed response. Known billing is recorded in full.

Development allocation is now fixed at 160 paired training tasks and 64 eligible validation tasks: 80/29 MBPP+, 50/20 LiveCodeBench, 25/10 SWE-bench and 5/5 Terminal-Bench. Additional control-eligible Terminal tasks are reserved for final evaluation or left unused. This keeps card synthesis and all calibration fits on one fixed training corpus while final environment preparation continues

Before final inference, freeze a diagnostic ablation per classifier, card variant, stage and training family. Each uses the same validation criterion of zero Sol-only losses, no per-benchmark quality loss and cost no greater than Sol. If no family candidate qualifies, record an explicit always-Sol fallback. Report all of these ablations, without selecting another winner from final scores, to distinguish card changes, probability fitting and threshold fitting

During final execution, a cost audit read a completed Terminal attempt ledger while the periodic exporter was copying that same ledger over its destination. A subsequent read matched the saved run cost exactly. Exported records now use atomic file replacement so concurrent readers never see a partially copied file. No solver attempt, billed response, grade, fitted policy or routing decision was changed


## Terminal runner interruption, September 16 UTC

At 00:06 UTC the Terminal runner, review watcher and keep-awake processes were absent. No terminating error was recorded; the cause remains unknown. The two active Scheme-interpreter containers were still running and neither had a completed agent record or trial result. Their partial transcripts, 95 billed responses ($4.252549 known), outstanding request IDs and container diffs were archived in `infrastructure_interruptions/20260916-terminal-process-loss`. Two outstanding requests have unknown billing and are not assumed free

The 58 completed Terminal trial results were hashed and preserved, without consulting their quality labels. Harbor resumed the identical job configuration and reran only the two interrupted attempts and 40 unstarted trials. The custom agent has no supported trajectory-resume path. The restarted processes use independent process sessions; inference budgets, model settings, task allocation and frozen policies are unchanged. Archived interruption costs belong to research overhead, separately from deployment policy cost. No completed attempt was rerun because of its outcome


## Response-limit diagnostics

Metadata inspection during the still-blinded final run found repeated length-stopped responses that used all 8,192 completion tokens without producing a tool call or visible text. Further inspection discovered the Anthropic message-preservation defect documented below. The affected agentic controls are superseded; their costs remain research overhead, and their responses are excluded from final quality and execution tables. The defect's effect on empty responses has not been established. `execution_diagnostics.json` describes only retained attempts. Empty length stops in corrected attempts are billed behavior under the frozen budget and do not themselves trigger a retry

## Anthropic tool-continuation correction, September 16 UTC

The benchmark client reconstructed assistant messages from content, tool calls and OpenAI reasoning items. This discarded Anthropic `thinking_blocks` and `provider_specific_fields`, which must be preserved on tool continuation. The corrected adapter returns the complete Anthropic assistant message unchanged. Both Sonnet and Opus passed live thinking-plus-tool round trips through the gateway. See the [provider's thinking/tool documentation](https://platform.claude.com/docs/en/build-with-claude/thinking-tool-workflows) and `harness_corrections/anthropic_thinking/live_probe.json`

All 50 SWE and 50 Terminal Anthropic controls are run under the correction, regardless of old outcomes. Fifty completed SWE attempts, 32 completed Terminal attempts and two in-progress Terminal trials were archived with hashes before restarting. The old final quality labels were not inspected. Single-response Anthropic MBPP+/LiveCodeBench controls are unaffected because they have no tool continuation. The OpenAI message merger is unchanged across 3,218 saved responses; 2,691 saved OpenAI tool turns were checked, including 2,678 containing intact reasoning items. All Luna/Sol training, validation, final attempts, frozen coefficients, thresholds and route selections are retained

The archive records 5,360 unaffected files verified unchanged, the preserved trial-result locations, canceled request IDs and the replacement schedule. Superseded controls, interrupted requests and adapter probes are included separately in research spending. Unknown outstanding-request bills are not assumed zero. SWE reruns use the retained peer's image digest and a fresh model-specific grader run prefix. No task, individual inference budget, prompt, grading criterion or fitted policy changed. The corrected Terminal agent identifies itself as adapter3

Once all 250 Luna/Sol final attempts and their grades are complete and the post-attempt routing plan is frozen, a primary-pair report may be generated while the additional Anthropic controls finish. This is an early release of the same prespecified comparison, with all 125 final tasks included; it does not select new policies or refit using final outcomes. The complete report still requires all 500 valid final attempts


## Review-route readiness guard

Before freezing the post-attempt routing plan, the runner now requires each Terminal Luna attempt to have completed grading with no infrastructure exception. Normal agent timeouts remain eligible. This guard inspects only completion and exception metadata; success labels and rewards are not inputs to any routing decision. A synthetic regression check verifies the same readiness result for successful and unsuccessful attempts and rejects transport failures. Learned coefficients, thresholds, features and candidate selections are unchanged


## Host low-power sleep and partial billing, September 16 UTC

The host entered low-power sleep at 02:42:53 UTC with 1% battery and woke on AC at 03:07:43 UTC, a 1,490-second pause despite the existing idle-sleep assertion. All benchmark processes survived. The in-flight Luna requests on regex-chess and path-tracing returned transport ReadError; the pinned mini-swe-agent model wrapper retried them automatically and both attempts continued. No whole attempt was rerun and no global power settings were changed. Power events and request IDs are retained in infrastructure_interruptions/20260916-low-power-sleep

Known billed responses stay attached to their attempts. The two interrupted requests have unknown charges. Policy reports count every unmetered request in the selected solver attempt, classifier/reviewer calls and discarded Luna attempt on escalation. Affected costs are lower bounds; savings versus a fully metered Sol baseline are upper bounds. Missing bills are not assumed zero. If a baseline is also unmetered, its cost comparison is marked unresolved. Cost uncertainty intervals use recorded bills only. This accounting change does not alter coefficients, thresholds or routes

Saved wall durations include host sleep. Elapsed duration is absent from both deterministic route features and reviewer evidence, so the pause itself is not used as a predictive feature. The report makes no latency comparison based on these durations


## Execution budget clarification

SWE solver containers have a 45-minute configured lifetime in both the original and corrected runner. Each shell command has a 90-second timeout with a five-second kill grace. Terminal agent and verifier timeouts come from each pinned task configuration. These environment limits apply in addition to the 150-call, USD 5 pre-query and 8,192-token response limits; this study does not measure unrestricted model performance.


The final primary billing audit also found one earlier DNA-assembly review ConnectTimeout with no successful response ledger. This is separate from the two host-sleep solver interruptions. Research accounting now discovers transport-only ledgers as well as billed-response ledgers. Review-cascade cost comparisons therefore carry three unmetered requests, while Luna-only comparisons carry two
