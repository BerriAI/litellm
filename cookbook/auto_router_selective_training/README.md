# Selective routing trained on clean paired attempts

These optional profiles learn when GPT-5.6 Sol adds a successful solve over GPT-5.6 Luna. They use one existing classifier response and a deterministic local prediction, with no second classifier call. They replace the earlier ROI cookbook, whose benchmark results were invalidated and are not the source of these fits

Training uses 160 fresh paired tasks and selection uses 64 separate validation tasks across SWE-bench, Terminal-Bench, MBPP+ and LiveCodeBench. The fitted policies were frozen before the 125-task held-out evaluation. Held-out results are still in progress

`strong-success-retention` minimizes validation cost while preserving every Sol-only success and meeting Sol quality within each benchmark. `quality-first-under-sol-budget` maximizes validation solves under Sol's total inference cost and permits task swaps. These names describe validation objectives, not guarantees on new requests

The selected upfront profiles use the original qualitative cards. The empirical card variation and all validation selections remain available in the accompanying records. Post-attempt review cascades are evaluated separately because they require a completed Luna attempt, verification and potentially a fresh Sol attempt

Set `ROI_GATEWAY_BASE_URL`, `ROI_GATEWAY_API_KEY` and `ROI_LOCAL_MASTER_KEY`, then start this branch with `litellm --config cookbook/auto_router_selective_training/proxy.yaml`. Use the model alias `selective-v3-strong-success-retention` or `selective-v3-quality-first-under-sol-budget`

The example config describes the SWE-bench agent harness. Match its execution conditions, solver endpoints, efforts and judge prompt to your benchmark. Route once on the opening task and pin the selected model for a comparable whole-task run. Independent per-turn routing or continuing Sol from a Luna patch requires a separate evaluation

`selective_policy` replaces the existing threshold decision and cannot be combined with the older probability calibration. Its `target` identifies the score: `scalar_calibration` and `per_model` estimate a success-probability difference, `paired` estimates Sol-only probability minus Luna-only probability, `rescue` estimates Sol-only probability, and `benefit_per_dollar` divides the paired difference by predicted incremental inference cost. Threshold units depend on this target. Raw classifier probabilities remain available in diagnostics

Defaults are unchanged when `selective_policy` is omitted. The local implementation validates coefficient dimensions and rejects another classifier's feature schema. Runtime parity checks compare scores and routing choices with the training implementation
