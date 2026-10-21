# Held-out capability routing pilot

## Scope and approval gate

This pilot compares a training-only spend-per-success capability map with the current Jev complexity router on five held-out SWE-bench tasks. It routes once per task for each arm. The published Terminal-Bench experiment switched models per turn, so its results are not directly comparable to this task-level routing design. If both arms select the same model, they still run separately in fresh containers with no shared patch or state.

No paid classifier or solver calls are allowed until design approval is relayed. The runner's live path also requires `ROUTING_POC_HELDOUT_DESIGN_APPROVED=approved`; `--dry-run` reads frozen local artifacts and makes no network, model, or solver calls.

## Frozen data and provenance

The held-out selection continues the seed rule `sha256("litellm-capability-routing-pilot-v1" + instance_id)` over Princeton SWE-bench dev at revision `e48e2bd1e9fecd5bbd641e9414ac59da9f2e69f6`, excludes the original five training tasks, and takes the next five candidates with public evaluation images. The held-out manifest SHA-256 is `c92cc0170b51f62326c1078cd981cdf0a8110261beeaa38a31be8a3fe04aa985`.

| Held-out task | Upstream registry digest | Local overlay image ID | Gold validation | Noop validation |
| --- | --- | --- | --- | --- |
| `pylint-dev__astroid-1164` | `sha256:2aaa44d8f0abfaa47e2b7b378b204e2341d7a98525888c58f64ec83fced446eb` | none | resolved | unresolved |
| `pylint-dev__astroid-2240` | `sha256:0bc107e0e3eedcf47322c8ed22ec961cf08b9b394688b0f1c86113378b4d768d` | none | resolved | unresolved |
| `pydicom__pydicom-793` | `sha256:9ad524560667adaa8367dad49c1dbad7aa9778c692c26c8c68edd507fe64f942` | `sha256:cf79289615c0f96fbb827e8b4843d8d45f9a6c82fefcb3a2d982473dac1f6041` | resolved with `C.UTF-8` locale | unresolved |
| `pvlib__pvlib-python-1191` | `sha256:0678cd6b58ff5b69756945921942444856a5038f82da465259d5a49bf95e7aad` | `sha256:c963b822ba3abd17a92d0160257a4b4e32b3067014b609d36da15b107107e6f3` | resolved with NumPy 1.26.4 | unresolved |
| `pylint-dev__astroid-927` | `sha256:9ca6c5ea2a9f260e64f7a2fca563fb60a6d2a58bff54f62a3fd3dc75937deba5` | none | resolved | unresolved |

The initial gold run completed all five tasks with three resolved and two unresolved. Review of the task logs showed environment-specific failures: pvlib test collection referenced `np.Inf`, removed in NumPy 2, and the pydicom UTF-8 filename test raised `UnicodeEncodeError` under the image's ASCII locale. The bounded overlays pin NumPy 1.26.4 for pvlib and set `LANG=C.UTF-8` and `LC_ALL=C.UTF-8` for pydicom. Both builds passed `pip check` with no broken requirements. The original upstream tags remain preserved as `routing_poc/orig.*`. `heldout_image_provenance.json` records registry digests separately from local overlay image IDs, while evaluation logs and gold patches remain local.

The official evaluator reran only the two affected tasks. Gold resolved both, with zero infrastructure failures, ambiguous failures, or errors. Noop left both unresolved with the same zero-failure counts. Combined with the three tasks resolved in the initial run, gold resolved all five. Noop left all five unresolved. The unchanged overlay images are used for both eventual routing arms, so the pydicom locale setting applies equally to both solvers. No task IDs, evaluation scripts, or test selections changed.

Evaluator datasets, predictions, logs, and outputs stay in the local `eval_heldout/` directory and are excluded from the cookbook branch because evaluator records may include gold patches. Recreate the pinned local dataset and noop control with `python -m poc.prepare_heldout --eval-only`; validation outputs are not copied to the source branch.

All validation used `SWE-bench/SWE-bench` dev at revision `c6fe717fd7a4c3ac1daa4055a4fd082c6a1d28a2`. The pinned dataset snapshot SHA-256 is `64b63a2804399e8e233909bba659c27abc74054c32c8f2a362dd5d82a585f07c`; the noop prediction SHA-256 is `e8e68b1ffbff54ca49310c274bb5a5e9b0b90dc4f059bcd7ae127ac6fe54d75c`.

## Capability scoring

The map is built read-only from the original five training records in `runs.jsonl`. A model's score for a category is its total spend, including failed attempts, divided by its graded successes, without smoothing. A zero-success cell has no finite score and is never selected. Fewer than three graded tasks sets the sparse flag but does not block a model. Unknown categories, classifier failures, and categories with no successful model fall back to the global scores. If every global model has zero successes, the decision is `unproven` and no solver runs.

The frozen map SHA-256 is `d5ecdba67cecc7d3bb8209b87b4c19cea6e4db54b9c7c5426739dd26ce1801a1`. Repository debugging scores are Haiku `$0.73440015`, Sonnet `$0.78796515`, and Opus `$2.12507325` per success. Feature implementation has zero successes for all three models and uses global fallback. Global scores are Haiku `$0.9589136500000001`, Sonnet `$0.99004815`, and Opus `$2.715796` per success. The map selects Haiku for every currently evidenced category.

## Jev baseline

The baseline uses the pinned LiteLLM complexity-router tree commit `b88f64e65f29a34ee7363a3e2cd8189e850e4387` from repository revision `35e36f3d477121909af0d83682578010f458d8be`. It instantiates `ComplexityRouter` and `ComplexityRouterConfig` with `classifier_type="oss_classifier"`, source default rubric and tiers, and `typesafe/jev-latest`. The injected client forwards the source-built request body to gateway `/v1/systemone`, uses a 120-second timeout and zero retries, and parses `JevSystemOneResponse`. The frozen example request artifact and its digest are `heldout_baseline_prompt.json` and `7e94a5dc85706dc1e1a81d26ddf89afedac31f14c59b22fce70c531ce6f951d4`.

Active tiers come from `config.labeled_tiers()` in source order: SIMPLE maps to Haiku, MEDIUM to Sonnet, COMPLEX and REASONING to Opus. NON_REASONING is disabled by default. If enabled, it maps to Haiku. Unknown tiers and classifier failures use Sonnet with an explicit reason. Each arm independently classifies the sanitized problem statement only.

## Execution safeguards

Three task workers process the frozen schedule. Each task's two arms run sequentially in a deterministic balanced order derived from `litellm-capability-routing-heldout-v1`. Each solver starts in a fresh clean container. The run cap is `$20`, computed only from held-out cost files, with `$1` reserved for every in-flight classifier or solver call. Each solver retains the existing `$1` task limit, 50-step limit, 600-second timeout, setup-only retry, 50 GiB dispatch reserve, and 20 GiB emergency disk stop.

The task-owned key alias is `routing-poc-heldout-2026-10-21`, scoped to the three solver models and `typesafe/jev-latest`, with a `$20` maximum budget and seven-day duration. Its secret is stored outside source control in `.private/heldout_key` with mode `0600`. Creating the key does not authorize or perform a model call.

All new execution records and costs go to `heldout_runs.jsonl`, `heldout_raw/`, `heldout_labels.json`, and `heldout_costs_aux.jsonl`. Existing training evidence remains read-only.
