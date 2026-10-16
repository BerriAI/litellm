# Python and Rust drift audit design

Use a deterministic change inventory, an agent audit, and required Rust review together. A checkpoint should say which Python behavior has been accounted for through a commit within a named scope. It cannot currently claim that Rust implements everything in Python

This proposal covers the initial inventory and enforcement design requested after [PR #46790](https://github.com/BerriAI/litellm/pull/46790). That prototype generates shared types and predicates for two Anthropic beta features without connecting its detectors to production routing. It was closed without merging. Shared generation remains useful for bounded contracts, but cannot cover orchestration, dynamic provider selection, fallback, streaming lifecycle, or host behavior by itself

The repository findings below are pinned to main commit [`6530b329f43ef456c502ba2db7656f871d62474e`](https://github.com/BerriAI/litellm/tree/6530b329f43ef456c502ba2db7656f871d62474e). GitHub rules were inspected on October 16, 2026. This is a proposal, with no deployed check, checkpoint, ownership change, or repository settings change

## Current coverage boundaries

The [rollout catalog](../../litellm/rust_bridge/catalog.py) makes Messages eligible for Rust opt-in for Anthropic, DeepSeek, and Vertex AI, subject to configuration. Other Messages contexts and Responses remain Python-only. Native implementation support, Python rollout, and gateway exposure are separate facts

Native Messages provider selection in `crates/inference-messages/src/common_utils.rs` covers more providers than the rollout catalog. Native Responses direct HTTP supports OpenAI and OpenRouter, but the Python binding rejects HTTP streaming in `crates/python-bridge/src/routes/responses/host.rs`. The gateway can use core Responses SSE. HTTP create, WebSocket sessions, and Python get, delete, list-input-items, cancel, and compact operations need separate scopes

The [provider rules](../crates/llms/AGENTS.md) explicitly record partial ports, including Responses model-specific parameter rewriting and Responses-to-Chat emulation. They also record deliberate strict SSE decoding, which differs from Python's handling of malformed events. These require named exceptions or gap records rather than a blanket parity assertion

The [inference rules](../crates/inference/AGENTS.md) keep accounting in Python on the legacy callback adapter path and give gateway accounting separate ownership. The [native router](../crates/router/README.md) is an exact deployment lookup scaffold, with retries, cooldowns, fallbacks, strategies, and multiple deployments unsupported. A native endpoint alone does not establish equivalent routing or accounting

## Initial audit inventory

Paths beginning with `crates/` below are relative to `litellm-rust/`. Each row is an audit family, split further by provider, operation, surface, and streaming mode when behavior differs

| Audit family | Python source | Rust implementation or boundary | Behavior to account for |
| --- | --- | --- | --- |
| Messages entry | `litellm/messages/dispatch.py`, `litellm/rust_bridge/messages/` | `crates/python-bridge/src/routes/messages/` | Public signatures, sync and async calls, bypass and admission |
| Messages orchestration | `litellm/llms/anthropic/pass_through/messages/handler.py` | `crates/inference-messages/src/` | Provider selection, preparation order, credentials, headers, hooks, cache, lifecycle |
| Messages transformations | `litellm/llms/base_llm/anthropic_messages/`, `litellm/llms/anthropic/pass_through/messages/transformation.py`, other provider Messages configs | `crates/llms/src/base_llm/messages/`, `crates/llms/src/<provider>/messages/` | Normalization, beta policy, tools, thinking, system messages, supported parameters |
| Messages payloads | `litellm/types/llms/anthropic_messages/` | `crates/llms-types/src/formats/messages/` | Accepted fields, discriminators, unknown fields, serialization, usage |
| Responses entry and orchestration | `litellm/responses/{dispatch,main}.py`, `litellm/rust_bridge/responses/` | `crates/python-bridge/src/routes/responses/`, `crates/inference-responses/src/` | Create versus other operations, provider selection, HTTP versus WebSocket, admission |
| Responses transformations | `litellm/llms/base_llm/responses/`, `litellm/llms/<provider>/responses/`, `litellm/completion_extras/litellm_responses_transformation/` | `crates/llms/src/base_llm/responses/`, `crates/llms/src/{openai,openrouter}/responses/` | Parameter rewriting, identifiers, tools, emulation, provider policy |
| Responses payloads | `litellm/types/llms/openai.py`, `litellm/types/responses/` | `crates/llms-types/src/formats/responses/` | Input and output items, event shapes, extra fields, usage |
| Streams and recovery | Messages `streaming_iterator.py`, `fake_stream_iterator.py`, `agentic_streaming_iterator.py`, Responses `streaming_iterator.py`, `sse_output_recovery.py` | `crates/llms/src/base_llm/{messages,responses}/streaming.rs`, `crates/llms/src/base_llm/sse.rs`, `crates/framer/`, inference drivers | Framing, terminal events, errors, buffering, partial output, cancellation, cleanup |
| Provider registry and capability data | `litellm/utils.py`, `litellm/litellm_core_utils/get_llm_provider_logic.py`, `litellm/llms/openai_like/`, model catalog | `crates/core-utils/`, `crates/inference/src/provider.rs`, route preparation, Python capability host | Dynamic providers, aliases, model capabilities, auth and endpoint selection |
| Shared shaping and transport | Prompt templates, timeout resolver, provider headers, `litellm/llms/custom_httpx/`, provider common utilities | `crates/core-utils/`, `crates/http/`, provider configs, Python host bindings | Shared helper changes, precedence, timeouts, request headers, HTTP errors |
| Public errors and metadata | `litellm/exceptions.py`, exception mapping utilities, `llm_response_utils/`, `litellm/anthropic_interface/`, `litellm/rust_bridge/failures.py` | Core error mapping, `crates/llms/src/error.rs`, bridge errors, gateway error projection | Exception class, status, body, request IDs, response headers, hidden metadata |
| Rollout and delegated execution | `litellm/rust_bridge/`, `litellm/utils.py` client decorators | `crates/python-bridge/`, `crates/host-python/`, `crates/callbacks-legacy-python/` | Fallback before execution, no fallback after execution starts, caller task identity, Python-owned state |
| Proxy and routing | `litellm/proxy/anthropic_endpoints/`, `litellm/proxy/response_api_endpoints/`, `litellm/router.py` | `crates/gateway-inference/`, `crates/router/`, `crates/router-types/` | HTTP contracts, retries, deployment selection, fallback, budget and auth integration |
| Cache, callbacks and accounting | `litellm/caching/`, logging utilities, bridge cache and legacy callback hosts | `crates/cache-response/`, inference caching, bridge cache, legacy callbacks, gateway dependencies | Cache identity and replay, callback order, usage, charges, terminal delivery |

Keep feature scopes for safeguards, advisor hooks, MCP, context management, compaction, file-search emulation, sessions, custom tools, and tool search explicit. An audit must establish whether each feature is native, delegated, falling back, unsupported, or an intentional difference before closing its scope

### Source selection

Start with broad candidate discovery over all tracked files under `litellm/`, relevant tests, all `litellm-rust/`, the model catalog and backup, provider-owned JSON configuration, and dependency manifests and locks. This catches new folders and dependencies outside known endpoint paths. File selection finds candidates, while symbol and call-chain review decides relevance

Direct Messages and Responses selectors should include `litellm/messages/**`, `litellm/anthropic_interface/**`, `litellm/llms/anthropic/pass_through/**`, `litellm/llms/base_llm/anthropic_messages/**`, `litellm/types/llms/anthropic_messages/**`, `litellm/responses/**`, `litellm/types/responses/**`, `litellm/llms/base_llm/responses/**`, `litellm/llms/*/responses/**`, and `litellm/llms/*/messages/**`

Provider layouts also require `litellm/llms/azure_ai/anthropic/**`, `litellm/llms/bedrock/claude_platform/**`, `litellm/llms/vertex_ai/vertex_ai_partner_models/anthropic/**`, provider-wide utilities, and `litellm/llms/openai_like/**`. Include the bridge, response adapters, proxy endpoint folders, `litellm/main.py`, `litellm/__init__.py`, `litellm/utils.py`, `litellm/router.py`, types, exceptions, caching, and shared core utilities as dependencies

In particular, watch prompt templates, provider resolution, header extraction, exception mapping, logging, settings, timeouts, `litellm/anthropic_beta_headers_manager.py`, and `litellm/anthropic_beta_headers_config.json`. Catalog edits require capability triage, not an automatic obligation to port unrelated model entries

Keep direct implementations, shared dependencies, and tests separate in the manifest so an audit can explain why a change affects a scope. Import analysis and trace observations can suggest missing dependencies, but dynamic imports, provider registries, and Python calls by string name make them insufficient as the sole selector. Unclassified candidate paths require explicit scope classification rather than silently passing

## Checkpoint and report contract

Use one versioned manifest at a proposed path such as `litellm-rust/python_parity.json`, with stable entries such as `messages.sdk.anthropic.create.nonstream` and `responses.gateway.openai.create.sse`. These examples name scope, without asserting current parity

| Field | Required meaning |
| --- | --- |
| `id`, `surface`, `operation`, `provider`, `mode` | Exact claim boundary, including sync or async differences where relevant |
| `python_paths`, `python_symbols`, `dependency_paths`, `rust_paths` | Source and implementation selectors, allowing many-to-many mapping |
| `support`, `rollout`, `delegation` | Native capability, admission predicates, rollout policy, and retained Python responsibility, independently recorded |
| `parity_baseline_sha` | Optional full landed default-branch SHA through which the scope has no unresolved parity obligations except explicitly approved differences |
| `reviewed_through_sha` | Optional separate audit cursor, permitted to advance with open work only when every outstanding obligation remains in the gap ledger |
| `rust_evidence_sha`, `evidence` | Existing immutable implementation revision and behavior evidence supporting the checkpoint |
| `gaps`, `intentional_differences` | Stable IDs, affected behavior, reason, owner role, evidence, and follow-up reference where one exists |

Unassessed scopes start without a parity baseline. Do not initialize every entry to the current main SHA. Establish each initial baseline by auditing its bounded behavior, including inherited behavior and existing gaps. A reviewed partial port remains partial

Advance a parity baseline only after every intervening behavior change has a resolved disposition and behavioral evidence. Retain intentional differences explicitly and recheck them when their source changes. An unsupported capability is an open migration gap, even if fallback keeps the public SDK working

For the initial implementation, advance baselines to landed default-branch commits in a follow-up checkpoint commit. Same-PR Rust compensation can close a PR obligation using head-specific evidence without advancing the baseline immediately. This avoids a commit containing its own SHA and avoids checkpoints pointing at pre-squash commits that disappear from default-branch ancestry

Evidence may reference an existing implementation commit preceding the checkpoint. Verify that watched source and implementation blobs at the tested revision match those at the checkpoint tip. A later Rust-only edit must trigger a new audit even if the Python baseline did not move

The JSON report and a short Markdown summary must bind full baseline, target, PR head, implementation, policy, and manifest revisions. For each changed behavior, include source path and symbol, hunk or blob reference, affected scopes, dependency reason, observable behavior, final implementation or fallback path, disposition, evidence, and remaining gap

Use dispositions `port_required`, `already_handled`, `shared_python`, `explicit_fallback`, `intentional_difference`, `no_runtime_effect`, and `unresolved`. The agent proposes these classifications. Rust review accepts or rejects them. A deterministic check verifies completeness and revision binding, but cannot prove that a human or model's behavioral conclusion is correct

Reports should show open behavioral obligations and newly introduced obligations separately. Count classified scopes and behavior changes, with explicit unknowns. Lines changed and percentages of matching functions do not measure feature parity

## Deterministic inventory and agent audit

Resolve immutable commit objects first. For each scope, let `B` be its parity baseline and `T` the target default-branch SHA. For a PR, let `H` be the submitted head and `M` the merge base with the target. Missing objects, shallow history, invalid ancestry, or an unaudited baseline produce an explicit error or unassessed result, never an empty successful audit

Use endpoint diffs for cumulative drift and merge-base diffs for the PR delta. The following commands describe the inventory stage, with variables supplied from trusted event metadata and the validated manifest

```sh
git cat-file -e "$B^{commit}"
git cat-file -e "$T^{commit}"
git merge-base --is-ancestor "$B" "$T"
git diff --no-ext-diff --no-textconv --find-renames=50% --name-status -z "$B" "$T"
M=$(git merge-base "$T" "$H")
git diff --no-ext-diff --no-textconv --find-renames=50% --name-status -z "$M" "$H"
```

Parse the NUL-delimited inventory before applying selectors. Match both old and new names against the union of trusted base policy and proposed policy. Additions, removals, renames out of scope, and scope deletions must remain visible. Do not let a PR shrink its own watch set to obtain a passing check

Use the full source diff and both snapshots for each affected scope, then inspect callers, dependencies, registrations, admission, fallback, public projection, and tests. Commit history explains intermediate work and reverts. The net endpoint diff determines current behavior changes

```sh
git log --format='%H %s' "$B..$T" -- <affected-paths>
git diff --no-ext-diff --no-textconv --find-renames=50% "$B" "$T" -- <affected-paths>
git show "$B:<path>"
git show "$T:<path>"
```

Inspect Rust and bridge changes against their recorded implementation revision as well. A Python edit can already flow through shared Python, be covered by an existing Rust implementation, or have compensating Rust changes in the same PR. Conversely, a Rust-only edit can break parity

Give the agent the validated inventory and relevant snapshots, with repository text treated as untrusted input. It should identify changed behavior, trace its final execution path per surface, propose the smallest behavioral test, and emit the bound report. It must not advance checkpoints, approve its own findings, or turn missing evidence into success

When `T` moves, rerun cumulative inventory against the new target and require behavior evidence from the actual combined target and PR merge result. Tests from unchanged `H` alone cannot cover those interactions. If a merge queue is enabled, use its actual merge-group SHA. A report for an old head or policy revision cannot satisfy the current check

## Existing tests and evidence

The [parity harness](../../tests/rust-python-harness/AGENTS.md) already defines SDK and gateway surfaces, complete strategy classification, comparisons of requests, public objects, errors, streams, and callbacks, and recording or replay helpers. Reuse those concepts and fixtures rather than creating another parity framework

Its current rules explicitly require local execution with no CI integration. `shared/unit_runners/contracts.py` has no registered unit contracts at this snapshot, and `strategies/e2e_parity/__init__.py` has no implemented Messages or Responses cases. `trace_parity` collects Python calls rather than Rust spans. These help discover scope but do not establish a parity baseline. Keep the first CI inventory checker independent of this local harness

The [Rust workflow](../../.github/workflows/test-rust.yml) runs formatting, Clippy, Cargo tests, generated trace checks, and fresh wheel tests. Its manually selected Python trigger paths omit most provider transformations and `litellm/responses/streaming_iterator.py`. Passing Rust CI does not establish that relevant Python changes were noticed

Behavior evidence must include the actual command, exit status, case identifiers, tested revision, wheel or native build identity, artifact location, and positive evidence that the native branch executed. Two successful Python and Rust test runs are insufficient when one falls back, both skip, or neither asserts the changed behavior

Use focused deterministic regression cases for request shaping, accepted fields, errors, lifecycle, callback ordering, cancellation, and stream events. Verify each affected SDK and gateway surface. A provider or live proxy check belongs with changes that require it, and any skips or unavailable credentials remain visible in the report. Do not present test commands as customer-facing proof of fix

## Required Rust review

[CODEOWNERS](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners) routes reviews and can require one listed owner when repository rules enable owner approval. Multiple owners on one pattern are alternatives. The last matching pattern wins, so separate overlapping Python and Rust lines do not require both approvals

At this snapshot, [`.github/CODEOWNERS`](../../.github/CODEOWNERS) owns only the two cost-map JSON files. [guard-main](https://github.com/BerriAI/litellm/rules/17360296) requires code-owner approval, but has stale approval dismissal and last-push approval disabled. [main-merge](https://github.com/BerriAI/litellm/rules/16489521) requires one approval and squash merging

Use GitHub's native [required reviewers](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets#required-reviewers) rule to require one Rust team approval for tracked Python changes. This repository already uses it in [Protect Sensitive Files](https://github.com/BerriAI/litellm/rules/14572570), with `required_reviewers` inside the `pull_request` rule's parameters. That rule requires team ID `16890674`, verified as `@BerriAI/litellm-engineering`, for `.github/**` and selected dependency and migration files

The required reviewer entry uses `minimum_approvals`, `file_patterns`, and `reviewer` with a numeric `id` and `type` of `Team`, as exposed by the [REST rules schema](https://docs.github.com/en/rest/repos/rules). Add an entry for the verified Rust team, and a second entry for the Python team if approval from both teams is required. Overlapping team memberships do not guarantee two distinct human reviewers

The visible team listing did not identify a Rust migration team. Before deployment, maintainers must identify or create the intended team, verify its membership and repository write access, and confirm whether engineering is the intended Python owner. Do not deploy a placeholder slug or numeric ID

For the pilot, require Rust review on the broad candidate Python paths, watched Rust and bridge implementation paths, relevant owned data and dependency files, and drift policy, checker, report contract, and workflow changes. Narrow the patterns only after the inventory supports that decision. Use one canonical selector definition to generate or validate ownership patterns and ruleset patterns, while testing each system's different glob semantics

If adding CODEOWNERS entries for automatic review requests, place broad entries first, narrower entries afterward, and preserve the current cost-map owners on the final matching entries. List every intended owner on the same line. Protect the CODEOWNERS file and drift infrastructure too. Keep native team approval as the enforcement mechanism

Enable stale approval dismissal for the governing pull-request rule before treating Rust approval as coverage of later pushes. Last-push approval alone can leave an older Rust approval in place while another reviewer approves new changes. Validate the effective combined rules and normal bypass behavior with representative PRs before rollout

## CI boundary and rollout

Run a lightweight inventory job on every PR, with an explicit unaffected result when there are no candidates. Workflow-level path filtering can leave a required check pending. Use read-only permissions for executing PR code. An agent job with credentials must operate on trusted base code and treat fork files as data, without installing or executing PR-controlled code. The [cost-map guard](../../.github/workflows/cost-map-guard.yml) provides a local example of separating privileged metadata inspection from untrusted changes

Bind published results to the current PR head and recheck the head before publishing. Handle new pushes and base changes, and add `merge_group` if merge queues are introduced. Missing agent output is an unavailable audit, never a parity pass. Keep model findings advisory and make the deterministic gate responsible for valid checkpoints, complete classifications, retained gaps, and matching evidence

First land the manifest schema, source inventory checker, report schema, and behavior tests for additions, deletions, renames, shared dependencies, new scopes, invalid SHAs, stale reports, and attempts to remove watched paths. Start unassessed and report-only. Inventory one small Messages provider scope and one Responses surface before claiming an initial baseline

Then establish the actual Rust team and configure native approval rules and CODEOWNERS routing. Validate Python-only, Rust-only, compensated, fallback, manifest-only, and fork PRs. After the pilot, require the deterministic check and extend scopes incrementally. The deployment step must verify live ruleset settings and current-head bot and CI findings rather than treating missing or pending results as approval

## Worked change inventory

Commit [`a3251a3bfde17fb1c88896a1f2f3b7f9d2ef0345`](https://github.com/BerriAI/litellm/commit/a3251a3bfde17fb1c88896a1f2f3b7f9d2ef0345) preserves provider request IDs in Responses streaming failure logs. It changes `litellm/responses/streaming_iterator.py`, shared `litellm/litellm_core_utils/llm_response_utils/get_headers.py`, Bedrock event decoding, and regression tests, with no Rust edits

```sh
git diff --no-ext-diff --no-textconv --name-status a3251a3bfde17fb1c88896a1f2f3b7f9d2ef0345^ a3251a3bfde17fb1c88896a1f2f3b7f9d2ef0345
git show a3251a3bfde17fb1c88896a1f2f3b7f9d2ef0345 -- litellm/responses/streaming_iterator.py litellm/litellm_core_utils/llm_response_utils/get_headers.py
```

The observable change is that stream failure exceptions retain provider request-ID headers. A Responses-folder-only watch set misses the shared helper and Bedrock decoder. Requiring a Rust edit for every Python edit also misclassifies the current SDK path, where Responses dispatches directly to Python under the rollout catalog and inherits the fix

Record the current SDK result as Python-owned behavior with dispatch and execution evidence, leaving native SDK streaming unsupported. Native admission also declines HTTP streaming, but proving that fallback path requires a separate test that overrides rollout. Separately inspect gateway error presentation and failure logging because the gateway uses native Responses SSE. The absence of a Rust edit is a candidate obligation, not proof of a gateway defect

A focused replay case would return HTTP SSE with a provider request-ID header, followed by a failure event. Compare the public error, status, and failure metadata appropriate to each surface, and prove which implementation executed. Until that trace and behavior evidence exist, keep the gateway disposition unresolved
