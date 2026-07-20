# Governance Dict and Model-Level Guardrail Merge Regression

> Status: implemented and verified
> Date: 2026-07-20

## Current State / Source Audit

The owning repository is `litellm-custom`. Work started from a clean checkout on
`litellm_guardrail_dict_merge`, branched directly from the refreshed
`upstream/litellm_internal_staging` commit
`bd44c9e305b89526d4c5d773ee39ca935561b9c8`. The neighboring
`litellm-governance` checkout is on
`agent/external-membership-sync-standalone-cronjob-analysis` and contains unrelated
user changes. Those files are outside this change and must remain untouched.

The current request path is:

1. Governance's `litellm-platform-adapter/.../callback.py::_inject_application_guardrails`
   writes an enable/disable map such as
   `{"input_token_limit": true, "min_version": false}` to
   `data["metadata"]["guardrails"]`.
2. Router selection adds the deployment ID to `metadata.model_info.id` and exposes
   the selected deployment through `Router.get_deployment()`.
3. `litellm/proxy/utils.py::_check_and_merge_model_level_guardrails()` reads the
   selected deployment's `litellm_params.guardrails`. It returns early only when
   that setting is absent.
4. `litellm/proxy/utils.py::_merge_guardrails_with_existing()` currently assumes
   the request representation is a list. A mapping is wrapped as the only list
   element and then passed through `set()`.
5. `litellm/integrations/custom_guardrail.py::get_guardrail_from_metadata()` and
   `_guardrail_is_in_requested_guardrails()` consume guardrail names from request
   metadata. The governance mapping must therefore remain a mapping so its
   explicit enabled/disabled state is not erased.

The pre-fix diagnostic against the base commit used the public merge entry point,
a dependency-injected router, and a deployment with
`guardrails: [input_token_limit]`. It failed at `utils.py::_merge_guardrails_with_existing()`
with `TypeError: unhashable type: 'dict'`, confirming the supplied root-cause report
against current source.

The completed lifecycle design on `feat/custom-v1.90.3`,
`litellm-platform-adapter/docs/platform/40-implementation/103-input-token-limit-deployment-lifecycle.md`,
establishes that deployment attachment is intentional: the selected deployment's
root `guardrails` binding activates the deployment pre-call hook after router
selection. Removing that attachment would disable enforcement and is not a fix.

## Observed Symptoms

With governance enabled, adding the following selected-deployment binding causes
`POST /v1/chat/completions` to return HTTP 500:

```yaml
model_list:
  - model_name: governed-model
    litellm_params:
      model: openai/provider-model
      guardrails:
        - input_token_limit
```

The request has already received governance metadata equivalent to:

```json
{
  "metadata": {
    "model_info": {"id": "selected-deployment-id"},
    "guardrails": {
      "input_token_limit": true,
      "min_version": false
    }
  }
}
```

The response is:

```json
{"error":{"message":"unhashable type: 'dict'","type":"None","param":"None","code":"500"}}
```

Without deployment-level `litellm_params.guardrails`, the merge helper returns
early, so there is no crash but the deployment-scoped input-token enforcement does
not run.

## Expected Behavior

The merge helper must support both established metadata representations:

- For a governance mapping, preserve every existing key/value and overlay each
  model-level guardrail name with `true`.
- For the vanilla list representation, preserve first-seen order while adding
  model-level names once.
- Preserve the existing scalar compatibility path by normalizing a truthy scalar
  to one list element. This avoids an unrelated behavior change outside the
  governance regression.
- Do not catch or suppress the exception at a caller. The representation mismatch
  must be resolved at the shared merge boundary.

The resulting governance example must be:

```json
{
  "input_token_limit": true,
  "min_version": false
}
```

The deployment overlay wins if a matching existing entry is `false`, because an
explicit model-level binding activates that guardrail for the selected deployment.

## Root Cause Analysis

`_merge_guardrails_with_existing()` treats every non-list value as a scalar. A
governance mapping is therefore transformed from a semantic map into
`[{"input_token_limit": true, "min_version": false}]`. The final expression
`list(set(existing_guardrails + model_level_guardrails))` attempts to hash that
mapping and raises `TypeError` before provider execution or guardrail enforcement.

Evidence that this is the defect rather than the deployment attachment:

- The exception is reproduced directly at the shared merge entry point on the
  current staging commit.
- The merge path is entered only after a selected deployment advertises
  `litellm_params.guardrails`.
- The lifecycle contract requires that advertisement so the deployment pre-call
  hook runs with selected model information.
- The governance producer intentionally includes `false` values to disable
  otherwise global/default-on guardrails; converting the map to a name list would
  lose that contract.

Rejected directions:

- Change governance to emit a list: this erases explicit `false` state and affects
  all mapping consumers.
- Remove `litellm_params.guardrails`: this avoids the merge only by disabling the
  required deployment hook.
- Catch `TypeError`: this hides the request-shape incompatibility and either loses
  model-level activation or fails open.
- Convert mapping keys to a list: this would treat disabled keys as enabled and
  discard their boolean values.
- Keep using `set()` only for lists: it produces nondeterministic order and is not
  needed for stable deduplication.

## Affected Surface Matrix

| Existing metadata | Model-level value | Expected result |
| --- | --- | --- |
| governance map with enabled and disabled keys | list with an existing enabled key | map preserved; no exception; disabled keys remain false |
| governance map with a false key matching the deployment | list containing that key | matching value becomes true; unrelated values remain unchanged |
| governance map | scalar guardrail name | map preserved; scalar key becomes true |
| vanilla list with duplicates | list with duplicates | first-seen ordered unique list |
| vanilla list | scalar guardrail name | first-seen ordered unique list |
| truthy scalar existing value | list | existing scalar compatibility remains a two-entry list |
| empty or missing metadata guardrails | empty or missing model value | empty list on direct merge; check helper still returns early when deployment value is `None` |
| deployment missing or selected model ID missing | any metadata shape | original data object returned unchanged |
| list containing an unhashable dynamic-configuration object | list | outside this mapping contract; existing error behavior is not broadened in this change |

## External Contracts

No public endpoint, database row, Redis key, environment variable, authentication
token, policy snapshot schema, Helm value, or migration changes.

The request-side contracts retained are:

- `metadata.guardrails` may be a vanilla list or a governance boolean mapping.
- `model_list[].litellm_params.guardrails` may be a list or scalar and is read from
  the selected deployment.
- `metadata.model_info.id` identifies the selected deployment used for the lookup.
- Mapping values are explicit enable/disable decisions; model-level attachment
  overlays named entries to `true`.
- List output order is stable and follows existing request entries before newly
  attached model entries.

## Implementation Plan

1. Add a focused regression to
   `tests/test_litellm/proxy/utils/helpers/test_guardrail_merge.py` through
   `_check_and_merge_model_level_guardrails()` with an injected router/deployment.
   Assert that the original governance mapping type, its `false` entry, and the
   enabled deployment key survive. Confirm it fails on the base implementation
   with the reported `TypeError`.
2. Keep the existing direct test for a list containing an unhashable mapping.
   That shape is distinct from a governance mapping stored directly at
   `metadata.guardrails` and remains outside this change.
3. Update `litellm/proxy/utils.py::_merge_guardrails_with_existing()`:
   normalize model-level values, branch on an existing mapping, preserve and
   overlay it with enabled names, and use `dict.fromkeys()` for ordered list
   deduplication. Keep truthy scalar existing values compatible.
4. Add the required minimal `AAP Feature / CUSTOM FORK: litellm-governance` marker
   only at the non-obvious mapping branch. Do not add broad comments or exception
   handling.
5. Format only the modified Python files, run focused tests and static checks, then
   run `make platform-verify` as required by repository instructions.
6. Start a local proxy with a governance-enabled reproduction config and a real
   provider. Capture the same curl before and after the source fix, an over-limit
   400, a normal 200, and the `AAP context token limit` execution log. Keep any
   local config/log artifacts untracked and out of commits.

No migration, backfill, generated client, cache invalidation, or deployment-order
change is required.

## Validation Plan

Focused pre-fix and post-fix regression:

```bash
uv run pytest \
  tests/test_litellm/proxy/utils/helpers/test_guardrail_merge.py \
  -q
```

Formatting and static validation:

```bash
uv run black \
  litellm/proxy/utils.py \
  tests/test_litellm/proxy/utils/helpers/test_guardrail_merge.py
uv run ruff check \
  litellm/proxy/utils.py \
  tests/test_litellm/proxy/utils/helpers/test_guardrail_merge.py
make platform-verify
```

If the touched files affect `ruff-strict-budget.json` or
`basedpyright-code-budget.json`, run `make lint-budget-update` and commit only a
lower baseline. Do not use mutable-budget exemptions.

Live proof uses the same reproduction config and request body on both revisions:

```bash
python litellm/proxy/proxy_cli.py \
  --config <local-governance-reproduction.yaml> \
  --detailed_debug --reload --use_v2_migration_resolver \
  2>&1 | tee <local-litellm.log>

curl -sS -i http://127.0.0.1:4000/v1/chat/completions \
  -H 'Authorization: Bearer <local-master-key>' \
  -H 'Content-Type: application/json' \
  -d @<normal-request.json>

curl -sS -i http://127.0.0.1:4000/v1/chat/completions \
  -H 'Authorization: Bearer <local-master-key>' \
  -H 'Content-Type: application/json' \
  -d @<over-limit-request.json>

grep -i "AAP context token limit" <local-litellm.log>
```

Acceptance evidence is the pre-fix 500, post-fix normal 200, post-fix over-limit
`400 input_token_limit_exceeded`, and execution log from the real proxy/provider
path. Pytest output is supporting regression evidence, not the live proof.

## Thought Experiments / Failure Modes

| Scenario | Result and safety property |
| --- | --- |
| stale governance snapshot contains an old disabled key | merge preserves the false key; this helper does not invent or delete snapshot state |
| selected deployment adds a guardrail absent from the map | one new true key is overlaid; all other decisions remain intact |
| selected deployment explicitly attaches a currently false key | deployment attachment wins for that selected deployment and enables the hook |
| disabled or deleted policy row disappears in a new snapshot | producer changes the next map; merge remains generation-agnostic and preserves the supplied snapshot |
| missing `metadata.guardrails` | vanilla empty-list path remains valid |
| missing deployment guardrail setting | check helper returns before merge, preserving existing request data |
| retry or fallback chooses another deployment | each selected deployment overlays its own advertised names without making mappings hashable or flattening state |
| partial rollout: old proxy with governance map and attached deployment | known fail-closed 500 remains until the proxy fix is rolled out; producer contract is unchanged |
| partial rollout: fixed proxy with existing governance adapter | map contract is backward compatible and requires no adapter coordination |
| rollback of this fix | restores the 500 whenever both governance mappings and model-level attachment are present; emergency operational mitigation would have to remove the binding and would also remove enforcement |
| duplicate list values | `dict.fromkeys()` retains first occurrence and deterministic order |
| non-string model-level item | map overlay requires hashable names as before; supported configuration remains guardrail names, not nested objects |
| hash or generation mismatch outside request metadata | no effect on this pure merge; snapshot/auth layers retain ownership of generation validation |

The mapping branch creates a new merged mapping and does not mutate the mapping
object supplied by governance. The helper retains its existing shallow-copy
semantics for the outer request and metadata container; changing copy depth is
outside this regression.

## Progress Status

| Work item | Status |
| --- | --- |
| repository, branch, and dirty-state audit | Done |
| current source and lifecycle contract audit | Done |
| direct pre-fix TypeError reproduction | Done |
| design document | Done in this docs-only commit |
| thought-experiment gate | Done; retained the distinct list-containing-dict error contract |
| failing regression test | Done; failed pre-fix with `TypeError` and passed post-fix |
| implementation | Done; `e7e779c232` |
| focused/static/full validation | Done, except unavailable `platform-verify` target documented below |
| real-provider proxy proof | Done; pre-fix 500, post-fix 200/400 |
| implementation commit | Done; `e7e779c232` |
| pull request | Tracked externally after repository verification |

## Completion Evidence

### Regression and implementation

The focused regression was added through a real `Router` with a registered
deployment rather than a class-level monkeypatch. Before the source change:

```text
FAILED tests/test_litellm/proxy/utils/helpers/test_guardrail_merge.py::test_check_and_merge_model_level_guardrails_preserves_governance_dict
E   TypeError: unhashable type: 'dict'
1 failed, 1 warning in 4.73s
```

After the implementation and final Ruff formatting:

```text
.............................                                            [100%]
29 passed, 1 warning in 6.68s
```

The implementation is commit `e7e779c232` with message:

```text
fix(guardrails): handle governance dict when merging model-level guardrails
```

### Real-provider proxy proof

The live proof used the complete platform adapter from `feat/custom-v1.90.3` in
a temporary worktree and applied the same `utils.py` implementation hunk for the
post-fix run. The untracked config enabled platform governance and static service
account auth, configured local governance fallback with `min_version`, registered
both `min_version` and `input_token_limit`, and attached
`guardrails: [input_token_limit]` plus the exact bundled-tokenizer model info to the
deployment. This made governance inject a non-empty mapping before the selected
deployment merge. No mock provider or mock response was configured.

The upstream was the real GitHub Models OpenAI-compatible inference endpoint with
`openai/gpt-4.1-mini`. The token came from the signed-in GitHub CLI keyring and was
passed only through the process environment.

Both runs used:

```bash
DEBUG=false \
GITHUB_MODELS_TOKEN="$(gh auth token)" \
LITELLM_MASTER_KEY=sk-live-proof-master \
GOVERNANCE_ENABLE_TOKEN_BUDGET=false \
PYTHONPATH="$worktree:$worktree/litellm-platform-adapter" \
/Users/dhsshin/Documents/LLMOps/litellm-custom/.venv/bin/python \
  litellm/proxy/proxy_cli.py \
  --config live-proof-config.yaml \
  --detailed_debug --reload --use_v2_migration_resolver --port 4011 \
  2>&1 | tee live-proof-<pre-or-post>.log
```

The same normal request before the fix:

```bash
curl -sS -i http://127.0.0.1:4011/v1/chat/completions \
  -H 'Authorization: Bearer proof-service-token' \
  -H 'X-Application-Id: live-proof-app' \
  -H 'Content-Type: application/json' \
  -d '{"model":"governed-model","messages":[{"role":"user","content":"Reply with exactly LIVE_PROXY_OK"}],"max_tokens":8,"metadata":{"version":"1.2.3"}}'
```

```http
HTTP/1.1 500 Internal Server Error
content-type: application/json

{"error":{"message":"unhashable type: 'dict'","type":"None","param":"None","code":"500"}}
```

The pre-fix traceback identified the production path, not only the direct helper:

```text
File "litellm/proxy/utils.py", line 2459, in post_call_success_hook
  guardrail_data = _check_and_merge_model_level_guardrails(...)
File "litellm/proxy/utils.py", line 6106, in _check_and_merge_model_level_guardrails
  return _merge_guardrails_with_existing(data, model_level_guardrails)
File "litellm/proxy/utils.py", line 6135, in _merge_guardrails_with_existing
  metadata["guardrails"] = list(set(existing_guardrails + model_level_guardrails))
TypeError: unhashable type: 'dict'
```

The identical request after the fix reached the real provider and returned:

```http
HTTP/1.1 200 OK
content-type: application/json

{"id":"chatcmpl-E3ZJa7k0oJMrKTxtPIvxBhrmMddfi","created":1784519442,"model":"governed-model","object":"chat.completion","system_fingerprint":"fp_a89d652ac6","choices":[{"finish_reason":"stop","index":0,"message":{"content":"LIVE_PROXY_OK","role":"assistant","provider_specific_fields":{"refusal":null},"annotations":[]},"provider_specific_fields":{"content_filter_results":{"hate":{"filtered":false,"severity":"safe"},"protected_material_code":{"detected":false,"filtered":false},"protected_material_text":{"detected":false,"filtered":false},"self_harm":{"filtered":false,"severity":"safe"},"sexual":{"filtered":false,"severity":"safe"},"violence":{"filtered":false,"severity":"safe"}}}}],"usage":{"completion_tokens":4,"prompt_tokens":13,"total_tokens":17,"completion_tokens_details":{"accepted_prediction_tokens":0,"audio_tokens":0,"reasoning_tokens":0,"rejected_prediction_tokens":0},"prompt_tokens_details":{"audio_tokens":0,"cached_tokens":0},"latency_checkpoint":{"engine_tbt_ms":7,"engine_ttft_ms":44,"engine_ttlt_ms":73,"pre_inference_ms":181,"service_tbt_ms":13,"service_ttft_ms":501,"service_ttlt_ms":528,"total_duration_ms":368,"user_visible_ttft_ms":320}},"service_tier":"default","prompt_filter_results":[{"prompt_index":0,"content_filter_results":{"hate":{"filtered":false,"severity":"safe"},"jailbreak":{"detected":false,"filtered":false},"self_harm":{"filtered":false,"severity":"safe"},"sexual":{"filtered":false,"severity":"safe"},"violence":{"filtered":false,"severity":"safe"}}}]}
```

The post-fix over-limit request used 180 repeated `token` words:

```bash
long_prompt=$(printf 'token %.0s' {1..180})
request_body=$(jq -nc --arg content "$long_prompt" \
  '{model:"governed-model",messages:[{role:"user",content:$content}],max_tokens:8,metadata:{version:"1.2.3"}}')
curl -sS -i http://127.0.0.1:4011/v1/chat/completions \
  -H 'Authorization: Bearer proof-service-token' \
  -H 'X-Application-Id: live-proof-app' \
  -H 'Content-Type: application/json' \
  -d "$request_body"
```

```http
HTTP/1.1 400 Bad Request
content-type: application/json

{"error":{"message":"Input token count 193 plus reserved output tokens 8 exceeds the configured context limit of 128.","type":"guardrail_violation","param":"InputTokenLimitGuardrail","code":"400","provider_specific_fields":{"error":{"message":"Input token count 193 plus reserved output tokens 8 exceeds the configured context limit of 128.","type":"guardrail_violation","param":"InputTokenLimitGuardrail","code":"input_token_limit_exceeded"},"input_token_count":193,"output_token_reservation":8,"requested_context_tokens":201,"context_token_limit":128}}}
```

The required execution-log check produced both outcomes:

```bash
grep -i 'AAP context token limit' live-proof-post.log
```

```text
12:50:41 - LiteLLM Proxy:DEBUG - AAP context token limit passed: tokenizer=GaussO3.5 input=18 reserved_output=8 requested_context=26 limit=128
12:50:55 - LiteLLM Proxy:WARNING - AAP context token limit exceeded: tokenizer=GaussO3.5 input=193 reserved_output=8 requested_context=201 limit=128
```

`grep -i "unhashable type: 'dict'" live-proof-post.log` returned no output.

### Formatting, lint, and unavailable target

Final validation results:

```text
Ruff format check on both modified Python files: passed
Ruff check on both modified Python files: passed
Whole litellm Ruff check: passed
Strict Ruff budget gate: passed
Type-discipline LIT budget gate: passed
basedpyright budget gate: passed; no rule above the staging baseline
e2e basedpyright: 0 errors, 0 warnings, 0 notes
Circular import check: passed
Import safety check: passed
```

Neither `ruff-strict-budget.json` nor `basedpyright-code-budget.json` changed, so
`make lint-budget-update` was not required.

Repository instructions require `make platform-verify`, but the refreshed
`upstream/litellm_internal_staging` Makefile has no such target. The literal result
was:

```text
make: *** No rule to make target `platform-verify'.  Stop.
```

`make pre-commit` also could not fetch `litellm_internal_staging` from this clone's
fork `origin` and requested absent dashboard `node_modules` despite no UI change.
The equivalent Python lint targets were therefore run directly against a temporary
`origin/litellm_internal_staging` ref pointing at the already refreshed
`upstream/litellm_internal_staging`; all passed as listed above.
