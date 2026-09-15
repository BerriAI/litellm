# RCA on LiteLLM-internal spend tags leaking into Bedrock passthrough request bodies

Tracked in: [BerriAI/litellm#30629](https://github.com/BerriAI/litellm/issues/30629)

Introduced by: the tag-merge design in `litellm/proxy/litellm_pre_call_utils.py`, where key, team, project and header spend tags are merged into the request body's `metadata` object in place. The behavior is old; it became a hard failure when the Bedrock invoke/converse passthrough route began forwarding that same body verbatim to AWS.

Consequence: any virtual key (or `x-litellm-tags` header) that carries spend-tracking tags makes the Bedrock invoke/converse passthrough route 400 with `{"message":"metadata.tags: Extra inputs are not permitted"}`, but only when the caller's body already contains a `metadata` object. AWS Bedrock's Anthropic schema permits `metadata.user_id` and nothing else. Claude Code in Bedrock mode (`CLAUDE_CODE_USE_BEDROCK=1`) sends `metadata.user_id` on every request, so every request trips the bug and the proxy is unusable for that path.

Status as of 2026-06-22: open. The key-level fix is in flight as [#30985](https://github.com/BerriAI/litellm/pull/30985) (open). The `x-litellm-tags` header follow-up was [#30994](https://github.com/BerriAI/litellm/pull/30994) (closed, superseded once #30985 pre-seeds `litellm_metadata` before the header merge). The generalized, category-level hardening is tracked separately as [#30301](https://github.com/BerriAI/litellm/issues/30301).

## Root cause

One Python `dict` plays two incompatible roles on this path and nothing in the type system tells them apart. It is both the LiteLLM-internal request data that the proxy enriches with billing and budget fields, and the provider-facing body that the passthrough route forwards verbatim to AWS. Because both roles are just `dict`, enrichment meant for the first role silently lands in the second.

A second, compounding footgun makes the leak survive even across separate reads of the body. The cached-body accessor returns a fresh outer dict but shares the nested objects by reference:

```python
# litellm/proxy/common_utils/http_parsing_utils.py:151
return {key: parsed_body[key] for key in accepted_keys}
```

The outer comprehension is a shallow copy, so the nested `metadata` object handed to the auth-time enrichment code is the same object that lives in `request.scope["parsed_body"]` and is later serialized to AWS. Mutating it in one place mutates it everywhere.

The pre-call enrichment then mutates that shared nested object in place, at several sites that all follow the same `data[metadata]["tags"] = ...` shape:

```python
# litellm/proxy/litellm_pre_call_utils.py:1120  (key-level)
data[_metadata_variable_name]["tags"] = LiteLLMProxyRequestSetup._merge_tags(...)
# :1651 (team-level), :1686 (project-level) follow the same pattern
# pass_through_endpoints.py:1478 (header-level)
metadata["tags"] = []
metadata["tags"].extend(tags_to_add)
```

The forward path then serializes the cached parsed body straight to the upstream request:

```python
# litellm/proxy/pass_through_endpoints/pass_through_endpoints.py:454
response = await async_client.request(..., json=_parsed_body, ...)
```

The only sanitizer on this path, `filter_internal_params()` in `litellm/litellm_core_utils/core_helpers.py`, strips MCP-related keys and nothing else, so `tags` is never removed before forwarding (per #30629's root-cause analysis). The net effect is that a LiteLLM-internal billing tag becomes a provider-payload field and AWS rejects the whole request.

## Timeline

This bug is one instance of a category that has been fixed route by route, never structurally. The dates below are the hard signal that a point-patch strategy does not converge.

1. The spend-tag feature merges caller, key, team and project tags into the request `metadata` so spend logs and per-tag budgets can attribute cost. The merge has always written into the native `metadata` object in place, which is harmless only as long as nothing forwards that object to a provider that validates its schema.

2. 2026-03-27: [#24661](https://github.com/BerriAI/litellm/pull/24661) merged, adding a strip of undocumented `metadata` keys before sending to the Anthropic API. This fixed the symptom on `/chat/completions` only. It is a point patch at one provider transform, not a guard on the shared body, so every other route stayed exposed.

3. 2026-05-06: [#27262](https://github.com/BerriAI/litellm/pull/27262) opened to do the same strip for `/v1/messages` (anthropic_messages). Still open as of this writing. The same fix is now being written a second time, for a second route.

4. 2026-06-12: [#30301](https://github.com/BerriAI/litellm/issues/30301) opened: "Harden provider transforms against LiteLLM-internal optional_params leaking into request bodies." The team has now explicitly named the category. No enforcement landed before the next instance was reported.

5. 2026-06: the Bedrock invoke/converse passthrough route (`POST /bedrock/model/{modelId}/invoke`) is found to forward the mutated body verbatim, producing the 400 in #30629. This is the same symptom class as #24661, #27262 and #30301, on a route none of those covered, and the route that Claude Code in Bedrock mode depends on.

6. 2026-06: [#30985](https://github.com/BerriAI/litellm/pull/30985) (open) fixes the key-level leak by pre-seeding `litellm_metadata` for the metadata-bearing routes before the merge runs, so the merge writes into a fresh dict rather than the shared body. [#30994](https://github.com/BerriAI/litellm/pull/30994) addressed the `x-litellm-tags` header variant and was closed as superseded once #30985 covered that vector too.

### Why each layer failed to catch it

The leak is conditional on the body already containing a `metadata` object, because tags are merged only into an existing `metadata`. A test that sends no `metadata` passes, which is exactly the case most tests use, while Claude Code sends `metadata.user_id` on every call and so always fails. The happy path masked the bug.

There is no type distinction between "LiteLLM-internal request data" and "provider-facing body." Both are `dict`, so a type checker had nothing to flag when enrichment wrote into the forwarded object.

The cached-body accessor's shallow-copy-outer, shared-nested-references behavior is an undocumented aliasing footgun with no test and no type guard. Code that looks like it is working on a private copy is in fact mutating shared state.

The single sanitizer, `filter_internal_params()`, encoded an allow-by-omission policy: it removed only the internal keys someone remembered to list (MCP keys), so any new internal field defaulted to leaking. There was never an exhaustive, typed contract of what counts as internal.

Each prior fix was a strip added at one route's transform. A strip is a deny-list applied after the leak is already in the object, so it has to be re-derived and re-applied at every new egress point, and it is silently absent at any point nobody patched yet. The recurrence across #24661, #27262 and #30629 is the predictable result.

## How to prevent this entire category at type-check and lint time

The goal is to make the leak unrepresentable: writing LiteLLM-internal data into the provider-facing body should fail `basedpyright` or the lint gate before it can run. The category has two sub-mechanisms, and both are closable.

The two sub-mechanisms are mutation of shared state (this bug: enrichment mutates a body object that is later forwarded) and missing strip of an internal field (the `stream_chunk_size` and `optional_params` variants in #30301). Both reduce to the same root, which is that one untyped `dict` is allowed to be both the internal bag and the wire payload.

### Layer 1: make the parsed body an immutable Mapping, deeply

Change the return type of `_read_request_body` and `_safe_get_request_parsed_body` from `dict` to a read-only `Mapping[str, JSONValue]`, with nested objects also typed as `Mapping`. Every in-place merge site then fails type checking. This is verified, not asserted; running a type checker on the mutation shape produces:

```
reportIndexIssue: "__setitem__" method not defined on type "Mapping[str, object]"
```

(`basedpyright` is a `pyright` fork and reports this identically; the proxy already runs the `basedpyright` gate in CI.) With this one change, all four merge sites (`litellm_pre_call_utils.py:1120`, `:1651`, `:1686` and `pass_through_endpoints.py:1478`) become compile-time errors and must be rewritten to build a new dict, which is the functional, no-mutation style the repo's `LIT001`/`LIT002` lints already require. The shared-reference footgun in `_safe_get_request_parsed_body` stops mattering, because a shared reference that cannot be mutated cannot leak.

### Layer 2: give internal data a type that cannot be the wire payload

Model the two roles as two types. Internal enrichment (tags, spend metadata, `litellm_metadata`) operates on a `LiteLLMRequestData` structure. The provider-facing body is a distinct `ProviderFacingBody` (the immutable Mapping from Layer 1). The single function that forwards upstream accepts only `ProviderFacingBody`, so handing it internal data is a type error. Crossing the boundary is done once, by an explicit projection that drops internal keys and returns the wire type. That projection replaces the scattered, allow-by-omission `filter_internal_params()` with one typed, tested choke point, and it is the general form of a per-provider `client.post` wrapper.

### Layer 3: close the provider metadata schema where it is known-closed

Where a provider's metadata contract is closed, encode it. For Anthropic on Bedrock that is `user_id` only. Model it as a closed `TypedDict` or a Pydantic model with `extra="forbid"`. Building it with a stray `tags` key then fails type checking, again verified:

```
reportReturnType: Type "dict[str, str | list[str]]" is not assignable to return type "ProviderMetadata"
```

This closes the missing-strip sub-mechanism that #30301 and the `stream_chunk_size` case share, because an internal field is no longer a member of the type that gets serialized.

### Layer 4: lint backstop

With Layer 1 in place, `LIT001`/`LIT002` shift from advice to enforcement for this code, since the immutable type makes seed-then-mutate impossible to express. A targeted lint that forbids subscript assignment on the parsed-body symbol is optional belt-and-suspenders.

This is the concrete implementation strategy for the already-filed #30301; the type boundary is the hardening that issue asks for.

## What the exemplar got right, and where this goes further

This RCA follows the structure of the internal `stream_chunk_size` RCA: a clear introduced / consequence / fixed-by header, a dated timeline, a layered "why didn't this get caught," concrete code-level prevention, and honesty about limits. That template is worth keeping.

The exemplar concluded that a foolproof type system is impossible while passthrough keeps an open `extra_body`, and that the foolproof version is reachable only with `drop_params=True`. That conclusion is correct for its framing, because deciding which unknown keys are control versus provider inside one open bag is undecidable without an allow-list. This category's core footgun is different and fully closable. It is mutation of shared state and role confusion, not key classification. You do not need to know every provider key to make the body immutable and to give internal data a separate type. The provider body can stay open, an arbitrary JSON mapping, and still be read-only, while internal data lives in a type that is structurally incapable of being the forwarded payload. So for this category the foolproof guarantee is reachable without giving up the passthrough feature, which is further than the exemplar could go for its case.

The exemplar's minor weaknesses, worth avoiding in future RCAs, are that its prevention is scoped to one provider transform (Bedrock Invoke) rather than the shared egress boundary, and that its prevention list has auto-numbering glitches that make the step count hard to read.

## Limitations and residual risk

Open-body passthrough still forwards whatever the client itself places in `metadata`. The guarantee here is only that LiteLLM-internal data will not be injected. If a provider rejects a client-supplied non-`user_id` metadata key, that is the client's own payload and out of scope for this category.

Deep-freezing nested bodies has a per-request cost that should be measured. A copy-on-enrich projection at the boundary is an alternative to freezing the cache if profiling shows the freeze is too expensive.

A closed provider-metadata schema (Layer 3) must track the provider's API. `user_id`-only reflects the current Anthropic-on-Bedrock contract and would need updating if AWS widens it.
