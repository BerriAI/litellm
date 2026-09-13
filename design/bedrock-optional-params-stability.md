# Leak-proofing provider request bodies: a typed internal-params boundary

Status: proposal
Owner: mateo
Motivating issues: #30371 (invoke leak), #30314 (titan embed leak), #30301 (the class)

## The problem in one paragraph

`optional_params` is an untyped `dict[str, Any]` that carries two unrelated kinds of data: provider inference parameters that belong in the request body (`max_tokens`, `temperature`, `top_p`), and LiteLLM-internal control knobs that must never reach a provider (`skip_mcp_handler`, `stream_chunk_size`, `fake_stream`, `cache_control_injection_points`). The provider transforms read the control knobs out of that dict and then splat the same dict straight into the request body, so the internal knobs ride along into the wire payload. Strict-schema providers reject unknown fields, so a single leaked knob turns into a hard request failure: `#30314` is a Titan embeddings call dying on `extraneous key [cache_control_injection_points] is not permitted`, `#30371` is the invoke path forwarding `skip_mcp_handler`, and `#30239` was `stream_chunk_size: Extra inputs are not permitted` against Bedrock.

## Why the current defenses do not hold

Two mechanisms exist today and neither closes the class. The first is per-handler whack-a-mole: each transform is expected to `pop()` every internal knob before it splats. `base_invoke_transformation.py` pops `stream` and `stream_chunk_size` by hand; the embeddings and image transforms pop nothing. This is a blocklist re-implemented from memory at every site, and it failed exactly as you would expect; `#30240` was two more `pop` calls bolted on after `stream_chunk_size` shipped to production. The second is `filter_internal_params` in `core_helpers.py`, a hard-coded set of three MCP keys, wired into exactly one of the dozens of Bedrock splat sites (`converse_transformation.py`). Everywhere else, including all five invoke branches, both Titan embedding transforms, and the Nova/Stability image transforms, the splat is unguarded.

The structural fact under all of this: there are ~129 `**optional_params` / `**inference_params` splat constructions across `litellm/llms/`, and the safety of each one depends on a human remembering an invisible rule. That is the bug generator we want to remove, not any single leaked key.

## On the typing hypothesis

The starting hypothesis was that 100 percent typing of `optional_params` would reduce leakage. That is directionally right about where the rot is, but the literal version does not work, and it is worth being precise about why so we build the right thing. A `TypedDict` is a plain `dict` at runtime; `{"prompt": prompt, **optional_params}` still copies every key regardless of annotations, and the type checker is gone by the time the body is serialized. Typing the bag does not stop a runtime splat.

The leverage that typing actually gives us is narrower and it is the core of this proposal: give the internal-knob namespace a single typed identity. The leak persists because the set of "things that are internal" lives only in people's heads and in scattered `pop` calls, so the blocklist drifts out of sync with reality the moment someone adds a knob. If instead every internal knob is a member of one typed enum, and both the code that injects a knob and the code that filters it derive from that same enum, the blocklist can no longer go stale; adding a knob without registering it becomes a type-level and test-level failure rather than a production 400. So we keep the spirit of the hypothesis (use the type system to make the failure unrepresentable) and drop the part that does not survive contact with runtime (annotating the whole bag).

The north-star version of this same idea is to stop conflating the two channels at all. `litellm_params` already travels alongside `optional_params` into every `transform_request`, and it is the natural home for control knobs. If every internal knob lived in `litellm_params` and `optional_params` held only provider fields, the splat would be safe by construction and no filter would be needed. Moving every injection and read site there is a large, risky refactor, so it is the direction of travel, not this change. The boundary-filter below is the shippable enforcement that makes the system safe now and converges toward that split later.

## Design

Three layers, smallest mechanism that makes the class unrepresentable.

1. Typed registry as single source of truth. Introduce `LiteLLMInternalParam` (a `StrEnum`) in `litellm/types/` enumerating every internal knob the audit surfaces: the MCP keys (`skip_mcp_handler`, `_skip_mcp_handler`, `mcp_handler_context`), `stream_chunk_size`, `fake_stream`, `cache_control_injection_points`, and whatever else the sweep finds. `filter_internal_params` stops carrying a hand-written literal set and derives its keys from this enum. Knob injection sites reference the enum member, so the registry and the filter cannot drift apart.

2. One enforced serialization boundary. Filter at the last possible moment, the point where params become a request body, so that handler logic which legitimately reads control knobs earlier still works. Rather than 129 hand-edited `pop` lines, route the body construction through the shared `BaseConfig` and base HTTP handler choke points plus the few bespoke Bedrock handlers (invoke, embeddings, image), so the splat sites are covered by construction. The filter is a blocklist on purpose: an allowlist keyed off each provider's declared params would over-drop the arbitrary native inference fields users pass through to Bedrock invoke via `extra_body`, which is a real and supported path. A blocklist sourced from the typed registry removes exactly the internal keys and preserves passthrough.

3. A mutation-killing regression test. One parametrized test walks the provider chat and embedding configs, calls `transform_request` with `optional_params` seeded with every registry key plus a couple of representative real provider params, serializes, and asserts two things: no registry key appears anywhere in the body, and the real provider params do survive. The second assertion is what stops a future "fix" from over-filtering and silently dropping `temperature`. A newly added internal knob that is not in the registry fails this test instead of failing in production.

## Scope and proof

Scope is all providers, closing the class rather than the three instances, with Bedrock as the first-class motivating surface. The implementation PR carries the registry, the boundary wiring, and the test; the rollout can be staged behind the same mechanism if review prefers a smaller first diff, but the decision here is to land the class fix.

Proof of fix is a live proxy hitting real Bedrock, not a pytest transcript. Before the change, `curl` the Titan image-embeddings model with `cache_control_injection_points` set and show the Bedrock 400; after, show the same call succeeding. Same shape for the invoke path with an internal knob present. The before/after pair is the evidence in the PR.

## Risks

The main risk is over-filtering, removing a key that a provider actually wants; assertion two in the test guards against it, and the registry only ever contains LiteLLM-internal keys. The secondary risk is a knob injected downstream of the boundary; the audit inventories injection sites, and the CI test catches any registry key that survives a transform. Cost is one dict comprehension per request, which is negligible against a network call. Out of scope here, and called out so it is not mistaken for covered, are the provider-feature fields that need reshaping rather than dropping (`#27532` `context_management`, `#28081` `betas`); those are translation work, not leakage, and the same audit can inventory them for a follow-up.
