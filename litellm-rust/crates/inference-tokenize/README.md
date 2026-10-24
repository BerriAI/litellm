# Native tokenization

`TokenizeRoute` accepts `litellm_inference_tokenize::TokenizeRequest`, which is the host execution envelope. It is distinct from `litellm_llms_types::formats::tokenize::TokenizeRequest`, which is the shared raw-text API payload. The public payload uses a strict text-only schema with required `model` and `prompt`, an `include` array, and an optional non-null boolean `add_special_tokens`

An empty `include` array is count-only. `TokenizeInclude::new(ids, tokens)` constructs requested representations, and `TokenizeInclude::output()` projects the internal representation enum. `From<llms_types::formats::tokenize::TokenizeRequest>` creates the host envelope and defaults operational fields. The public payload has no `output` field, unknown fields are rejected, and explicit `null` for `add_special_tokens` is rejected. Projection happens after provider response validation, and include values are never sent as provider parameters

The only include values are `token_ids` and `tokens`. Duplicate entries, unknown entries, wrong types, and null values are rejected. Omitting `include` is equivalent to `include: []`, and model must be nonempty while prompt may be empty or whitespace without trimming

```json
{"model":"public-alias","prompt":"hello"}
```

The count-only response contains a count with null representation arrays. A representation request can ask for either or both arrays

```json
{"model":"public-alias","prompt":"hello","include":["token_ids","tokens"]}
```

Omitting `add_special_tokens` preserves the provider policy. Hosted vLLM accepts explicit true and false values. Operational metadata belongs to the host envelope and is not forwarded as provider kwargs. Existing provider wire flags are unchanged, and count-only projection does not claim provider-side wire or cost savings

Downstream issue28444 acceptance requires fresh native-build coverage for `/tokenize` and `/v1/tokenize` across count-only and requested representation modes. A registered `CustomLogger` must prove that pre-call prompt mutation reaches the executed input, post-success response replacement reaches the wire HTTP result with correct `Content-Length` and context, each hook runs exactly once, and the mutation-bypass negative check remains enforced. Native `after_provider_response` is a notification and does not rewrite responses. This is not a generic passthrough fix, and legacy `/utils/token_counter` behavior remains unchanged
