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

Hosts can select a deployment-configured local tokenizer with `TokenizeBackend::Local`. The host supplies a `Sourced<Arc<LocalTokenizerConfig>>` whose source must be deployment or environment. Request-sourced local configuration is rejected before artifact access. Local calls reject credentials, API bases, headers, provider parameters, and provider selection

```rust
use std::sync::Arc;

use litellm_auth::{InputSource, Sourced};
use litellm_inference_tokenize::{TokenizeBackend, TokenizeRequest};
use litellm_llms_types::formats::tokenize::TokenizeInclude;
use litellm_token_counter::raw_text::{
    LocalTokenizerConfig, RawTextArtifact, RawTextEngine,
};

let config = LocalTokenizerConfig {
    id: "support-tokenizer".into(),
    revision: "2025-01".into(),
    engine: RawTextEngine::HuggingFace,
    artifact: RawTextArtifact::File {
        path: "/srv/tokenizers/support.json".into(),
        sha256: "configured-64-character-sha256-digest".into(),
    },
    special_text: None,
    add_special_tokens: None,
};
let request = TokenizeRequest {
    model: "public-alias".into(),
    prompt: "Example".into(),
    include: TokenizeInclude::new(true, true),
    add_special_tokens: None,
    backend: TokenizeBackend::Local {
        config: Sourced::new(Arc::new(config), InputSource::Deployment),
    },
    ..Default::default()
};
```

Local engine and artifact types are explicit. Hugging Face accepts tokenizer JSON and preserves its normalizer, postprocessor, truncation, and padding. Tiktoken accepts a named encoding and supplied rank content. File artifacts require a SHA-256 digest. URL loading, implicit model selection, and request-controlled artifact paths are unsupported

Hugging Face defaults to special-text recognition and special-token insertion. Tiktoken defaults to literal special text and rejects special-token insertion. Tiktoken does not provide token pieces. Raw-text Hugging Face loading rejects positive BPE dropout because it is stochastic, while absent or zero dropout remains accepted. An explicit local `add_special_tokens` value clones the trusted configuration and participates in registry identity. Local count-only calls use the standard encode-and-length path and must not enable experimental or specialized counting features downstream

Omitting `add_special_tokens` preserves the configured local policy or the provider policy. Hosted vLLM accepts explicit true and false values, Watsonx rejects either explicit value before secret resolution, and Tiktoken rejects true while allowing false. Operational metadata belongs to the host envelope and is not forwarded as provider kwargs. Existing provider wire flags are unchanged, and count-only projection does not claim provider-side wire or cost savings

The local response model is the configured tokenizer identity. Local execution facts use `provider = "local"`, `model = config.id`, the existing `ResultSource::Provider` value for a computed result rather than HTTP, and `Cost::Reported` with amount zero. Local calls do not run provider pre or post hooks, do not fabricate HTTP objects, and still await `result_ready`. Provider response models use the resolved provider model and provider behavior remains unchanged

`LocalTokenizerRegistry` bounds cached tokenizer instances to 32 by default and limits blocking work to available parallelism. Hosts may inject a shared registry or preload trusted configs. Artifact reads, hashing, codec initialization, counting, and encoding run on bounded blocking workers. A warm file-backed entry remains usable if its source file is later removed because its verified content hash is pinned in the loaded instance

Downstream issue28444 acceptance requires fresh native-build coverage for `/tokenize` and `/v1/tokenize`, local and provider, across count-only and requested representation modes. A registered `CustomLogger` must prove that pre-call prompt mutation reaches the executed input, post-success response replacement reaches the wire HTTP result with correct `Content-Length` and context, each hook runs exactly once, and the mutation-bypass negative check remains enforced. Native `after_provider_response` is a notification and does not rewrite responses. This is not a generic passthrough fix, and legacy `/utils/token_counter` behavior remains unchanged
