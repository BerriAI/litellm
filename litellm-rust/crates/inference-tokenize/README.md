# Native tokenization

`TokenizeRoute` accepts `litellm_inference_tokenize::TokenizeRequest`, which is the host execution envelope. It is distinct from `litellm_llms_types::formats::tokenize::TokenizeRequest`, which is the shared raw-text API payload. Import `TokenizeOutput` explicitly from `litellm_llms_types::formats::tokenize`. The output choice selects `count`, `ids`, `tokens`, `both`, or `available`, and defaults to `available`. Projection happens after provider response validation and the choice is never sent as a provider parameter. Strict representation requests fail when the selected provider cannot return that representation

Hosts can select a deployment-configured local tokenizer with `TokenizeBackend::Local`. The host supplies a `Sourced<Arc<LocalTokenizerConfig>>` whose source must be deployment or environment. Request-sourced local configuration is rejected before artifact access. Local calls reject credentials, API bases, headers, provider parameters, and provider selection.

```rust
use std::sync::Arc;

use litellm_auth::{InputSource, Sourced};
use litellm_inference_tokenize::{TokenizeBackend, TokenizeRequest};
use litellm_llms_types::formats::tokenize::TokenizeOutput;
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
    output: TokenizeOutput::Both,
    backend: TokenizeBackend::Local {
        config: Sourced::new(Arc::new(config), InputSource::Deployment),
    },
    ..Default::default()
};
```

Local engine and artifact types are explicit. Hugging Face accepts tokenizer JSON and preserves its normalizer, postprocessor, truncation, and padding. Tiktoken accepts a named encoding and supplied rank content. File artifacts require a SHA-256 digest. URL loading, implicit model selection, and request-controlled artifact paths are unsupported.

Hugging Face defaults to special-text recognition and special-token insertion. Tiktoken defaults to literal special text and rejects special-token insertion. Tiktoken does not provide token pieces. Raw-text Hugging Face loading rejects positive BPE dropout because it is stochastic. Dropout absent or zero remains accepted. Local count-only calls use the standard encode-and-length path and must not enable experimental or specialized counting features downstream.

The public response keeps the requested model alias, while execution facts use the configured tokenizer identity. Local facts use `provider = "local"`, `model = config.id`, the existing `ResultSource::Provider` value for a computed result rather than HTTP, and `Cost::Reported` with amount zero. Local calls do not run provider pre or post hooks, do not fabricate HTTP objects, and still await `result_ready`. Provider response behavior remains unchanged.

`LocalTokenizerRegistry` bounds cached tokenizer instances to 32 by default and limits blocking work to available parallelism. Hosts may inject a shared registry or preload trusted configs. Artifact reads, hashing, codec initialization, counting, and encoding run on bounded blocking workers. A warm file-backed entry remains usable if its source file is later removed because its verified content hash is pinned in the loaded instance.

Downstream issue28444 acceptance requires fresh native-build coverage for `/tokenize` and `/v1/tokenize`, local and provider, across count and available outputs. A registered `CustomLogger` must prove that pre-call prompt mutation reaches the executed input, post-success response replacement reaches the returned HTTP result with correct `Content-Length` and context, and each hook runs exactly once, including a mutation-bypass negative check. Native `after_provider_response` remains a notification and does not rewrite responses. This downstream acceptance is not a generic passthrough fix. Legacy `/utils/token_counter` behavior remains unchanged
