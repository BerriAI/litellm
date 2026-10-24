# Tokenizer registry

This branch parks the local tokenizer work that was cut from #48102 and #48146. Its tip is the old #48146 head, so it still holds `token-counter/src/raw_text.rs`, `inference-tokenize/src/local.rs`, the tiktoken `from_ranks` and Hugging Face policy changes, and the Python `backend=` wiring. None of it should merge from here. Each PR below lifts its piece onto the current default branch

## Problem

Tokenizers are loaded and cached in five places, each with its own idea of what makes two tokenizers the same

| Where | What | Key | Gap |
|---|---|---|---|
| `litellm/utils.py` `_load_huggingface_tokenizer` | cohere, anthropic, deepseek, llama2, llama3. Anthropic and deepseek come from bundled JSON, the rest from the HF Hub | `(kind, backend)` plus a TTL failure cache | Hub downloads are unpinned |
| `litellm/utils.py` `_select_custom_tokenizer_helper` | caller-supplied HF tokenizers | `(identifier, revision, auth_token, backend)` | `revision="main"` sticks to whatever the process downloaded first, no content hash |
| `litellm/litellm_core_utils/default_encoding.py` | tiktoken ranks | hash-named files in `tokenizers/`, found through `TIKTOKEN_CACHE_DIR` | the lookup is an env var side effect |
| `token-counter-tiktoken/src/ranks.rs` | six `OnceCell` encoders | encoding name | ranks cannot be supplied at runtime |
| `litellm/rust_bridge/token_counter.py` `_counter` | native counters, `lru_cache(maxsize=4)` | `(factory, tokenizer)` | only anthropic, cl100k_base and o200k_base are native, so deepseek, cohere, llama and p50k/r50k fall back to Python |

The parked `LocalTokenizerRegistry` keys on `(id, revision, sha256, engine, policy)`, is bounded, and loads each entry once under concurrency. That is the right shape, but only `/tokenize` could use it, so it added a sixth cache instead of replacing any

## Goal

One Rust registry that every tokenizer goes through. A tokenizer is a descriptor with an id, a revision, a content hash, an engine and a special-token policy. Bundled artifacts become descriptors checked into the tree. Model to tokenizer selection (`huggingface_tokenizer_kind`) stays in Python behind `litellm/rust_bridge/host/`, so no selection rules are ported into Rust

```rust
pub struct TokenizerDescriptor {
    pub id: String,
    pub revision: String,
    pub engine: Engine,            // HuggingFace | Tiktoken { encoding }
    pub artifact: Artifact,        // Bundled { name } | Inline { content } | File { path, sha256 }
    pub special_text: Option<SpecialTextPolicy>,
    pub add_special_tokens: Option<bool>,
}

let anthropic = registry.get(&TokenizerDescriptor::bundled("anthropic")).await?;
let count = anthropic.count("hello")?;
```

## Stack

### PR 1, registry foundation in token-counter

`feat(rust): add content-addressed tokenizer registry`

Takes `raw_text.rs`, the registry half of `local.rs`, `TiktokenTokenizer::from_ranks`, `HuggingFaceTokenizer::from_json_with_policy` and `is_deterministic` from this branch. Adds a `Bundled` artifact for the files in `litellm/litellm_core_utils/tokenizers/` with their digests pinned. No caller changes. Tests cover descriptor validation, policy defaults per engine, hash mismatch, one load under concurrent misses, failed loads not being cached, and capacity eviction

### PR 2, move counting onto the registry

`refactor(token-counter): resolve every tokenizer through the registry`

`rust_bridge/token_counter.py` asks the registry for a descriptor instead of the three hardcoded native tokenizers, which makes deepseek and p50k/r50k native. `_select_custom_tokenizer_helper` resolves `revision` to a commit before caching, so `main` no longer pins the first download. `ranks.rs` statics become bundled descriptors. Parity tests compare counts from the Python and Rust paths for every bundled tokenizer

### PR 3, local backend on /tokenize

`feat(tokenize): add a deployment-configured local tokenizer backend`

Brings back `TokenizeBackend::Local`, `execute_local`, the bridge `project_backend`, the `backend=` SDK argument, the Router local-deployment checks and the local tests from this branch, all resolving through the shared registry instead of a route-owned one. It depends on the merged tokenize stack (#48092, #48091, #48102, #48146)

## Out of scope

Upstream error headers on native routes. #48102 used to add `headers` to `TransportError::Http` for this. It belongs to the typed `Rejected` reply on `litellm_rust_error_context`, which covers every route at once
