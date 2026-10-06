# Token counting

`Tokenizer` is the text-counting interface. `TextCodec` adds encoding, decoding, and a name. `TokenCounter` applies LiteLLM request, message, and tool accounting using any `Tokenizer`

Counts follow the codec: tiktoken treats special-token spellings as ordinary text, while Hugging Face applies its added tokens, post-processing, padding, and truncation. `fast=True` preserves those semantics and requests acceleration where available. Unsupported configurations use the normal codec, including tiktoken encodings without a scanner and builds without the `fast` feature. Invalid input and process-guard errors still propagate. Runtime request counting currently uses the normal codec; the custom accelerator is retained for explicit use and testing

`FastCounter: TextCodec` exposes an optional accelerator over a loaded codec. `None` means callers should use that codec. The Python bridge caches this selection per immutable tokenizer, shares it with request counters, and initializes it with the GIL released. Hugging Face can also choose the full encoder per input when added tokens require it

The `fast` feature provides `fast::FastTokenizer` from `litellm-token-counter-fast`. `TokenCounter::from_json_fast` uses this implementation

The `huggingface` feature provides `huggingface::HuggingFaceTokenizer` through the upstream `tokenizers` library. `TokenCounter::from_json` uses this implementation

The `tiktoken` feature provides `tiktoken::TiktokenTokenizer` through `tiktoken-rs`. Select an encoding with `TokenCounter::from_tiktoken`. The supported names are `cl100k_base`, `o200k_base`, `o200k_harmony`, `p50k_base`, `p50k_edit`, `r50k_base`, and `gpt2`

All three backends are enabled by default in this crate and the Python extension. With `default-features = false`, Rust callers can supply their own `Tokenizer` to `TokenCounter::new` without compiling a built-in backend

Python `tiktoken` and `tokenizers` remain runtime dependencies and the default implementations. The catalog independently selects the tokenizer and request-counting routes. Enabling Rust changes factory dispatch; existing tokenizer objects keep their backend. Native Hugging Face wrappers provide an immutable encoding and decoding API, while training and mutable configuration remain available through the Python backend

Budget checks, cost calculation, and the `max_tokens` adjustment policy belong to `litellm-core-utils`. The counter does not own prices, budgets, or request limits

Run the feature matrix with:

```sh
cargo test -p litellm-token-counter
cargo test -p litellm-token-counter --no-default-features
cargo test -p litellm-token-counter --no-default-features --features fast
cargo test -p litellm-token-counter --no-default-features --features huggingface
cargo test -p litellm-token-counter --no-default-features --features tiktoken
```
