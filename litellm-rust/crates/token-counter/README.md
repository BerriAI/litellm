# Token counting

`Tokenizer` is the text-counting interface. `TokenCounter` applies LiteLLM request, message, and tool accounting using any implementation of that interface

The `fast` feature provides `fast::FastTokenizer` from `litellm-token-counter-fast`. `TokenCounter::from_json_fast` uses this implementation

The `huggingface` feature provides `huggingface::HuggingFaceTokenizer` through the upstream `tokenizers` library. `TokenCounter::from_json` uses this implementation

The `tiktoken` feature provides `tiktoken::TiktokenTokenizer` through `tiktoken-rs`. Select an encoding with `TokenCounter::from_tiktoken`. The supported names are `cl100k_base`, `o200k_base`, `o200k_harmony`, `p50k_base`, `p50k_edit`, `r50k_base`, and `gpt2`

All three backends are enabled by default. The Python extension builds with `fast` only, which keeps the wheel at the size it had before the split. With `default-features = false`, callers can supply their own `Tokenizer` to `TokenCounter::new` without compiling a built-in backend

Budget checks, cost calculation, and the `max_tokens` adjustment policy belong to `litellm-core-utils`. The counter does not own prices, budgets, or request limits

Run the feature matrix with:

```sh
cargo test -p litellm-token-counter
cargo test -p litellm-token-counter --no-default-features
cargo test -p litellm-token-counter --no-default-features --features fast
cargo test -p litellm-token-counter --no-default-features --features huggingface
cargo test -p litellm-token-counter --no-default-features --features tiktoken
```
