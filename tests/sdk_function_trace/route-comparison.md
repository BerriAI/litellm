# SDK route trace audit

Run the four native HTTP route families in both modes from the repository root:

```bash
uv run tests/sdk_function_trace/compare.py --route all --both --check
```

The local fixture matrix on 2026-09-02 completed 15 successful engine invocations and one expected skip. Every successful invocation issued exactly one local HTTP request. Required instrumented stages were present and obeyed their dependency order

| Route | Python async | Python sync | Rust async | Rust sync |
| --- | --- | --- | --- | --- |
| Chat completions, Anthropic | Pass | Pass | Pass | Pass |
| Messages, Anthropic | Pass | Unsupported, skipped | Pass | Pass |
| OCR, Mistral | Pass | Pass | Pass | Pass |
| Audio transcription, Bedrock | Dispatch only | Dispatch only | Pass | Pass |

The same canonical step sequence ran in sync and async for each engine with both modes available. Bedrock transcription's Python SDK delegates to Rust, so its two successful calls do not establish independent provider parity. Realtime and Responses WebSockets are outside this HTTP fixture runner

The observed provider lookup, request transformation, HTTP request, and response transformation phases agree in dependency order. The complete trees still differ. Chat and Messages perform request transformation inside Python handlers and inside Rust preparation. OCR also checks supported parameters before mapping in Python, while Rust calls that check from its mapper. These differences do not change the required order of request transformation, HTTP, and response transformation

Rust provider lookup and HTTP were implemented but uninstrumented. The audit adds spans around those existing operations. The shared HTTP helper delegates to the existing request builder and preserves route-specific error handling. It adds no new provider transport implementation

Python `validate_environment` still has no single equivalent Rust hook: Rust splits credentials, headers, and URL construction across preparation and provider methods. Python chat's `supported_openai_params` handles OpenAI parameters, while the direct Rust chat API accepts already mapped provider parameters. URL helpers are also not instrumented here. These gaps remain visible instead of being mapped to functions with different contracts

Passing this matrix establishes the declared stage coverage and ordering for one non-streaming fixture per route. It does not establish complete function, request, response, error, or provider parity. The previously recorded OCR response and provider-contract gaps remain in `ocr-comparison.md`
