# SDK route trace audit

Run the four native HTTP route families in both modes from the repository root:

```bash
uv run python -m tests.sdk_function_trace.compare --route all --both --check
```

The local fixture matrix on 2026-09-02 completed 15 successful engine invocations and one expected skip. Every successful invocation issued exactly one local HTTP request. All five comparable route/mode pairs have identical canonical steps in the same order, with no Python-only or Rust-only steps

| Route | Python async | Python sync | Rust async | Rust sync |
| --- | --- | --- | --- | --- |
| Chat completions, Anthropic | Pass | Pass | Pass | Pass |
| Messages, Anthropic | Pass | Unsupported, skipped | Pass | Pass |
| OCR, Mistral | Pass | Pass | Pass | Pass |
| Audio transcription, Bedrock | Dispatch only | Dispatch only | Pass | Pass |

The same canonical step sequence ran in sync and async for each engine with both modes available. Bedrock transcription's Python SDK delegates to Rust, so its two successful calls do not establish independent provider parity. Realtime and Responses WebSockets are outside this HTTP fixture runner

Chat and OCR also have identical projected nesting in both modes. Async Messages has the same helper nesting beneath its handler, but Python starts that handler on a worker thread, so it appears as a second root. The comparison preserves this physical thread boundary and checks step order independently of absolute depth

Rust now resolves chat providers and supported parameters before entering its handler. Chat and Messages validate the environment and transform requests inside their handlers. Messages builds the final URL after transformation. OCR resolves its config and maps supported parameters during preparation, then validates credentials, builds the URL, and transforms the request inside its handler. Its during-call guardrails still run before HTTP, within the provider-call lifecycle phase

The environment hooks execute credential and header validation. Chat's supported-parameter hooks return OpenAI names paired with provider names and feed the existing request acceptance checks. The direct Rust API still accepts provider-mapped parameters, and its supported subset is smaller than Python's. Matching the pipeline does not establish identical parameter contracts

`--check` now fails if either comparable engine has missing, extra, or reordered canonical steps, even if its individual stage checks pass. Bedrock transcription and sync Messages report `UNAVAILABLE` for cross-language parity; native execution is still checked. Passing establishes step coverage and order for one non-streaming fixture per route, not complete request, response, error, or provider parity. The previously recorded OCR response gaps remain in `ocr-comparison.md`
