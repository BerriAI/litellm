# Local SDK parity report

From the repository root:

```bash
uv run python -m tests.sdk_function_trace.local_parity
```

One invocation groups SDK response parity and function-trace parity by Rust SDK route, with separate sync and async rows. Every case is marked non-strict xfail: XPASS means the comparison matched; XFAIL prints a mismatch, unsupported operation, or missing-coverage reason. Ordinary parity failures do not fail the command; collection and runner errors still do

SDK comparisons call the public Python SDK with `rust=False` and `rust=True` against a local HTTP mock provider. They require the native extension and verify Rust execution attribution so Python fallback cannot count as a match. Responses are compared after excluding hidden metadata and, for chat completions, generated IDs and creation timestamps. No provider credentials or running proxy are needed

Implemented comparisons cover Mistral OCR and Anthropic chat completions/Messages. Function coverage currently reuses the Mistral OCR transformation trace tests. Other rows explicitly report coverage gaps; the report is not exhaustive provider, streaming, error, or request-shape coverage

The module is outside pytest's default filename patterns and the entry point refuses to run when `CI` is set. Existing function-parity tests retain their normal failure behavior
