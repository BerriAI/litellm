# characterization_xai — Grok (xai) corpus

Provenance: GENERATED from v1 in-process at HEAD, not vendored. The
characterization branch (mateo/translation-characterization-providers) has
zero xai fixtures and no recorded xai vendor traffic exists in the repo, so
every snapshot pins v1-as-executed (the primary reference under the
differential rule in 05-provider-expansion.md). The wire shapes in
`fixtures/` are hand-authored to the documented xAI API shapes plus the
quirks researcher-3 verified in-process (finish_reason "", reasoning token
accounting, num_sources_used, the `choices: []` usage tail).

- `cases/` — OpenAI-format chat requests (request seam input). Models cover
  grok-2-1212 / grok-3 / grok-3-mini / grok-4-0709 / grok-code-fast-1, the
  serve side of all three per-model gates (stop, frequency_penalty,
  reasoning_effort).
- `fixtures/responses/` — provider response bodies `{model, body}`.
- `fixtures/streams/` — SSE chunk payload lists `{stream_options, events}`.
- `snapshots/` — v1 output at the pinned seams (canonical JSON: sorted keys,
  2-space indent, trailing newline):
  - `requests/`: `get_optional_params("xai")` -> extra_body pop ->
    `XAIChatConfig.transform_request` (the httpx-handler call order)
  - `responses/`: `XAIChatConfig.transform_response().model_dump()` (LIVE on
    the httpx path; includes the xai usage post-steps)
  - `streams/`: SSE data-lines -> `XAIChatCompletionStreamingHandler` ->
    `CustomStreamWrapper("xai")` chunk dumps, ambient frozen at 1718064000

Regenerate (a reviewed snapshot diff, never silent):
`LITELLM_LOCAL_MODEL_COST_MAP=True python -m tests.test_litellm.translation.generate_xai_snapshots`
