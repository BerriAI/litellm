# Exa search

Use `litellm.search()` or `await litellm.asearch()` with `search_provider="exa_ai"` and `EXA_API_KEY`. Native Exa parameters retain their camelCase names. See the [current search contract](https://exa.ai/docs/reference/search).

```python
import litellm

response = litellm.search(
    query="recent advances in battery recycling",
    search_provider="exa_ai",
    contents={"highlights": True},
)
for result in response.results:
    print(result.title, result.url, result.snippet)
    print(result.model_dump().get("highlights"))
print(response.model_dump().get("requestId"))
```

Exa defaults to `type="auto"` and 10 results. The public `numResults` range is 1 to 100, with limits that may vary by mode. The current modes are `instant`, `fast`, `auto`, `deep-lite`, `deep`, and `deep-reasoning`. `additionalQueries` is for deep modes. LiteLLM forwards these values without translating them into legacy `neural` or `keyword` modes. Category hints, domain filters, publication windows, moderation, `objective`, `systemPrompt`, `userIdentifier`, and enterprise `compliance` are forwarded. Exa validates provider-specific limits and unsupported combinations.

Exa defaults `moderation` and `stream` to false, text rendering verbosity to `compact`, `includeHtmlTags` to false, extras counts and `subpages` to zero, and `livecrawlTimeout` to 10,000 milliseconds. Omitted freshness controls use cached content when available and fetch pages without cached content. LiteLLM leaves these defaults to Exa.

LiteLLM retains its existing compatibility default of `contents={"text": True}` when `contents` is omitted. Pass `contents={}` or `None` for metadata only, or request highlights explicitly as above. An explicit contents object is never supplemented with full text. `max_results`, `search_domain_filter`, and `country` map to `numResults`, `includeDomains`, and `userLocation`. When both spellings are supplied, the unified parameter takes precedence, as before. `max_tokens_per_page` is not mapped to a character limit.

For location, `userLocation` also accepts `{"country": "US"}` or `{"latitude": 40.7, "longitude": -74.0}` with an optional country. Coordinates must be provided together.

## Extraction options

`contents.text` accepts a boolean or an object with `maxCharacters`, `includeHtmlTags`, `verbosity` (`compact`, `standard`, `full`), `includeSections`, and `excludeSections`. Section names are `header`, `navigation`, `banner`, `body`, `sidebar`, `footer`, and `metadata`. Section selection is best effort. Exa recommends `maxAgeHours=0` to apply rendering controls to freshly fetched content.

`contents.highlights` accepts a boolean or an object with a selection `query`, `maxCharacters`, or the beta options below. `contents.summary` accepts an object with `query` and a JSON `schema`. Prefer one extraction mode appropriate to the task. Summary generation adds per-result synthesis.

Freshness and discovery options are `maxAgeHours`, `livecrawlTimeout`, `snapshotAsOf`, `subpages`, and `subpageTarget`. `maxAgeHours` controls cache freshness, not publication recency. `snapshotAsOf` retrieves stored historical content and must not be combined with `maxAgeHours`.

`contents.extras` accepts counts for `links`, `imageLinks`, `richLinks`, `richImageLinks`, and `codeBlocks`. Returned extras, subpages, entities, and unknown future fields are preserved.

## Dynamic Highlights (research preview)

The documented contract requires Exa's beta header for both `dynamic` and highlight `verbosity`. LiteLLM forwards caller-supplied `extra_headers`, including this header. It does not silently enroll requests in a beta.

```python
response = await litellm.asearch(
    query="recent advances in battery recycling",
    search_provider="exa_ai",
    contents={"highlights": {"dynamic": True, "verbosity": "medium"}},
    extra_headers={"Exa-Beta": "dynamic-highlights-2026-08-28"},
)
```

Dynamic Highlights allocates a shared context budget across results. `verbosity` accepts `low`, `medium`, or `high`. Do not combine `dynamic` with `maxCharacters`, or `verbosity` with `maxCharacters` or `numSentences`. Access and behavior remain subject to Exa's preview availability.

## Structured output

```python
response = litellm.search(
    query="recent advances in battery recycling",
    search_provider="exa_ai",
    contents={"highlights": True},
    systemPrompt="Use cited evidence. State when evidence is missing.",
    outputSchema={
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    },
)
output = response.model_dump()["output"]
print(output["content"])
print(output["grounding"])
```

`outputSchema` works across modes. Its root is `text` or `object`. Exa limits object schemas to 10 properties in total and two nesting levels, and arrays require `items`. Complete output and grounding are preserved without conversion. Missing output remains absent; present null or empty values remain present.

## Streaming

```python
stream = await litellm.asearch(
    query="recent advances in battery recycling",
    search_provider="exa_ai",
    contents={"highlights": True},
    outputSchema={"type": "text", "description": "A short cited answer"},
    stream=True,
)
async with stream:
    async for event in stream:
        print(event.model_dump())
print(stream.final_response.model_dump())
```

The synchronous equivalent uses `with litellm.search(..., stream=True) as stream` and `for event in stream`. Use the context manager or call `close()` / `await aclose()` when abandoning a stream early. Early close and cancellation record LiteLLM's per-query charge once, using the results received so far; they do not populate `final_response` without a `done` event.

Events preserve their native `type` and payload. The documented types are `text-delta`, `grounding`, `results`, `stream-reset`, `done`, and `error`. On `stream-reset`, discard previously accumulated synthesized text. `done.output` is authoritative. `final_response` becomes available after `done`, combining the most recent results with the terminal output and metadata. Provider error events, malformed events, and premature disconnects raise an exception and close the response. Heartbeats are ignored. SSE framing is decoded incrementally, including fragmented UTF-8 and multiline data.

Exa sends ordinary JSON for `stream=True` without an `outputSchema`. In that case the SDK returns a regular `SearchResponse`, not an iterator. The HTTP content type determines how the response is decoded, including when a server returns JSON despite an output schema. `complete_response=True` consumes an SSE stream and returns its final `SearchResponse`; a JSON response is returned directly. Completed streams use the same success callbacks and cost logging as other completed LiteLLM calls.

## Response compatibility

`response.results` retains LiteLLM's `title`, `url`, `snippet`, `date`, and `last_updated`. Snippets prefer nonempty text, then joined highlights, then summary. Every original Exa result field is also preserved, including text, highlights, scores, author, publication date, images, extras, subpages, and entities. Top-level `requestId`, `output`, `costDollars`, `searchTime`, legacy fields, and unknown extensions survive `model_dump()` and `model_dump_json()`.

`costDollars` is Exa's estimate, not an invoice. Preserving it does not change LiteLLM's separate cost calculation or billing policies.

## Deprecated fields

Legacy callers can still forward `startCrawlDate`, `endCrawlDate`, `includeText`, `excludeText`, and `context`. Exa documents crawl-date filters as ignored. Prefer publication windows for publication dates and `contents.snapshotAsOf` for stored historical content. Use text or highlights instead of context.

Nested legacy fields `contents.livecrawl`, `contents.crawledBeforeDate`, and highlight `numSentences` / `highlightsPerUrl` also pass through for compatibility. Prefer `maxAgeHours`, `snapshotAsOf`, and highlights defaults or `maxCharacters`, respectively. LiteLLM does not reinterpret deprecated fields or silently delete them.

## Validation

Run the offline adapter, fragmented-SSE, and real local HTTP tests with the repository's locked development environment.

```bash
uv run --no-sync pytest tests/unit/llms/exa_ai/search \
  tests/unit/llms/base_llm/search/test_streaming_iterator.py \
  tests/integration/sdk/test_exa_search_wire.py
```

To make paid live requests using an already configured `EXA_API_KEY`, run the opt-in matrix. It writes returned data and event sequences, but not credentials. It alternates synchronous and asynchronous calls and checks object output against its JSON schema.

```bash
uv run --no-sync python cookbook/exa_search/validate_live.py --output /tmp/exa-validation.json
```

Use `--case dynamic-highlights` or another case name to run a subset. Preview access, enterprise compliance, source availability, and best-effort extraction can vary between accounts and requests. Acceptance of a subpage or extras request does not guarantee those fields will be present for every source.
