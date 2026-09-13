# OCR parity fixtures

The recording command runs four stages:

1. Generate deterministic SDK inputs for every configured OCR target
2. Build target-scoped, deduplicated recording jobs
3. Record upstream responses through one globally bounded worker pool
4. Persist each fixture and report whether it was recorded, cached, or failed

Run it with:

```shell
uv run python -m tests.rust-python-harness.strategies.e2e_parity.sdk.ocr.fixtures.record --examples 1000
```

`--concurrency` defaults to 2 and caps active recording jobs across all targets. Increase it explicitly when provider
quotas permit. Concurrency limits do not guarantee a request-per-minute quota; HTTP 408, 429, and 5xx responses fail
recording without being saved. Rerunning retries missing fixtures and reuses successful recordings

New recordings are VCR YAML cassettes. The corpus retains the original 31 migrated cassettes and adds live recordings
for all four providers. Original response bytes, statuses, headers, and recording timestamps are preserved

For Vertex, authenticate and select a project once:

```shell
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

The recording command reads the project from `VERTEXAI_PROJECT`, `VERTEX_PROJECT`, or the active gcloud configuration,
then gets an OAuth access token with `gcloud auth print-access-token`. Tokens stay in memory and are removed from
recorded headers. `VERTEX_AI_ACCESS_TOKEN` or the legacy `VERTEX_AI_API_KEY` can override token lookup. The
`VERTEXT_API_KEY` express-mode key is not used as a Bearer token. Mistral defaults to `us-central1`; DeepSeek defaults
to the global host and `global` location. `VERTEX_DEEPSEEK_LOCATION` and `VERTEX_DEEPSEEK_API_BASE` override the latter

Azure accepts `AZURE_KEY` and `AZURE_ENDPOINT` as fallbacks for both Azure OCR contracts. Provider-specific variables
take precedence. Set `AZURE_DEPLOYMENT_NAME=mistral-ocr-4-0` to record that deployment instead of the default
`mistral-document-ai-2512`. Mistral and Reducto use `MISTRAL_API_KEY` and `REDUCTO_API_KEY`

To migrate an existing JSON fixture directory locally:

```shell
uv run python -m tests.rust-python-harness.strategies.e2e_parity.sdk.ocr.fixtures.migrate --fixture-dir tests/rust-python-harness/strategies/e2e_parity/sdk/ocr/fixtures/data
```

The migration replays each old response through the Python SDK to reconstruct missing requests, writes and validates
the YAML cassette, then removes its JSON predecessor. It calls only local recording/replay servers and needs no provider
credentials. Reconstructed requests are labeled `python_replay`; they are not historical wire captures. Filenames use
the current normalized SDK input hash, including the fixture contract

OCR strategies generate public `litellm.ocr()` and `litellm.aocr()` inputs. Every case contains the normalized model,
document, optional provider override, and LiteLLM keyword arguments. The fixture-only `contract` literal selects the
input schema and is removed before calling the SDK. Strategies never build provider wire payloads

Each contract's strategy contains baselines and cases for its supported top-level OCR parameters. The
contracts are Mistral, Azure-hosted Mistral, Vertex-hosted Mistral, Azure Document Intelligence, Vertex DeepSeek,
Reducto v3, and Reducto legacy. Credentials and endpoints only control target discovery, so a machine records the
contracts it has configured and skips the rest

`--examples 1000` exhausts the current finite strategies: 32 Mistral, 15 Azure Mistral, 17 Azure Document Intelligence,
16 Vertex Mistral, 2 Vertex DeepSeek, 55 Reducto v3, and 3 Reducto legacy cases, including fixed rejected inputs.
This covers the defined strategy choices, not every possible value accepted by the schemas. The live run recorded
139 of these 140 cases. Vertex Mistral's standalone `document_annotation_format` case repeatedly returned HTTP 500
and remains pending. Its bounding-box annotation, annotation-prompt, and confidence cases record upstream 404/422
rejections; schema acceptance does not imply support by the hosted model

Reducto fixtures record upload and parse responses. Their parity cases remain non-strict expected failures until the
Rust OCR bridge supports Reducto. Azure and Vertex generation paths are unit-tested without credentials in CI, so the
committed corpus does not need live recordings for every target

Every recording target owns a small fixed provider-rejected corpus, independent of replay implementation support.
Those inputs are recorded separately from generated valid inputs. Local validation failures use no recorded response;
the parity suite checks those unsupported providers and models, malformed documents, invalid request formats, invalid
Azure Document Intelligence parameters, and invalid headers in sync and async SDK calls
