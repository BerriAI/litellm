# OCR parity fixtures

The recording command runs four stages:

1. Generate deterministic SDK inputs for every configured OCR target
2. Build target-scoped, deduplicated recording jobs
3. Record upstream responses through one globally bounded worker pool
4. Persist each fixture and report whether it was recorded, cached, or failed

Run it with:

```shell
uv run python -m tests.test_litellm.ocr.fixtures.record --examples 4 --concurrency 4
```

`--concurrency` caps provider calls across all targets. Independent jobs finish after a failure, then the command exits
nonzero if any job failed

New recordings are VCR YAML cassettes. The committed corpus contains 31 migrated cassettes: 18 Mistral, 9 Reducto v3,
and 4 Reducto legacy. Their original response bytes, statuses, headers, and recording timestamps are preserved

To migrate an existing JSON fixture directory locally:

```shell
uv run python -m tests.test_litellm.ocr.fixtures.migrate --fixture-dir tests/test_litellm/ocr/fixtures/data
```

The migration replays each old response through the Python SDK to reconstruct missing requests, writes and validates
the YAML cassette, then removes its JSON predecessor. It calls only local recording/replay servers and needs no provider
credentials. Reconstructed requests are labeled `python_replay`; they are not historical wire captures. Filenames use
the current normalized SDK input hash, including the fixture contract

OCR strategies generate public `litellm.ocr()` and `litellm.aocr()` inputs. Every case contains the normalized model,
document, optional provider override, and LiteLLM keyword arguments. The fixture-only `contract` literal selects the
input schema and is removed before calling the SDK. Strategies never build provider wire payloads

Each contract has a required corpus containing a baseline and cases for its supported top-level OCR parameters. The
contracts are Mistral, Azure-hosted Mistral, Vertex-hosted Mistral, Azure Document Intelligence, Vertex DeepSeek,
Reducto v3, and Reducto legacy. Credentials and endpoints only control target discovery, so a machine records the
contracts it has configured and skips the rest

Reducto fixtures record upload and parse responses. Their parity cases remain non-strict expected failures until the
Rust OCR bridge supports Reducto. Azure and Vertex generation paths are unit-tested without credentials in CI, so the
committed corpus does not need live recordings for every target

Every recording target owns a small fixed provider-rejected corpus, independent of replay implementation support.
Those inputs are recorded separately from generated valid inputs. Local validation failures use no recorded response;
the parity suite checks those unsupported providers and models, malformed documents, invalid request formats, invalid
Azure Document Intelligence parameters, and invalid headers in sync and async SDK calls
