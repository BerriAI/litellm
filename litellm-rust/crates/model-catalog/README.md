# Model catalog

`litellm-model-catalog` builds an immutable snapshot from caller supplied JSON bytes. It has no network, Python, registration, or refresh behavior. The caller supplies optional source, revision, and ETag provenance. Parse and validation are separate so small synthetic catalogs can use explicit integrity limits

The parser treats `sample_spec` and `fallback_generalizations` as reserved top level metadata. `fallback_rules()` exposes the typed rule array when present; this crate does not execute regex generalizations. Model entries retain all JSON fields except `aliases`, including unknown fields. `field()` returns `None` for an absent key and a JSON null, false, or zero value for a present key. The returned values are borrowed, so callers cannot mutate the snapshot

Each entry also deserializes into `ModelInfo`, a typed mirror of `model_prices_and_context_window.schema.json`'s `modelEntry` definition, reachable via `ModelEntry::info()`. All schema fields are optional on `ModelInfo`, including `litellm_provider` which the schema marks required, so small synthetic catalogs still parse. Unknown fields are not part of `ModelInfo`; they remain on `fields()`. Building with the `schema` feature adds `schemars` derives and exposes `model_entry_json_schema()` for emitting the entry's JSON Schema. Parse and validation failures are reported by the `Error` enum in `error.rs`, while catalog logic lives in `catalog.rs`

The integration tests read the repository's catalog and schema files at test time, assert every entry round-trips through `ModelInfo`, and verify that the generated schema's properties match the repository schema

Aliases point to their canonical entries. An alias that exactly matches any canonical key is skipped; the first canonical entry claiming an alias wins. Invalid alias lists and nonstring names are skipped and reported by `alias_issues()`. Exact lookup wins. For a case insensitive miss, the last key with the same lowercase spelling wins, following Python's lowercase map built after aliases are appended. This uses Rust Unicode lowercasing, which can differ from Python for unusual Unicode model IDs

`validate()` counts canonical entries before alias expansion and excludes both reserved keys. It enforces an explicit minimum and backup shrink ratio, with Python defaults of 50 models and 0.5. Parsing rejects nonobject model entries and known fields with the wrong JSON type, but ignores unknown fields. It does not enforce every constraint in the JSON schema, calculate prices, resolve providers, or check provenance authenticity. The caller decides how to handle validation failures

This snapshot does not represent Python's live mutable `litellm.model_cost`, nested dict and list mutation, or mutation of dicts previously returned by Python APIs. It has no bridge or runtime integration

## Benchmarks

`cargo bench -p litellm-model-catalog --bench catalog` measures parsing plus alias indexing and exact lookup. For a local Python baseline on the same fixture, use:

```sh
python3 -m timeit -s 'import json, pathlib; body = pathlib.Path("../model_prices_and_context_window.json").read_bytes()' 'json.loads(body)'
```

Run these commands from `litellm-rust`. Python's command measures JSON loading only, without alias expansion or snapshot construction. The Rust benchmark does not include future Python object materialization, so these numbers are not an end to end runtime comparison
