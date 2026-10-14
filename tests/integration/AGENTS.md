# tests/integration

Real proxy, Postgres, Redis, scripted upstream. `README.md` has shards and CI wiring

## What good looks like

Poll async spend rows, never sleep. Assert request bodies at the upstream because proxy responses can hide leaks.
`monkeypatch.setenv` is fine, but don't patch LiteLLM functions in integration tests

```python
rows = eventually(
    lambda: read_rows('SELECT spend FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (body["id"],)),
    lambda values: len(values) == 1,
    seconds=70,
)
assert float(rows[0]["spend"]) == pytest.approx(10 * 0.001 + 5 * 0.0001 + 7 * 0.002 + 4 * 0.002)
```

## Where it goes

Group by user-facing domain: `pricing`, `spend`, `routing`, `mcp`. Put files in a `GROUPS` directory in
`run.py`, with no manifest or `covers` marker. After the 2026-10-14 Responses `input_file` conversion
dropped pricing coverage, preserve distinct route/request-shape coverage and check for duplicates. A product bug
uses `pytest.skip("BUG: <symptom>")` at the top, never a test fix or deletion. Tests needing no network,
proxy, DB, or Redis belong in `tests/unit`
