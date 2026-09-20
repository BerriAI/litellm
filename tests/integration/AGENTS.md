# tests/integration

Real proxy, Postgres, Redis, scripted upstream. `README.md` has shards and CI wiring

## What good looks like

The root example is from here. The spend row lands async: poll, never sleep

```python
rows = eventually(
    lambda: read_rows('SELECT spend FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (body["id"],)),
    lambda values: len(values) == 1,
    seconds=70,
)
assert float(rows[0]["spend"]) == pytest.approx(10 * 0.001 + 5 * 0.0001 + 7 * 0.002 + 4 * 0.002)
```

`sleep(3)` fails on a slow runner and taxes every fast one. Assert the outbound body in the upstream
handler; a leaked field is invisible from the response. `monkeypatch.setenv` is fine; patching our own
function in a full stack is not

## Where it goes

By the domain a user would name: `pricing`, `spend`, `routing`. Add the node and its `covers` ids to
`contracts.json` or collection fails. Needs no proxy, DB or Redis: `tests/unit`
