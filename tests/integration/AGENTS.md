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
handler. A leaked field is invisible from the response. `monkeypatch.setenv` is fine. Patching our own
function in a full stack is not

## Where it goes

By the domain a user would name: `pricing`, `spend`, `routing`, `mcp`. A file only needs to live in a
directory that a `GROUPS` entry in `run.py` selects. There is no manifest and no `covers` marker on new
tests. A product bug the test exposes is `pytest.skip("BUG: <symptom>")` at the top of the body, not a
fix in the test and not a deletion. Needs no network, proxy, DB or Redis: `tests/unit`

CircleCI's integration groups run every file here except the OCI and sandbox tests, which no CI runs,
and the files in `GITHUB_FILES` in `run.py`. CircleCI runs on main, not on pull requests, so a new file
here first runs after its PR merges. `GITHUB_FILES` is only for a file that needs what CircleCI's groups
cannot give it, like the images `image-scan.yml` builds or the plain pytest run CircleCI's `roi-database`
job gives `test_roi_observed.py`. Add a file there only in that case, together with its dedicated job,
and list a new dedicated GitHub workflow in `DEDICATED_INTEGRATION_WORKFLOWS` in
`.github/scripts/assert_ci_coverage.py`. The `ci-coverage` check fails a `GITHUB_FILES` file that any
other workflow runs, like a shared `test-unit.yml` shard (#46780)
