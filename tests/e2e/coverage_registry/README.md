# e2e coverage metadata

E2E coverage is declared directly on pytest tests. There are no coverage YAML
files.

Each collected pytest under `tests/e2e` must have:

```python
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.e2e_coverage(
        module="core_llms",
        endpoint="/chat/completions",
        provider="openai",
        params=["tools", "streaming"],
    ),
]
```

Use a module-level `pytestmark` when every test in a file covers the same
surface. Use a per-test marker when a file mixes endpoints, providers, or params.

## Required fields

- `module`: one of the known dashboard modules in `schema.py`
- `endpoint`: a known endpoint or surface, such as `/chat/completions`,
  `/v1/messages`, `/v1/batches`, `/budget/*`, or `/spend/*`
- `provider`: a known provider/integration, such as `proxy`, `openai`,
  `anthropic`, `vertex_ai`, `prometheus`, or `multiple`
- `params`: one or more explicit lowercase parameter/behavior names

## How coverage is measured

The collector runs pytest in collect-only mode, validates `e2e_coverage` markers,
and expands each marker into unique:

```text
module x endpoint x provider x param
```

coverage units.

The report shows:

- unique coverage units by module
- collected pytest test count by module
- endpoint/provider/param breakdowns for Grafana tables
- missing or invalid metadata counts for CI

This answers questions like:

```text
How many /chat/completions rate-limit params do we exercise?
How many budget tests exist?
Which modules gained or lost endpoint coverage over time?
```

## Commands

Render the report:

```bash
cd tests/e2e
PYTHONPATH=. python -m coverage_registry.collector
```

Emit Loki lines after the e2e job:

```bash
cd tests/e2e
PYTHONPATH=. python -m coverage_registry.collector --format loki --strict
```

Validate every collected pytest has metadata:

```bash
cd tests/e2e
PYTHONPATH=. python -m coverage_registry.check_coverage_sync
```

CI runs the sync check in `.github/workflows/test-code-quality.yml`.

## Adding coverage

1. Add the pytest.
2. Add `@pytest.mark.e2e_coverage(...)` or module-level `pytestmark`.
3. Pick the closest module, endpoint, provider, and params.
4. If the endpoint or provider is real but rejected, add it to `schema.py`.
5. Run `PYTHONPATH=. python -m coverage_registry.check_coverage_sync` from
   `tests/e2e`.
