# Grafana Dashboard Brief

This dashboard should answer two questions:

1. How much e2e coverage do we have by module right now?
2. Is e2e coverage by module improving or regressing over time?

The dashboard should use plain **Coverage** language. Avoid exposing P0/P1/P2 in the
default view; tiers can be added later as a filter or drilldown.

## Panels

### Coverage by Module

Use a bar chart or table with one row per module:

- Core LLMs
- Non-Core LLMs
- MCPs
- Management/UI
- Reliability & Performance
- Logging & Guardrails
- Other

Each row should show:

- `covered`
- `total`
- `coverage_percent`

Formula:

```text
coverage_percent = covered / total * 100
```

### Coverage Trend by Module

Use a time-series chart with one line per module.

- X-axis: CI run timestamp, scrape timestamp, or pushed metric timestamp
- Y-axis: `coverage_percent`
- Series label: module name

This shows whether coverage is improving across dates.

## Data Contract

Generate coverage data from the registry collector:

```bash
cd tests/e2e
PYTHONPATH=. python -m coverage_registry.collector --format prometheus --strict
```

For artifact-based jobs, JSON is also available:

```bash
cd tests/e2e
PYTHONPATH=. python -m coverage_registry.collector --format json --strict
```

Prometheus metrics:

```text
litellm_e2e_coverage_cells{module="<module>",state="covered"} <count>
litellm_e2e_coverage_cells{module="<module>",state="total"} <count>
litellm_e2e_coverage_percent{module="<module>"} <percent>
litellm_e2e_coverage_orphan_markers <count>
litellm_e2e_coverage_collection_errors <count>
```

Grafana gets the trend by storing these metrics over time. No date needs to be encoded
inside the metric itself.

## Alerts

Start with two alerts:

- Unknown marker count is greater than zero.
- Coverage percent for any module drops compared with the previous successful run.

After existing collection warnings are fixed, add:

- Collection error count is greater than zero.

## Non-Goals

- Do not use line coverage for this dashboard. This is behavior coverage, not source-line
  coverage.
- Do not let tests invent module names. Tests only declare `@pytest.mark.covers(...)`;
  the registry decides which module a cell belongs to.
