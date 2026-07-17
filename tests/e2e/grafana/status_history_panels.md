# Grafana: package status history for e2e

Dashboard: [LiteLLM E2E](https://berriai.grafana.net/d/mup2cfn/litellm-e2e) (`mup2cfn`).

The old **test suite status history** panel scraped pytest progress lines and
grouped by **file basename** (`test_foo.py`). That does not scale: multi-class
files collapse to one bit, and full `node_id` cardinality melts status-history.

## Emitter

After each test finishes, `tests/e2e/conftest.py` prints one logfmt line:

```
E2E_RESULT package=logging file=test_langfuse_e2e.py outcome=failed duration_ms=1500 node_id="logging/..." covers=cell.id
```

## Panel: package status history (replace panel 11)

**Type:** Status history  
**Interval:** 15m (or 1h for multi-day ranges)  
**Description:** Per top-level package under `tests/e2e/`: red if any test failed or errored in the bucket.

```logql
max by (package) (
  max_over_time(
    {service_name="litellm-e2e"}
      |= "E2E_RESULT"
      | logfmt
      | outcome != ""
      | label_format result=`{{ if or (eq .outcome "failed") (eq .outcome "error") }}1{{ else }}0{{ end }}`
      | unwrap result
      [$__interval]
  )
)
```

Value mappings: `0` → Pass (green), `1` → Fail (red).

If `service_name` is missing on older scrapes, use:

```logql
{cluster="berrie-litellm-stage", pod=~"litellm-e2e-.+"}
```

instead of `{service_name="litellm-e2e"}`.

## Panel: failed tests (logs drill-down)

```logql
{service_name="litellm-e2e"} |= "E2E_RESULT" | logfmt | outcome=~"failed|error"
```

Show fields: `package`, `file`, `node_id`, `covers`, `duration_ms`.

## Panel (optional): filter by package variable

Dashboard variable `package` (custom or from label_values on E2E_RESULT):

```logql
{service_name="litellm-e2e"} |= "E2E_RESULT" | logfmt | package=`$package` | outcome=~"failed|error"
```

## Do not

- Put full `node_id` as the status-history series key (cardinality).
- Rely on `::S+ PASSED` progress regex as the primary signal once E2E_RESULT is live.
