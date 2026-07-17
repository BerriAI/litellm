# Grafana: package status history for e2e

Dashboard: [LiteLLM E2E](https://berriai.grafana.net/d/mup2cfn/litellm-e2e) (`mup2cfn`).

The old **test suite status history** panel scraped pytest progress lines and
grouped by **file basename** (`test_foo.py`). That does not scale: multi-class
files collapse to one bit, and full `node_id` cardinality melts status-history.

## Artifact

The run emits a standard pytest JUnit XML report instead of a bespoke
`E2E_RESULT` log line. Point pytest at a file with `--junitxml`:

```
uv run pytest tests/e2e --junitxml=e2e-report.xml
```

Each finished test is one `<testcase>`; pytest fills in `classname`, `name`,
`time` (duration in seconds), and a child `<failure>`, `<error>`, or `<skipped>`
element for a non-passing outcome (a bare `<testcase>` is a pass). The two custom
signals ride along as `<property>` entries, attached to every test in
`conftest.py::pytest_collection_modifyitems` (logic in `junit_properties.py`):

```xml
<testcase classname="logging.test_langfuse_e2e" name="TestX.test_y" time="1.5">
  <properties>
    <property name="package" value="logging" />
    <property name="covers" value="logging.langfuse.success.logs_spend" />
  </properties>
  <failure message="assert ...">...</failure>
</testcase>
```

- `package`: top-level suite dir under `tests/e2e/` (`root` for top-level files),
  normalized so a repo-root run and a suite-cwd run agree
- `covers`: comma-joined `@pytest.mark.covers` cell ids (empty string when none)

Outcome maps from the child element: `<failure>` -> failed, `<error>` -> error,
`<skipped>` -> skipped, none -> passed. Duration is the `time` attribute

## Shipping to Loki

JUnit XML is not line-based, so it is not tailed directly the way the old logfmt
line was. Ship it with a thin converter in the e2e job (Grafana Agent / promtail
cannot parse XML on their own): after the run, walk `e2e-report.xml` and print one
logfmt line per `<testcase>` to the pod's stdout, which the existing scrape
already forwards to Loki. A short `xml.etree` step is enough; the emitted line
should carry `package`, `covers`, `outcome`, `duration_ms`, and `node_id`
(`classname::name`) so the queries below keep working unchanged. Building that
converter and its scrape config is infra-side and out of scope for this repo
change

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

Value mappings: `0` -> Pass (green), `1` -> Fail (red).

If `service_name` is missing on older scrapes, use:

```logql
{cluster="berrie-litellm-stage", pod=~"litellm-e2e-.+"}
```

instead of `{service_name="litellm-e2e"}`.

## Panel: failed tests (logs drill-down)

```logql
{service_name="litellm-e2e"} |= "E2E_RESULT" | logfmt | outcome=~"failed|error"
```

Show fields: `package`, `covers`, `node_id`, `duration_ms`.

## Panel (optional): filter by package variable

Dashboard variable `package` (custom or from label_values on the shipped lines):

```logql
{service_name="litellm-e2e"} |= "E2E_RESULT" | logfmt | package=`$package` | outcome=~"failed|error"
```

## Do not

- Put full `node_id` as the status-history series key (cardinality).
- Rely on `::S+ PASSED` progress regex as the primary signal.
