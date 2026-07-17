# Grafana: e2e failure alert -> Slack (Devin remediation)

This directory holds the alerting-as-code that drives remediation off the e2e
suite going red. It replaces the old in-runner `bob_the_builder.py`
`pytest_sessionfinish` hook, which called Devin through the proxy's own
`/mcp-rest` gateway to open fix PRs.

## Why this instead of an in-runner hook

The old hook had three problems that this design removes.

It was a circular dependency. The suite goes red most often when the proxy is
unhealthy, which is exactly when the hook's call back to
`PROXY_BASE_URL/mcp-rest` also failed. It swallowed every exception, so a
degraded proxy produced no remediation and no signal at all. This alert instead
evaluates the `E2E_RESULT` lines the harness already ships to Loki, so it fires
even when the runner or proxy is degraded.

It put remediation orchestration inside the test runner. The runner's job is to
run tests and report results; routing a red result to whoever fixes it is
alerting infrastructure. Grafana Alerting is the standard place for that.

It hand-rolled dedup with a sha256 over the failing node ids. Grafana already
groups and suppresses repeat notifications. The rule's `notification_settings`
group by `package` and re-notify only every `repeat_interval`, so one ongoing
failing package produces one Slack thread, not a new call per run.

## The signal

`tests/e2e/conftest.py` (`pytest_runtest_makereport`) prints one logfmt line per
finished test via `e2e_result_reporter.py`:

```
E2E_RESULT package=logging file=test_langfuse_e2e.py outcome=failed duration_ms=1500 node_id="logging/..." covers=cell.id
```

Loki ingests these under `{service_name="litellm-e2e"}`. See
`../status_history_panels.md` for the status-history dashboard that reads the
same signal.

## What the rule does

`alert_rules.yaml` defines one Grafana-managed rule, `litellm-e2e-suite-failure`,
evaluated every 1m over a 5m lookback against the Loki datasource:

```logql
sum by (package) (
  count_over_time({service_name="litellm-e2e"} |= "E2E_RESULT" | logfmt | outcome=~"failed|error" [5m])
)
```

Query `A` counts failed/errored lines per package over the window, `B` reduces to
the last value per series, and `C` thresholds `> 0`. A package with any failure
in the window fires an instance labelled with that `package`. The instance
annotations carry a ready-to-paste Loki drill-down query for the exact node ids.
The 1m interval sits well inside the 5m lookback, so each failing line is seen by
several consecutive evaluations and a boundary or late-ingested line is never
missed.

The `{service_name="litellm-e2e"}` stream selector has to match the label the
scrape actually attaches to the e2e runner's stdout, or the query matches nothing
and the rule silently never fires. If your scrape does not set `service_name`,
swap the selector for the pod-based fallback the status-history panels already
use (`{cluster="berrie-litellm-stage", pod=~"litellm-e2e-.+"}`, see
`../status_history_panels.md`) in both `alert_rules.yaml` and the drill-down
annotation.

`contact_points.yaml` defines the `devin-e2e-remediation` Slack contact point.
The rule routes straight to it through `notification_settings.receiver` (Grafana
simplified routing, v11+), so no change to the org's root notification policy
tree is needed, and none is provisioned here to avoid clobbering it. On older
Grafana, drop `notification_settings` from the rule and add a matching child
route (`service = litellm-e2e` -> `devin-e2e-remediation`) under the existing
root policy instead. Devin watches the target Slack channel and opens the tickets
and fix PRs that `bob_the_builder` used to open directly.

## Applying it

These are standard Grafana file-provisioning documents. Point Grafana at this
directory (or copy the two YAML files into the provisioning path) so both load:

```
provisioning/alerting/contact_points.yaml
provisioning/alerting/alert_rules.yaml
```

Two values are environment-supplied so no secret or environment-specific id
lands in git. Grafana expands `$__env{VAR}` in provisioning files at load time:

- `SLACK_E2E_WEBHOOK_URL`: incoming-webhook URL for the channel Devin listens on
- `LOKI_DATASOURCE_UID`: uid of the Loki datasource that holds `E2E_RESULT`

If you manage the Grafana Cloud stack (`berriai.grafana.net`) with Terraform
instead, the same three objects map to `grafana_contact_point` and
`grafana_rule_group`; keep this YAML as the reference for the query, grouping,
and routing.
