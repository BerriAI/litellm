package main

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

// bifrostSide is the side and key set name genconfig writes as a Bifrost config. Bifrost records
// requests in its own schema (the logs and governance_budgets tables, bifrost_* metrics, its own
// S3 log objects), so its records are mapped onto the same checks; the levels and records it has
// no counterpart for are reported as not applicable rather than failed.
const bifrostSide = "bifrost"

func isBifrost(side string) bool { return side == bifrostSide }

// bifrostQuery selects the Bifrost key set's records as the same JSON row query() selects for a
// LiteLLM side. It runs against Bifrost's logs database. Log rows are found by virtual key id,
// budgets by the ids genconfig gives them; Bifrost's customer stands in for LiteLLM's user, and
// the open-source build has no users, team members or daily tables.
func bifrostQuery(set string) string {
	prefix := strings.ReplaceAll(setPrefix(set), "'", "''")
	return fmt.Sprintf(`WITH s AS (SELECT * FROM logs WHERE virtual_key_id LIKE '%[1]s-%%'),
g AS (SELECT coalesce(routing_rule_name, '') AS model_group, count(*) AS rows, sum(prompt_tokens) AS p,
             sum(completion_tokens) AS c, sum(cost)::float8 AS sp FROM s GROUP BY 1),
-- Each deployment is its own custom provider, named like the mock's /d/<name> path.
d AS (SELECT provider AS name, count(*) AS rows,
             sum(prompt_tokens::float8 * prompt_tokens) AS pp, sum(prompt_tokens::float8 * completion_tokens) AS pc,
             sum(completion_tokens::float8 * completion_tokens) AS cc, sum(cost::float8 * prompt_tokens) AS sp,
             sum(cost::float8 * completion_tokens) AS sc
      FROM s GROUP BY 1),
b AS (SELECT id, current_usage::float8 AS usage FROM governance_budgets WHERE id LIKE 'b-%[1]s-%%')
SELECT json_build_object(
  'dialect', 'bifrost',
  'rows', (SELECT count(*) FROM s),
  'success_rows', (SELECT count(*) FROM s WHERE status = 'success'),
  'prompt_tokens', (SELECT coalesce(sum(prompt_tokens), 0) FROM s),
  'completion_tokens', (SELECT coalesce(sum(completion_tokens), 0) FROM s),
  'spend', (SELECT coalesce(sum(cost), 0)::float8 FROM s),
  'zero_spend_rows', (SELECT count(*) FROM s WHERE coalesce(cost, 0) <= 0),
  'rows_with_team', (SELECT count(*) FROM s WHERE team_id LIKE '%[1]s-team-%%'),
  'rows_with_user', (SELECT count(*) FROM s WHERE customer_id = '%[1]s-customer'),
  'rows_with_model_id', (SELECT count(*) FROM s WHERE coalesce(provider, '') <> ''),
  -- logging_headers copies the raw header values into metadata.
  'rows_with_spend_metadata', (SELECT count(*) FROM s
     WHERE metadata::jsonb->>'x-cost-center' = '%[1]s'
       AND (metadata::jsonb->>'x-litellm-spend-logs-metadata')::jsonb->>'run' = '%[1]s'),
  'groups', (SELECT coalesce(json_object_agg(model_group, json_build_object('rows', rows,
       'prompt_tokens', p, 'completion_tokens', c, 'spend', sp)), '{}') FROM g),
  'deployments', (SELECT coalesce(json_object_agg(name, json_build_object('rows', rows, 'pp', pp, 'pc', pc,
       'cc', cc, 'sp', sp, 'sc', sc)), '{}') FROM d),
  'keys_spend', (SELECT coalesce(sum(usage), 0) FROM b WHERE id ~ '^b-%[1]s-[0-9]+$'),
  'teams_spend', (SELECT coalesce(sum(usage), 0) FROM b WHERE id LIKE 'b-%[1]s-team-%%'),
  'users_spend', (SELECT coalesce(sum(usage), 0) FROM b WHERE id = 'b-%[1]s-customer')
);
`, prefix)
}

func notApplicable(r *report, name, side, why string) {
	r.add(check{Name: name, Side: side, Want: "-", Got: "n/a", OK: true, Note: why})
}

func checkBifrostDB(r *report, side string, e expected, db dbTotals) {
	ok := e.Total.Requests
	r.add(info("cross-replica state", side, "none: open-source Bifrost keeps rate limits, budgets and "+
		"counters in each replica's memory (written to Postgres every 10 s), with no Redis or shared store"))
	exact(r, "log rows", side, ok, db.Rows)
	exact(r, "log rows marked success", side, ok, db.Success)
	exact(r, "log prompt tokens", side, e.Total.Prompt, db.Prompt)
	exact(r, "log completion tokens", side, e.Total.Completion, db.Completion)
	exact(r, "log rows priced at $0", side, 0, db.ZeroSpend)
	exact(r, "log rows naming the key's team", side, ok, db.WithTeam)
	exact(r, "log rows naming the key's customer", side, ok, db.WithUser)
	notApplicable(r, "log rows naming the key's user", side, "users are enterprise-only")
	exact(r, "log rows naming the deployment", side, ok, db.WithModel)
	exact(r, "log rows with request spend metadata", side, ok, db.Metadata)
	notApplicable(r, "log rows with key spend metadata", side, "virtual keys carry no metadata")
	for _, name := range sortedKeys(e.Groups) {
		want := e.Groups[name]
		got := db.Groups[name]
		if got == nil {
			got = &groupTotals{}
		}
		r.add(check{Name: "rows / prompt / completion for " + name, Side: side,
			Want: fmt.Sprintf("%d / %d / %d", want.Requests, want.Prompt, want.Completion),
			Got:  fmt.Sprintf("%d / %d / %d", got.Rows, got.Prompt, got.Completion),
			OK:   want.Requests == got.Rows && want.Prompt == got.Prompt && want.Completion == got.Completion})
	}
	dollars(r, "key budget usage = logs", side, db.Spend, db.KeySpend)
	dollars(r, "team budget usage = logs", side, db.Spend, db.TeamSpend)
	dollars(r, "customer budget usage = logs", side, db.Spend, db.UserSpend)
	notApplicable(r, "user spend = spend logs", side, "users are enterprise-only")
	notApplicable(r, "team member spend = spend logs", side, "team members are enterprise-only")
	notApplicable(r, "daily user and team tables", side, "Bifrost keeps no daily rollups")
}

// checkBifrostMetrics maps the LiteLLM counters onto Bifrost's telemetry plugin. Bifrost counts
// requests by outcome and upstream attempts, which retries can only add to.
func checkBifrostMetrics(r *report, side string, e expected, db dbTotals, m map[string]*family) {
	value := func(name string) (float64, bool) { return sample(m, name) }
	success, okS := value("bifrost_success_requests_total")
	failed, _ := value("bifrost_error_requests_total")
	r.add(check{Name: "metric bifrost_success_requests_total + bifrost_error_requests_total", Side: side,
		Want: fmt.Sprint(e.Sent), Got: present(okS, fmt.Sprintf("%.0f", success+failed)), OK: okS && success+failed == float64(e.Sent)})
	counts := []struct {
		name string
		want int64
	}{
		{"bifrost_success_requests_total", e.Total.Requests},
		{"bifrost_input_tokens_total", e.Total.Prompt},
		{"bifrost_output_tokens_total", e.Total.Completion},
	}
	for _, c := range counts {
		got, ok := value(c.name)
		r.add(check{Name: "metric " + c.name, Side: side, Want: fmt.Sprint(c.want),
			Got: present(ok, fmt.Sprintf("%.0f", got)), OK: ok && got == float64(c.want)})
	}
	for _, name := range []string{"bifrost_upstream_requests_total", "bifrost_upstream_latency_seconds_count"} {
		got, ok := value(name)
		r.add(check{Name: "metric " + name, Side: side, Want: fmt.Sprintf(">= %d", e.Total.Requests),
			Got: present(ok, fmt.Sprintf("%.0f", got)), OK: ok && got >= float64(e.Total.Requests)})
	}
	got, ok := value("bifrost_cost_total")
	r.add(check{Name: "metric bifrost_cost_total = logs", Side: side,
		Want: fmt.Sprintf("$%.6f", db.Spend), Got: present(ok, fmt.Sprintf("$%.6f", got)), OK: ok && near(db.Spend, got)})
}

// Bifrost's S3 objects hold the log row with its JSON columns as strings, which s3stub
// -max-string may cut short, so token counts are read from the start of token_usage.
var (
	bifrostPrompt     = regexp.MustCompile(`"prompt_tokens":(\d+)`)
	bifrostCompletion = regexp.MustCompile(`"completion_tokens":(\d+)`)
)

func (t *s3Totals) addBifrost(p map[string]any) {
	text := func(k string) string { s, _ := p[k].(string); return s }
	filled := func(keys ...string) bool {
		for _, k := range keys {
			if s := text(k); s != "" && s != "[]" && s != "null" {
				return true
			}
		}
		return false
	}
	if text("status") == "success" {
		t.Success++
	}
	usage := text("token_usage")
	for _, f := range []struct {
		re  *regexp.Regexp
		sum *int64
	}{{bifrostPrompt, &t.Prompt}, {bifrostCompletion, &t.Completion}} {
		if m := f.re.FindStringSubmatch(usage); m != nil {
			n, _ := strconv.ParseInt(m[1], 10, 64)
			*f.sum += n
		}
	}
	cost, _ := strconv.ParseFloat(text("cost"), 64)
	t.Cost += cost
	if filled("input_history", "responses_input_history") && filled("output_message", "responses_output") {
		t.WithBodies++
	}
	if strings.HasPrefix(text("team_id"), setPrefix(bifrostSide)+"-team-") {
		t.WithTeam++
	}
	var meta map[string]any
	if json.Unmarshal([]byte(text("metadata")), &meta) == nil {
		for k := range meta {
			t.Metadata[k] = true
		}
	}
}

func checkBifrostLogs(r *report, side string, e expected, db dbTotals, t *s3Totals) {
	ok := e.Total.Requests
	exact(r, "S3 request logs", side, ok, t.Objects)
	if t.Read < t.Objects {
		r.add(check{Name: "S3 logs read", Side: side, Want: fmt.Sprint(t.Objects), Got: fmt.Sprint(t.Read),
			OK: true, Note: fmt.Sprintf("totals below cover the first %d", t.Read)})
		return
	}
	exact(r, "S3 logs with status success", side, ok, t.Success)
	exact(r, "S3 logs with input and output", side, ok, t.WithBodies)
	exact(r, "S3 logs naming the key's team", side, ok, t.WithTeam)
	exact(r, "S3 log prompt tokens", side, e.Total.Prompt, t.Prompt)
	exact(r, "S3 log completion tokens", side, e.Total.Completion, t.Completion)
	dollars(r, "S3 log cost = logs", side, db.Spend, t.Cost)
}
