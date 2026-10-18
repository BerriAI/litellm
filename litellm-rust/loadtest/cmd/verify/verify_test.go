package main

import (
	"strings"
	"testing"
)

func TestParseMetricsSumsSeriesAndKeepsLabelNames(t *testing.T) {
	text := `# HELP litellm_spend_metric_total Spend
# TYPE litellm_spend_metric_total counter
litellm_spend_metric_total{model="a",team="x,y"} 1.5
litellm_spend_metric_total{model="b",end_user="q\"z"} 2.25 1700000000
litellm_requests 3
`
	out := map[string]*family{}
	if err := parseMetrics(strings.NewReader(text), out); err != nil {
		t.Fatal(err)
	}
	if got := out["litellm_spend_metric_total"].sum; got != 3.75 {
		t.Fatalf("sum = %v", got)
	}
	labels := out["litellm_spend_metric_total"].labels
	for _, want := range []string{"model", "team", "end_user"} {
		if !labels[want] {
			t.Fatalf("label %s missing from %v", want, labels)
		}
	}
	if out["litellm_requests"].sum != 3 {
		t.Fatalf("unlabelled sample not read")
	}
}

func TestChecksPassOnlyWhenRecordsMatch(t *testing.T) {
	e := expected{Sent: 2, Total: tokens{Requests: 2, Prompt: 20, Completion: 10},
		Groups: map[string]*tokens{"g": {Requests: 2, Prompt: 20, Completion: 10}}}
	day := daily{Spend: 0.5, Requests: 2, Successful: 2, Prompt: 20, Completion: 10}
	db := dbTotals{Rows: 2, Success: 2, Prompt: 20, Completion: 10, Spend: 0.5, WithTeam: 2, WithUser: 2,
		WithModel: 2, Metadata: 2, Groups: map[string]*groupTotals{"g": {Rows: 2, Prompt: 20, Completion: 10, Spend: 0.5}},
		KeySpend: 0.5, TeamSpend: 0.5, UserSpend: 0.5, MemberSpend: 0.5, DailyUser: day, DailyTeam: day}
	var r report
	checkDB(&r, "rust", e, db)
	if n := r.failed(); n != 0 {
		t.Fatalf("%d checks failed:\n%s", n, r.markdown())
	}
	db.TeamSpend = 0.25
	db.Groups["g"].Completion = 9
	r = report{}
	checkDB(&r, "rust", e, db)
	if n := r.failed(); n != 2 {
		t.Fatalf("want 2 failures, got %d:\n%s", n, r.markdown())
	}
}

func TestQueryScopesToTheKeySet(t *testing.T) {
	q := query("rust")
	for _, want := range []string{"key_alias LIKE 'loadtest-rust-%'", `'run' = 'loadtest-rust'`, `"LiteLLM_DailyTeamSpend"`} {
		if !strings.Contains(q, want) {
			t.Fatalf("query lacks %q", want)
		}
	}
}

func TestPricesAreFittedFromRows(t *testing.T) {
	// Rows (prompt, completion) priced at $3 and $15 per million tokens.
	rows := [][2]float64{{10000, 50}, {30000, 10}, {20000, 400}}
	var d deploymentTotals
	for _, row := range rows {
		spend := row[0]*3e-6 + row[1]*15e-6
		d.Rows++
		d.PP += row[0] * row[0]
		d.PC += row[0] * row[1]
		d.CC += row[1] * row[1]
		d.SP += spend * row[0]
		d.SC += spend * row[1]
	}
	in, out, ok := d.prices()
	if !ok || !closeTo(in, 3e-6) || !closeTo(out, 15e-6) {
		t.Fatalf("prices = %v, %v, %v", in, out, ok)
	}
	if _, _, ok := (deploymentTotals{Rows: 1, PP: 1, CC: 1}).prices(); ok {
		t.Fatal("one row fitted two prices")
	}
}

func TestHeaderChecksSeeForwardedAndStrippedHeaders(t *testing.T) {
	e := expected{Sent: 3}
	headers := map[string]map[string]int64{
		"x-request-origin":              {"loadtest-rust": 3, "loadtest-python": 3},
		"x-trace-id":                    {"loadtest-rust-trace": 3},
		"x-caller-team":                 {"loadtest-rust-team-0": 2, "loadtest-rust-team-1": 1},
		"x-cost-center":                 {},
		"x-litellm-spend-logs-metadata": {`{"run": "loadtest-python"}`: 3},
	}
	var r report
	checkHeaders(&r, "rust", e, headers)
	if n := r.failed(); n != 0 {
		t.Fatalf("%d checks failed:\n%s", n, r.markdown())
	}
	r = report{}
	checkHeaders(&r, "python", e, headers)
	// No trace or team headers for python, and its spend metadata header was not stripped.
	if n := r.failed(); n != 3 {
		t.Fatalf("want 3 failures, got %d:\n%s", n, r.markdown())
	}
}

func TestBifrostQueryScopesToTheKeySet(t *testing.T) {
	q := bifrostQuery("bifrost")
	for _, want := range []string{"virtual_key_id LIKE 'loadtest-bifrost-%'", "id ~ '^b-loadtest-bifrost-[0-9]+$'",
		"id = 'b-loadtest-bifrost-customer'", "'dialect', 'bifrost'"} {
		if !strings.Contains(q, want) {
			t.Fatalf("query lacks %q", want)
		}
	}
}

func TestBifrostChecksReportMissingLevelsAsNotApplicable(t *testing.T) {
	e := expected{Sent: 2, Total: tokens{Requests: 2, Prompt: 20, Completion: 10},
		Groups: map[string]*tokens{"g": {Requests: 2, Prompt: 20, Completion: 10}}}
	db := dbTotals{Dialect: "bifrost", Rows: 2, Success: 2, Prompt: 20, Completion: 10, Spend: 0.5, WithTeam: 2,
		WithUser: 2, WithModel: 2, Metadata: 2, Groups: map[string]*groupTotals{"g": {Rows: 2, Prompt: 20, Completion: 10}},
		KeySpend: 0.5, TeamSpend: 0.5, UserSpend: 0.5}
	var r report
	checkDB(&r, bifrostSide, e, db)
	if n := r.failed(); n != 0 {
		t.Fatalf("%d checks failed:\n%s", n, r.markdown())
	}
	if md := r.markdown(); !strings.Contains(md, "| team member spend = spend logs | bifrost | - | n/a |") {
		t.Fatalf("member spend not reported as n/a:\n%s", md)
	}
	db.UserSpend = 0.25
	r = report{}
	checkDB(&r, bifrostSide, e, db)
	if n := r.failed(); n != 1 {
		t.Fatalf("want 1 failure, got %d:\n%s", n, r.markdown())
	}
}

func TestBifrostLogObjectsReadCutTokenUsage(t *testing.T) {
	tot := &s3Totals{bifrost: true, Fields: map[string]bool{}, Metadata: map[string]bool{}}
	obj := `{"status":"success","cost":"0.00055","team_id":"loadtest-bifrost-team-3",` +
		`"token_usage":"{\"prompt_tokens\":100,\"completion_tokens\":5,\"total_tokens\":105,\"cost\":{\"inp",` +
		`"input_history":"[]","responses_input_history":"[{\"type\":\"message\"}]","output_message":"",` +
		`"responses_output":"[{\"id\":\"x\"}]","metadata":"{\"x-cost-center\":\"loadtest-bifrost\"}"}`
	if err := tot.add([]byte(obj)); err != nil {
		t.Fatal(err)
	}
	if tot.Prompt != 100 || tot.Completion != 5 || !near(tot.Cost, 0.00055) || tot.WithBodies != 1 || tot.WithTeam != 1 ||
		tot.Success != 1 || !tot.Metadata["x-cost-center"] {
		t.Fatalf("totals %+v", tot)
	}
}

func TestComparisonsWithBifrostSkipTelemetrySchemas(t *testing.T) {
	d := &dbTotals{Deployments: map[string]*deploymentTotals{}}
	m := map[string]*family{"litellm_spend_metric_total": {labels: map[string]bool{}}}
	var r report
	compareSides(&r, "rust", bifrostSide, d, d, m, map[string]*family{}, nil, nil)
	if md := r.markdown(); strings.Contains(md, "metrics only on") {
		t.Fatalf("compared metric names with Bifrost:\n%s", md)
	}
}
