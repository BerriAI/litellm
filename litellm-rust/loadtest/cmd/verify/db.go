package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"strings"
)

// groupTotals is one model group's spend logs.
type groupTotals struct {
	Rows       int64   `json:"rows"`
	Prompt     int64   `json:"prompt_tokens"`
	Completion int64   `json:"completion_tokens"`
	Spend      float64 `json:"spend"`
}

// deploymentTotals are one deployment's sums over its spend logs, enough to fit its per-token
// prices by least squares: spend = in * prompt + out * completion on every row.
type deploymentTotals struct {
	Rows int64   `json:"rows"`
	PP   float64 `json:"pp"`
	PC   float64 `json:"pc"`
	CC   float64 `json:"cc"`
	SP   float64 `json:"sp"`
	SC   float64 `json:"sc"`
}

// prices solves the normal equations; ok is false when the rows cannot tell the two prices apart.
func (d deploymentTotals) prices() (in, out float64, ok bool) {
	det := d.PP*d.CC - d.PC*d.PC
	if d.Rows < 2 || det <= 1e-9*d.PP*d.CC {
		return 0, 0, false
	}
	return (d.SP*d.CC - d.SC*d.PC) / det, (d.PP*d.SC - d.PC*d.SP) / det, true
}

// dbTotals is the one JSON row `verify sql` selects for a key set.
type dbTotals struct {
	Rows        int64                        `json:"rows"`
	Success     int64                        `json:"success_rows"`
	Prompt      int64                        `json:"prompt_tokens"`
	Completion  int64                        `json:"completion_tokens"`
	Spend       float64                      `json:"spend"`
	ZeroSpend   int64                        `json:"zero_spend_rows"`
	WithTeam    int64                        `json:"rows_with_team"`
	WithUser    int64                        `json:"rows_with_user"`
	WithModel   int64                        `json:"rows_with_model_id"`
	Metadata    int64                        `json:"rows_with_spend_metadata"`
	Groups      map[string]*groupTotals      `json:"groups"`
	Deployments map[string]*deploymentTotals `json:"deployments"`
	KeySpend    float64                      `json:"keys_spend"`
	TeamSpend   float64                      `json:"teams_spend"`
	UserSpend   float64                      `json:"users_spend"`
	MemberSpend float64                      `json:"members_spend"`
	DailyUser   daily                        `json:"daily_user"`
	DailyTeam   daily                        `json:"daily_team"`
}

type daily struct {
	Spend      float64 `json:"spend"`
	Requests   int64   `json:"requests"`
	Successful int64   `json:"successful"`
	Prompt     int64   `json:"prompt_tokens"`
	Completion int64   `json:"completion_tokens"`
}

func sqlCmd(args []string) {
	fs := flag.NewFlagSet("sql", flag.ExitOnError)
	set := fs.String("set", "", "key set name, as given to genconfig -key-sets")
	fs.Parse(args)
	fmt.Fprint(os.Stdout, query(*set))
}

// query selects a key set's records as one JSON value. Owners are found by the set's id prefix,
// spend logs by the set's keys.
func query(set string) string {
	prefix := setPrefix(set)
	p := strings.ReplaceAll(prefix, "'", "''") + "-"
	return fmt.Sprintf(`WITH k AS (SELECT token, key_alias, spend FROM "LiteLLM_VerificationToken" WHERE key_alias LIKE '%[1]s%%'),
s AS (SELECT l.* FROM "LiteLLM_SpendLogs" l JOIN k ON k.token = l.api_key),
g AS (SELECT coalesce(model_group, '') AS model_group, count(*) AS rows, sum(prompt_tokens) AS p,
             sum(completion_tokens) AS c, sum(spend) AS sp FROM s GROUP BY 1),
du AS (SELECT * FROM "LiteLLM_DailyUserSpend" WHERE user_id LIKE '%[1]s%%'),
dt AS (SELECT * FROM "LiteLLM_DailyTeamSpend" WHERE team_id LIKE '%[1]s%%'),
-- Deployments by the mock's /d/<name> path: the gateways give deployments different ids.
d AS (SELECT substring(api_base FROM '/d/([^/]+)') AS name, count(*) AS rows,
             sum(prompt_tokens::float8 * prompt_tokens) AS pp, sum(prompt_tokens::float8 * completion_tokens) AS pc,
             sum(completion_tokens::float8 * completion_tokens) AS cc, sum(spend * prompt_tokens) AS sp,
             sum(spend * completion_tokens) AS sc
      FROM s WHERE api_base LIKE '%%/d/%%' GROUP BY 1)
SELECT json_build_object(
  'rows', (SELECT count(*) FROM s),
  'success_rows', (SELECT count(*) FROM s WHERE coalesce(status, 'success') = 'success'),
  'prompt_tokens', (SELECT coalesce(sum(prompt_tokens), 0) FROM s),
  'completion_tokens', (SELECT coalesce(sum(completion_tokens), 0) FROM s),
  'spend', (SELECT coalesce(sum(spend), 0) FROM s),
  'zero_spend_rows', (SELECT count(*) FROM s WHERE spend <= 0),
  'rows_with_team', (SELECT count(*) FROM s WHERE team_id LIKE '%[1]s%%'),
  'rows_with_user', (SELECT count(*) FROM s WHERE "user" LIKE '%[1]s%%'),
  'rows_with_model_id', (SELECT count(*) FROM s WHERE coalesce(model_id, '') <> ''),
  'rows_with_spend_metadata', (SELECT count(*) FROM s
     WHERE metadata->'spend_logs_metadata'->>'run' = '%[2]s'
       AND metadata->'spend_logs_metadata'->>'cost_center' LIKE 'key-%%'),
  'groups', (SELECT coalesce(json_object_agg(model_group, json_build_object('rows', rows,
       'prompt_tokens', p, 'completion_tokens', c, 'spend', sp)), '{}') FROM g),
  'deployments', (SELECT coalesce(json_object_agg(name, json_build_object('rows', rows, 'pp', pp, 'pc', pc,
       'cc', cc, 'sp', sp, 'sc', sc)), '{}') FROM d),
  'keys_spend', (SELECT coalesce(sum(spend), 0) FROM k),
  'teams_spend', (SELECT coalesce(sum(spend), 0) FROM "LiteLLM_TeamTable" WHERE team_id LIKE '%[1]s%%'),
  'users_spend', (SELECT coalesce(sum(spend), 0) FROM "LiteLLM_UserTable" WHERE user_id LIKE '%[1]s%%'),
  'members_spend', (SELECT coalesce(sum(spend), 0) FROM "LiteLLM_TeamMembership" WHERE team_id LIKE '%[1]s%%'),
  'daily_user', (SELECT json_build_object('spend', coalesce(sum(spend), 0), 'requests', coalesce(sum(api_requests), 0),
       'successful', coalesce(sum(successful_requests), 0), 'prompt_tokens', coalesce(sum(prompt_tokens), 0),
       'completion_tokens', coalesce(sum(completion_tokens), 0)) FROM du),
  'daily_team', (SELECT json_build_object('spend', coalesce(sum(spend), 0), 'requests', coalesce(sum(api_requests), 0),
       'successful', coalesce(sum(successful_requests), 0), 'prompt_tokens', coalesce(sum(prompt_tokens), 0),
       'completion_tokens', coalesce(sum(completion_tokens), 0)) FROM dt)
);
`, p, strings.ReplaceAll(prefix, "'", "''"))
}

// close reports whether two dollar amounts agree to a millionth (summing in another order, or a
// counter's float accumulation, changes the last digits).
func near(a, b float64) bool {
	return math.Abs(a-b) <= 1e-6*math.Max(1, math.Max(math.Abs(a), math.Abs(b)))
}

func exact(r *report, name, side string, want, got int64) {
	r.add(check{Name: name, Side: side, Want: fmt.Sprint(want), Got: fmt.Sprint(got), OK: want == got})
}

func dollars(r *report, name, side string, want, got float64) {
	r.add(check{Name: name, Side: side, Want: fmt.Sprintf("$%.6f", want), Got: fmt.Sprintf("$%.6f", got), OK: near(want, got)})
}

func checkDB(r *report, side string, e expected, db dbTotals) {
	ok := e.Total.Requests
	exact(r, "spend log rows", side, ok, db.Rows)
	exact(r, "spend log rows marked success", side, ok, db.Success)
	exact(r, "spend log prompt tokens", side, e.Total.Prompt, db.Prompt)
	exact(r, "spend log completion tokens", side, e.Total.Completion, db.Completion)
	exact(r, "spend log rows priced at $0", side, 0, db.ZeroSpend)
	exact(r, "spend log rows naming the key's team", side, ok, db.WithTeam)
	exact(r, "spend log rows naming the key's user", side, ok, db.WithUser)
	exact(r, "spend log rows naming the deployment", side, ok, db.WithModel)
	exact(r, "spend log rows with request + key spend metadata", side, ok, db.Metadata)
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
	dollars(r, "key spend = spend logs", side, db.Spend, db.KeySpend)
	dollars(r, "team spend = spend logs", side, db.Spend, db.TeamSpend)
	dollars(r, "user spend = spend logs", side, db.Spend, db.UserSpend)
	dollars(r, "team member spend = spend logs", side, db.Spend, db.MemberSpend)
	for _, d := range []struct {
		name  string
		daily daily
	}{{"daily user", db.DailyUser}, {"daily team", db.DailyTeam}} {
		dollars(r, d.name+" spend = spend logs", side, db.Spend, d.daily.Spend)
		exact(r, d.name+" requests", side, ok, d.daily.Requests)
		exact(r, d.name+" successful requests", side, ok, d.daily.Successful)
		exact(r, d.name+" prompt tokens", side, e.Total.Prompt, d.daily.Prompt)
		exact(r, d.name+" completion tokens", side, e.Total.Completion, d.daily.Completion)
	}
}
