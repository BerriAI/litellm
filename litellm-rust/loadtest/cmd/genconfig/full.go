package main

import (
	"bufio"
	"crypto/rand"
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"fmt"
	"log"
)

// customCallbacks holds the Python proxy's stand-ins for the Rust gateway's request hooks.
//
//go:embed custom_callbacks.py
var customCallbacks string

// Owners per key set in full mode: key i belongs to user i%fullUsers, and user u to team
// u%fullTeams, so every user and team holds several keys.
const (
	fullUsers = 100
	fullTeams = 10
	// Limits no load-test rate reaches, so every request runs every check and none is refused.
	fullRPM      = 1_000_000
	fullTPM      = 1_000_000_000_000
	fullBudget   = 1_000_000
	fullParallel = 1000
)

// fullFeatureHeaders are what a client of a key set sends with -features full so each request
// hook has work: the Rust gateway's forward_headers, spend_metadata, strip_headers and
// provider_metadata, and the Python stand-ins in custom_callbacks.py, read the
// same names. Values carry the set's prefix, so mockllm -count-headers can tell sides apart.
// genconfig writes them to headers.txt (headers-<set>.txt), one loadgen -header per line.
func fullFeatureHeaders(prefix string) []string {
	return []string{
		"x-request-origin: " + prefix,
		"x-trace-id: " + prefix + "-trace",
		"x-cost-center: " + prefix,
		`x-litellm-spend-logs-metadata: {"run": "` + prefix + `"}`,
	}
}

const deploymentCapacity = "      rpm: 1000000\n      tpm: 1000000000000\n"

// The callbacks are Python's names, which the Rust gateway reads too; custom_callbacks.* are the
// Python stand-ins for the Rust request hooks, which the Rust gateway skips.
const fullFeatureSettings = `litellm_settings:
  require_auth_for_metrics_endpoint: false
  callbacks:
    - dynamic_rate_limiter_v3
    - prometheus
    - s3_v2
    - custom_callbacks.forward_headers
    - custom_callbacks.spend_metadata
    - custom_callbacks.strip_headers
    - custom_callbacks.provider_metadata
  s3_callback_params:
    s3_bucket_name: os.environ/LOADTEST_S3_BUCKET
    s3_region_name: us-east-1
    s3_endpoint_url: os.environ/LOADTEST_S3_ENDPOINT
    s3_path: os.environ/LOADTEST_S3_PATH
    s3_aws_access_key_id: os.environ/LOADTEST_S3_ACCESS_KEY_ID
    s3_aws_secret_access_key: os.environ/LOADTEST_S3_SECRET_ACCESS_KEY
  hooks:
    - forward_headers:
        headers: ["x-request-origin", "x-trace-*"]
    - spend_metadata:
        from_headers:
          x-litellm-spend-logs-metadata: json
          x-cost-center: cost_center
        from_key_metadata: [cost_center]
    - strip_headers:
        headers: ["x-litellm-spend-logs-metadata", "x-cost-center"]
    - provider_metadata:
        body:
          metadata.user_id: "{end_user|key.user_id}"
        headers:
          x-caller-team: "{key.team_alias}"
`

// setPrefix names a key set's keys, users, teams and budgets, so sets never collide in one
// database.
func setPrefix(set string) string {
	if set == "" {
		return "loadtest"
	}
	return "loadtest-" + set
}

// writeOwners writes a key set's member budget, teams, users and team memberships.
func writeOwners(sw *bufio.Writer, prefix string) {
	fmt.Fprintf(sw, `INSERT INTO "LiteLLM_BudgetTable" (budget_id, max_budget, rpm_limit, tpm_limit, created_by, updated_by) VALUES
  ('%s-member', %d, %d, %d, 'loadtest', 'loadtest')
ON CONFLICT (budget_id) DO NOTHING;
`, prefix, fullBudget, fullRPM, fullTPM)
	sw.WriteString(`INSERT INTO "LiteLLM_TeamTable" (team_id, team_alias, max_budget, rpm_limit, tpm_limit, metadata) VALUES` + "\n")
	for t := range fullTeams {
		if t > 0 {
			sw.WriteString(",\n")
		}
		fmt.Fprintf(sw, `  ('%s-team-%d', '%s-team-%d', %d, %d, %d, '{"cost_center": "team-%d"}')`,
			prefix, t, prefix, t, fullBudget, fullRPM, fullTPM, t)
	}
	sw.WriteString("\nON CONFLICT (team_id) DO NOTHING;\n")
	sw.WriteString(`INSERT INTO "LiteLLM_UserTable" (user_id, user_alias, max_budget, rpm_limit, tpm_limit, teams) VALUES` + "\n")
	for u := range fullUsers {
		if u > 0 {
			sw.WriteString(",\n")
		}
		fmt.Fprintf(sw, `  ('%s-user-%d', '%s-user-%d', %d, %d, %d, ARRAY['%s-team-%d'])`,
			prefix, u, prefix, u, fullBudget, fullRPM, fullTPM, prefix, u%fullTeams)
	}
	sw.WriteString("\nON CONFLICT (user_id) DO NOTHING;\n")
	sw.WriteString(`INSERT INTO "LiteLLM_TeamMembership" (user_id, team_id, budget_id) VALUES` + "\n")
	for u := range fullUsers {
		if u > 0 {
			sw.WriteString(",\n")
		}
		fmt.Fprintf(sw, `  ('%s-user-%d', '%s-team-%d', '%s-member')`, prefix, u, prefix, u%fullTeams, prefix)
	}
	sw.WriteString("\nON CONFLICT (user_id, team_id) DO NOTHING;\n")
}

// writeKeys writes one key set: raw keys to kw, hashed rows to sw, with owners and limits in
// full mode.
func writeKeys(kw, sw *bufio.Writer, set string, keys int, keySeed string, full bool) {
	prefix := setPrefix(set)
	if full && keys > 0 {
		writeOwners(sw, prefix)
	}
	const batch = 1000
	for i := range keys {
		raw := make([]byte, 16)
		if keySeed == "" {
			if _, err := rand.Read(raw); err != nil {
				log.Fatal(err)
			}
		} else {
			// Predictable by design, for load tests only. The default set keeps its old keys.
			seed := keySeed
			if set != "" {
				seed += "/" + set
			}
			derived := sha256.Sum256(fmt.Appendf(nil, "%s/%d", seed, i))
			copy(raw, derived[:])
		}
		key := "sk-lt-" + hex.EncodeToString(raw)
		sum := sha256.Sum256([]byte(key))
		fmt.Fprintln(kw, key)
		if i%batch == 0 {
			if i > 0 {
				sw.WriteString("\nON CONFLICT (token) DO NOTHING;\n")
			}
			if full {
				sw.WriteString(`INSERT INTO "LiteLLM_VerificationToken" (token, key_alias, models, user_id, team_id, rpm_limit, tpm_limit, max_parallel_requests, max_budget, metadata) VALUES` + "\n")
			} else {
				sw.WriteString(`INSERT INTO "LiteLLM_VerificationToken" (token, key_alias, models) VALUES` + "\n")
			}
		} else {
			sw.WriteString(",\n")
		}
		if full {
			u := i % fullUsers
			fmt.Fprintf(sw, `  ('%s', '%s-%d', ARRAY[]::text[], '%s-user-%d', '%s-team-%d', %d, %d, %d, %d, '{"cost_center": "key-%d"}')`,
				hex.EncodeToString(sum[:]), prefix, i, prefix, u, prefix, u%fullTeams, fullRPM, fullTPM, fullParallel, fullBudget, i)
		} else {
			fmt.Fprintf(sw, "  ('%s', '%s-%d', ARRAY[]::text[])", hex.EncodeToString(sum[:]), prefix, i)
		}
	}
	if keys > 0 {
		sw.WriteString("\nON CONFLICT (token) DO NOTHING;\n")
	}
}
