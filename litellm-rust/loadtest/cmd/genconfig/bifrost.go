package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"path/filepath"
	"strings"
)

// bifrostSet is the key set genconfig writes as a Bifrost config instead of LiteLLM key rows.
// Bifrost (github.com/maximhq/bifrost, open-source build) boots its providers, routing and
// governance from config.json into its config store, so the seed Job loads nothing for it.
const bifrostSet = "bifrost"

// bifrostDeployment is one deployment of the LiteLLM config as Bifrost sees it: a custom provider
// named after the mock's /d/<id> path, so log rows and metrics name the deployment.
type bifrostDeployment struct {
	id, group, base, model string
	kind                   string // openai, anthropic or bedrock
}

// bifrostPrices are the cost map's prices (model_prices_and_context_window.json) for the models
// genconfig deploys. Bifrost reads LiteLLM's field names; the lane's verify compares each
// deployment's fitted price across sides, which catches drift.
var bifrostPrices = map[string]map[string]any{
	"gpt-6-astra": {
		"mode":                                    "chat",
		"input_cost_per_token":                    1e-05,
		"output_cost_per_token":                   5e-05,
		"input_cost_per_token_above_272k_tokens":  2e-05,
		"output_cost_per_token_above_272k_tokens": 7.5e-05,
		"cache_read_input_token_cost":             1e-06,
		"cache_creation_input_token_cost":         1.25e-05,
	},
	"claude-opus-5-5": {
		"mode":                            "chat",
		"input_cost_per_token":            4e-06,
		"output_cost_per_token":           2e-05,
		"cache_read_input_token_cost":     2e-07,
		"cache_creation_input_token_cost": 5e-06,
	},
	"us.anthropic.claude-opus-5-5": {
		"mode":                            "chat",
		"input_cost_per_token":            4.4e-06,
		"output_cost_per_token":           2.2e-05,
		"cache_read_input_token_cost":     2.2e-07,
		"cache_creation_input_token_cost": 5.5e-06,
	},
}

// bifrostOptions are the knobs a fair comparison pins; deploy prints them.
type bifrostOptions struct {
	full        bool
	keys        int
	keySeed     string
	concurrency int
	bufferSize  int
	pricingFile string
	bedrockTLS  string // the mock's TLS listener, host:port
	outDir      string
}

// writeBifrost writes bifrost-config.json and bifrost-pricing.json beside config.yaml, plus
// keys-bifrost.txt and, with full features, headers-bifrost.txt. Connection settings come from
// environment variables (Bifrost's env. references): BIFROST_PG_{HOST,PORT,USER,PASSWORD,DB} for
// the config and logs stores and LOADTEST_S3_* for the request log payloads, which go under the
// bifrost/ prefix. The pricing file is read from o.pricingFile, where the gateway mounts it.
func writeBifrost(deps []bifrostDeployment, o bifrostOptions) {
	cfg := map[string]any{"$schema": "https://www.getbifrost.ai/schema"}
	client := map[string]any{
		"initial_pool_size":         o.bufferSize,
		"drop_excess_requests":      false,
		"enable_logging":            o.full,
		"disable_content_logging":   false,
		"enforce_auth_on_inference": true,
		// Like the LiteLLM configs' require_auth_for_metrics_endpoint: false, so verify can scrape.
		"whitelisted_routes": []string{"/metrics"},
	}
	pg := map[string]any{
		"host": "env.BIFROST_PG_HOST", "port": "env.BIFROST_PG_PORT", "user": "env.BIFROST_PG_USER",
		"password": "env.BIFROST_PG_PASSWORD", "db_name": "env.BIFROST_PG_DB", "ssl_mode": "disable",
	}
	cfg["config_store"] = map[string]any{"enabled": true, "type": "postgres", "config": pg}
	// Prices from the file genconfig writes, not getbifrost.ai; no background catalog fetches.
	cfg["framework"] = map[string]any{"pricing": map[string]any{
		"pricing_url":               "file://" + o.pricingFile,
		"model_parameters_url":      "file://" + o.pricingFile,
		"pricing_sync_interval":     86400,
		"mcp_library_sync_interval": 0,
		"live_models_sync_interval": 0,
	}}

	providers := map[string]any{}
	for _, d := range deps {
		var key map[string]any
		network := map[string]any{
			"default_request_timeout_in_seconds": 1800,
			"stream_idle_timeout_in_seconds":     600,
			// LiteLLM's router_settings.num_retries; no cooldown equivalent.
			"max_retries":           2,
			"allow_private_network": true,
		}
		if d.kind == "bedrock" {
			// Static keys, so every request is SigV4-signed as on the other sides. Bifrost's
			// Bedrock provider takes hosts, not a base URL, and always uses HTTPS: the runtime and
			// control-plane hosts are the mock's TLS listener, and the deployment travels in the
			// inference profile ARN, which Bifrost puts in the path (/model/d/<id>/<model>/...).
			host := strings.TrimPrefix(strings.TrimPrefix(o.bedrockTLS, "https://"), "http://")
			key = map[string]any{"name": d.id, "models": []string{"*"}, "weight": 1.0,
				"bedrock_key_config": map[string]any{"access_key": "mock-loadtest-access-key",
					"secret_key": "mock-loadtest-secret", "region": "us-east-1", "arn": "d/" + d.id,
					"endpoints": map[string]any{"runtime": host, "control_plane": host}}}
			network["insecure_skip_verify"] = true
		} else {
			network["base_url"] = d.base
			key = map[string]any{"name": d.id, "value": "sk-mock", "models": []string{"*"}, "weight": 1.0}
		}
		providers[d.id] = map[string]any{
			"keys":                        []any{key},
			"network_config":              network,
			"concurrency_and_buffer_size": map[string]any{"concurrency": o.concurrency, "buffer_size": o.bufferSize},
			"custom_provider_config":      map[string]any{"base_provider_type": d.kind},
		}
	}
	cfg["providers"] = providers

	gov := map[string]any{}
	// Each model group is a routing rule with its deployments as equal-weight targets, Bifrost's
	// counterpart of simple-shuffle. The rule's name is the group, so log rows name it.
	var rules []any
	var groups []string
	targets := map[string][]any{}
	for _, d := range deps {
		if _, ok := targets[d.group]; !ok {
			groups = append(groups, d.group)
		}
		targets[d.group] = append(targets[d.group], map[string]any{"provider": d.id, "model": d.model})
	}
	for i, g := range groups {
		ts := targets[g]
		for _, t := range ts {
			t.(map[string]any)["weight"] = 1.0 / float64(len(ts))
		}
		rules = append(rules, map[string]any{
			"id": "rule-" + g, "name": g, "enabled": true, "scope": "global", "priority": i,
			"cel_expression": fmt.Sprintf("model == %q", g), "targets": ts,
		})
	}
	gov["routing_rules"] = rules

	prefix := setPrefix(bifrostSet)
	keys := bifrostKeys(o.keys, o.keySeed)
	if o.full {
		var limits, teams, vks, models []any
		limit := func(id string) string {
			limits = append(limits, map[string]any{"id": id,
				"request_max_limit": fullRPM, "request_reset_duration": "1m",
				"token_max_limit": fullTPM, "token_reset_duration": "1m"})
			return id
		}
		// Budgets are declared inline on their owner, which is how Bifrost links them.
		budget := func(id string) []any {
			return []any{map[string]any{"id": id, "max_limit": fullBudget, "reset_duration": "1M"}}
		}
		// Bifrost's open-source hierarchy is virtual key -> team -> customer; users and team
		// members are enterprise-only, so the customer stands in for the user level.
		customer := prefix + "-customer"
		gov["customers"] = []any{map[string]any{"id": customer, "name": customer,
			"budgets": budget("b-" + customer), "rate_limit_id": limit("rl-" + customer)}}
		for t := range fullTeams {
			id := fmt.Sprintf("%s-team-%d", prefix, t)
			teams = append(teams, map[string]any{"id": id, "name": id, "customer_id": customer,
				"budgets": budget("b-" + id), "rate_limit_id": limit("rl-" + id)})
		}
		for i, k := range keys {
			id := fmt.Sprintf("%s-%d", prefix, i)
			vks = append(vks, map[string]any{"id": id, "name": id, "value": k, "is_active": true,
				"allow_all_providers": true, "team_id": fmt.Sprintf("%s-team-%d", prefix, (i%fullUsers)%fullTeams),
				"budgets": budget("b-" + id), "rate_limit_id": limit("rl-" + id)})
		}
		// Model capacity: each deployment's rpm and tpm, as model configs scoped to its provider.
		for _, d := range deps {
			models = append(models, map[string]any{"id": "mc-" + d.id, "model_name": d.model, "provider": d.id,
				"rate_limit_id": limit("rl-" + d.id)})
		}
		gov["rate_limits"], gov["teams"], gov["virtual_keys"], gov["model_configs"] = limits, teams, vks, models

		// forward_headers forwards these directly; strip_headers' names never leave (only the
		// allowlist is forwarded) and are denied anyway. spend_metadata's nearest equivalent copies
		// the raw header values into the log row's metadata. provider_metadata has none.
		client["header_filter_config"] = map[string]any{
			"allowlist": []string{"x-request-origin", "x-trace-*"},
			"denylist":  []string{"x-litellm-spend-logs-metadata", "x-cost-center"},
		}
		client["logging_headers"] = []string{"x-litellm-spend-logs-metadata", "x-cost-center"}
		cfg["logs_store"] = map[string]any{"enabled": true, "type": "postgres", "config": pg,
			"object_storage": map[string]any{
				"type": "s3", "bucket": "env.LOADTEST_S3_BUCKET", "region": "us-east-1",
				"endpoint": "env.LOADTEST_S3_ENDPOINT", "access_key_id": "env.LOADTEST_S3_ACCESS_KEY_ID",
				"secret_access_key": "env.LOADTEST_S3_SECRET_ACCESS_KEY", "prefix": bifrostSet,
				"force_path_style": true,
			}}
		cfg["plugins"] = []any{
			map[string]any{"enabled": true, "name": "telemetry", "config": map[string]any{}},
		}
	} else {
		var vks []any
		for i, k := range keys {
			id := fmt.Sprintf("%s-%d", prefix, i)
			vks = append(vks, map[string]any{"id": id, "name": id, "value": k, "is_active": true, "allow_all_providers": true})
		}
		gov["virtual_keys"] = vks
		cfg["logs_store"] = map[string]any{"enabled": false}
	}
	cfg["governance"] = gov
	cfg["client"] = client

	writeJSONFile(filepath.Join(o.outDir, "bifrost-config.json"), cfg)
	// Bifrost prices a request by its provider's name, and a custom provider is not priced as its
	// base provider, so bifrost-pricing.json has one row per deployment ("<id>/<model>").
	pricing := map[string]map[string]any{}
	for _, d := range deps {
		row := map[string]any{"provider": d.id}
		for k, v := range bifrostPrices[d.model] {
			row[k] = v
		}
		pricing[d.id+"/"+d.model] = row
	}
	writeJSONFile(filepath.Join(o.outDir, "bifrost-pricing.json"), pricing)
	write(filepath.Join(o.outDir, "keys-"+bifrostSet+".txt"), strings.Join(keys, "\n")+"\n")
	if o.full {
		write(filepath.Join(o.outDir, "headers-"+bifrostSet+".txt"), strings.Join(fullFeatureHeaders(prefix), "\n")+"\n")
	}
}

// bifrostKeys derives the set's keys like writeKeys does, with Bifrost's sk-bf- prefix, which is
// how it tells a virtual key from a provider key.
func bifrostKeys(n int, keySeed string) []string {
	keys := make([]string, 0, n)
	for i := range n {
		var raw [16]byte
		if keySeed == "" {
			keySeed = "loadtest"
		}
		derived := sha256.Sum256(fmt.Appendf(nil, "%s/%s/%d", keySeed, bifrostSet, i))
		copy(raw[:], derived[:])
		keys = append(keys, "sk-bf-lt-"+hex.EncodeToString(raw[:]))
	}
	return keys
}

func writeJSONFile(path string, v any) {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		log.Fatal(err)
	}
	write(path, string(b)+"\n")
}
