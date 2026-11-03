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

// portkeySet is the key set genconfig writes as Portkey gateway routing instead of LiteLLM key rows.
// The open-source Portkey gateway (github.com/Portkey-AI/gateway) keeps no model list, keys or
// database: every request carries its routing in an x-portkey-config header. So the set becomes
// portkey-headers.json, the header loadgen sends for each model group (-model-headers), and
// keys-portkey.txt, which the gateway never checks (it has no client auth) and exists only so the
// request bytes match the other sides.
const portkeySet = "portkey"

// portkeyHeader is the routing header name. Its value is a Portkey config (JSON).
const portkeyHeader = "x-portkey-config"

// writePortkey writes portkey-headers.json and keys-portkey.txt beside config.yaml. Each model
// group's config load-balances evenly over the group's deployments, Portkey's counterpart of
// simple-shuffle, and retries twice like router_settings.num_retries. Every target points at its
// deployment's /d/<id> path on the mock with custom_host, so the mock counts hits per deployment as
// it does for the other sides. Bedrock targets carry static AWS keys, so every request is SigV4
// signed; custom_host replaces the Bedrock runtime URL, so they reach the mock over plain HTTP
// (Portkey sends Anthropic-model messages to Bedrock Converse, which the mock serves).
//
// The gateway refuses a custom_host on a private or .local name unless it is listed in
// TRUSTED_CUSTOM_HOSTS, so the pod sets that to the mock's host name.
func writePortkey(deps []bifrostDeployment, keys []string, outDir string) {
	byGroup := map[string][]any{}
	var groups []string
	for _, d := range deps {
		if _, ok := byGroup[d.group]; !ok {
			groups = append(groups, d.group)
		}
		var t map[string]any
		switch d.kind {
		case "bedrock":
			t = map[string]any{
				"provider":              "bedrock",
				"aws_access_key_id":     "mock-loadtest-access-key",
				"aws_secret_access_key": "mock-loadtest-secret",
				"aws_region":            "us-east-1",
				"custom_host":           d.base,
			}
		case "openai", "anthropic":
			// Both providers append their endpoint (/chat/completions, /messages) to a base that
			// ends in /v1, as their public base URLs do.
			t = map[string]any{"provider": d.kind, "api_key": "sk-mock", "custom_host": d.base + "/v1"}
		default:
			log.Fatalf("portkey: no mapping for deployment %s of kind %q", d.id, d.kind)
		}
		t["override_params"] = map[string]any{"model": d.model}
		t["weight"] = 1
		byGroup[d.group] = append(byGroup[d.group], t)
	}
	headers := map[string]map[string]string{}
	for _, g := range groups {
		cfg := map[string]any{
			"strategy": map[string]any{"mode": "loadbalance"},
			"retry":    map[string]any{"attempts": 2},
			"targets":  byGroup[g],
		}
		b, err := json.Marshal(cfg)
		if err != nil {
			log.Fatal(err)
		}
		headers[g] = map[string]string{portkeyHeader: string(b)}
	}
	writeJSONFile(filepath.Join(outDir, "portkey-headers.json"), headers)
	write(filepath.Join(outDir, "keys-"+portkeySet+".txt"), strings.Join(keys, "\n")+"\n")
}

// portkeyKeys derives the set's keys like writeKeys does (same sk-lt- shape and length), so each
// request carries the same bytes as on the other sides.
func portkeyKeys(n int, keySeed string) []string {
	if keySeed == "" {
		keySeed = "loadtest"
	}
	keys := make([]string, 0, n)
	for i := range n {
		derived := sha256.Sum256(fmt.Appendf(nil, "%s/%s/%d", keySeed, portkeySet, i))
		keys = append(keys, "sk-lt-"+hex.EncodeToString(derived[:16]))
	}
	return keys
}
