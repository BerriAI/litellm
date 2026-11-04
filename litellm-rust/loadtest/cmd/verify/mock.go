package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// mockHeaders reads the mock's per-value request header counts (mockllm -count-headers).
func mockHeaders(base string) (map[string]map[string]int64, error) {
	resp, err := http.Get(strings.TrimSuffix(base, "/") + "/stats")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("/stats answered %d", resp.StatusCode)
	}
	var stats struct {
		Headers map[string]map[string]int64 `json:"headers"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&stats); err != nil {
		return nil, err
	}
	return stats.Headers, nil
}

// checkHeaders checks what the request hooks did to the side's requests, as the provider saw
// them: forwarded headers arrive, stripped ones do not, and the caller's team is added. Upstream
// retries can only add requests, so arrivals are at least the successful requests.
func checkHeaders(r *report, side string, e expected, headers map[string]map[string]int64) {
	prefix := setPrefix(side)
	count := func(name string, match func(string) bool) int64 {
		var n int64
		for value, c := range headers[name] {
			if match(value) {
				n += c
			}
		}
		return n
	}
	ok := e.Sent - e.Failed
	arrived := count("x-request-origin", func(v string) bool { return v == prefix })
	atLeast := func(name string, got int64) {
		r.add(check{Name: name, Side: side, Want: fmt.Sprintf(">= %d", ok), Got: fmt.Sprint(got), OK: got >= ok})
	}
	atLeast("upstream requests with x-request-origin forwarded", arrived)
	exact(r, "upstream requests with x-trace-* forwarded", side, arrived,
		count("x-trace-id", func(v string) bool { return v == prefix+"-trace" }))
	if isBifrost(side) {
		notApplicable(r, "upstream requests with x-caller-team added", side, "Bifrost cannot add headers from the key's team")
	} else {
		exact(r, "upstream requests with x-caller-team added", side, arrived,
			count("x-caller-team", func(v string) bool { return strings.HasPrefix(v, prefix+"-team-") }))
	}
	for _, name := range []string{"x-cost-center", "x-litellm-spend-logs-metadata"} {
		exact(r, "upstream requests still carrying "+name, side, 0,
			count(name, func(v string) bool { return strings.Contains(v, prefix) }))
	}
}

// setPrefix is genconfig's id prefix for a key set.
func setPrefix(set string) string {
	if set == "" {
		return "loadtest"
	}
	return "loadtest-" + set
}
