package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func coverageSpecFixture(paths map[string][]string) map[string]map[string]json.RawMessage {
	spec := make(map[string]map[string]json.RawMessage)
	for path, methods := range paths {
		operations := make(map[string]json.RawMessage)
		for _, method := range methods {
			operations[method] = json.RawMessage(`{}`)
		}
		spec[path] = operations
	}
	return spec
}

func writeAllowlist(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "allowlist.txt")
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestParseAllowlist(t *testing.T) {
	path := writeAllowlist(t, `# comment
GET /team/spend/report

post /key/regenerate  # inline reason
`)
	entries, err := parseAllowlist(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 2 || !entries["GET /team/spend/report"] || !entries["POST /key/regenerate"] {
		t.Fatalf("unexpected entries: %v", entries)
	}
}

func TestParseAllowlistRejectsMalformedLines(t *testing.T) {
	path := writeAllowlist(t, "GET\n")
	if _, err := parseAllowlist(path); err == nil {
		t.Fatal("expected error for malformed line")
	}
}

func TestAuditCoverageFailsOnUncoveredManagementEndpoint(t *testing.T) {
	spec := coverageSpecFixture(map[string][]string{
		"/team/new":            {"post"},
		"/team/spend/report":   {"get"},
		"/chat/completions":    {"post"},
		"/health/liveliness":   {"get"},
		"/v1/chat/completions": {"post"},
	})
	calls := []endpointCall{{Method: "POST", Path: "/team/new"}}
	violations := auditCoverage(calls, spec, nil)
	if len(violations) != 1 || !strings.Contains(violations[0], "GET /team/spend/report") {
		t.Fatalf("unexpected violations: %v", violations)
	}
}

func TestAuditCoverageAllowlistSuppressesUncovered(t *testing.T) {
	spec := coverageSpecFixture(map[string][]string{"/team/spend/report": {"get"}})
	violations := auditCoverage(nil, spec, map[string]bool{"GET /team/spend/report": true})
	if len(violations) != 0 {
		t.Fatalf("unexpected violations: %v", violations)
	}
}

func TestAuditCoverageFailsOnStaleCoveredEntry(t *testing.T) {
	spec := coverageSpecFixture(map[string][]string{"/team/new": {"post"}})
	calls := []endpointCall{{Method: "POST", Path: "/team/new"}}
	violations := auditCoverage(calls, spec, map[string]bool{"POST /team/new": true})
	if len(violations) != 1 || !strings.Contains(violations[0], "stale allowlist entry: POST /team/new is covered") {
		t.Fatalf("unexpected violations: %v", violations)
	}
}

func TestAuditCoverageFailsOnEntryMissingFromSchema(t *testing.T) {
	spec := coverageSpecFixture(map[string][]string{"/team/new": {"post"}})
	calls := []endpointCall{{Method: "POST", Path: "/team/new"}}
	violations := auditCoverage(calls, spec, map[string]bool{"POST /team/removed": true})
	if len(violations) != 1 || !strings.Contains(violations[0], "POST /team/removed is not a management endpoint") {
		t.Fatalf("unexpected violations: %v", violations)
	}
}

func TestAuditCoverageMatchesPathParams(t *testing.T) {
	spec := coverageSpecFixture(map[string][]string{"/team/{team_id}/callback": {"get"}})
	calls := []endpointCall{{Method: "GET", Path: "/team/{param}/callback"}}
	violations := auditCoverage(calls, spec, nil)
	if len(violations) != 0 {
		t.Fatalf("unexpected violations: %v", violations)
	}
}

func TestMountedDeclarativeAPIsAreManagementPaths(t *testing.T) {
	for _, path := range []string{
		"/cache/settings",
		"/config/cost_discount_config",
		"/coordination_redis/settings",
		"/router/settings",
	} {
		if !isManagementPath(path) {
			t.Fatalf("%s should be classified as a management path", path)
		}
	}
	for _, path := range []string{"/chat/completions", "/health/liveliness"} {
		if isManagementPath(path) {
			t.Fatalf("%s should not be classified as a management path", path)
		}
	}
}

func TestBundledAllowlistEntriesAreManagementPaths(t *testing.T) {
	entries, err := parseAllowlist("coverage_allowlist.txt")
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) == 0 {
		t.Fatal("bundled allowlist parsed to zero entries")
	}
	for entry := range entries {
		fields := strings.Fields(entry)
		if !isManagementPath(fields[1]) {
			t.Fatalf("allowlist entry %q is not under a management prefix", entry)
		}
	}
}
