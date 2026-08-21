package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

func writeFixture(t *testing.T, dir, name, body string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func extractFixture(t *testing.T, files map[string]string) extraction {
	t.Helper()
	dir := t.TempDir()
	for name, body := range files {
		writeFixture(t, dir, name, body)
	}
	result, err := extractProviderCalls(dir)
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func callSet(calls []endpointCall) []string {
	set := make(map[string]bool)
	for _, call := range calls {
		set[call.Method+" "+call.Path] = true
	}
	keys := make([]string, 0, len(set))
	for key := range set {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func TestExtractResolvesAllCallShapes(t *testing.T) {
	result := extractFixture(t, map[string]string{
		"consts.go": `package p

const (
	endpointModelNew    = "/model/new"
	endpointModelUpdate = "/model/update"
	endpointMCPRead     = "/v1/mcp/server"
)
`,
		"calls.go": `package p

import "fmt"

func (c *Client) a() {
	c.sendRequest("POST", "/team/new", nil)
	c.sendRequest("GET", fmt.Sprintf("/team/info?team_id=%s", "x"), nil)
}

func b(client *Client, isUpdate bool, serverID string) {
	MakeRequest(client, "POST", "/credentials", nil)
	endpoint := endpointModelNew
	if isUpdate {
		endpoint = endpointModelUpdate
	}
	MakeRequest(client, "POST", endpoint, nil)
	readEndpoint := fmt.Sprintf("%s/%s", endpointMCPRead, serverID)
	MakeRequest(client, "GET", readEndpoint, nil)
}
`,
	})
	if len(result.Unresolved) != 0 {
		t.Fatalf("unexpected unresolved: %v", result.Unresolved)
	}
	got := callSet(result.Calls)
	want := []string{
		"GET /team/info",
		"GET /v1/mcp/server/{param}",
		"POST /credentials",
		"POST /model/new",
		"POST /model/update",
		"POST /team/new",
	}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestExtractFailsClosedOnDynamicPath(t *testing.T) {
	result := extractFixture(t, map[string]string{
		"calls.go": `package p

func a(c *Client, path string) {
	c.sendRequest("GET", path, nil)
}
`,
	})
	if len(result.Unresolved) != 1 {
		t.Fatalf("want 1 unresolved call site, got %v", result.Unresolved)
	}
}

func TestExtractFlagsRawHTTPRequestOutsideHelpers(t *testing.T) {
	result := extractFixture(t, map[string]string{
		"rogue.go": `package p

import "net/http"

func a() {
	http.NewRequest("GET", "http://example.com/model/new", nil)
}
`,
	})
	if len(result.Unresolved) != 1 || !strings.Contains(result.Unresolved[0], "raw http.NewRequest") {
		t.Fatalf("want raw request violation, got %v", result.Unresolved)
	}
}

func TestExtractAllowsRawHTTPRequestInHelpers(t *testing.T) {
	result := extractFixture(t, map[string]string{
		"utils.go": `package p

import "net/http"

func MakeRequest(client *Client, method, endpoint string, body interface{}) {
	http.NewRequest(method, endpoint, nil)
}
`,
	})
	if len(result.Unresolved) != 0 {
		t.Fatalf("unexpected unresolved: %v", result.Unresolved)
	}
}

func specFixture(t *testing.T) map[string]map[string]json.RawMessage {
	t.Helper()
	raw := `{
		"paths": {
			"/team/new": {"post": {}},
			"/organization/update": {"patch": {}},
			"/credentials/{credential_name}": {"get": {}, "delete": {}}
		}
	}`
	dir := t.TempDir()
	specPath := filepath.Join(dir, "spec.json")
	if err := os.WriteFile(specPath, []byte(raw), 0o644); err != nil {
		t.Fatal(err)
	}
	paths, err := loadSpecPaths(specPath)
	if err != nil {
		t.Fatal(err)
	}
	return paths
}

func TestAuditDetectsMissingPathAndWrongMethod(t *testing.T) {
	spec := specFixture(t)
	violations := auditCalls([]endpointCall{
		{Method: "POST", Path: "/team/new", Pos: "a.go:1"},
		{Method: "GET", Path: "/credentials/{param}", Pos: "a.go:2"},
		{Method: "POST", Path: "/organization/update", Pos: "a.go:3"},
		{Method: "POST", Path: "/gone/away", Pos: "a.go:4"},
	}, spec)
	if len(violations) != 2 {
		t.Fatalf("want 2 violations, got %v", violations)
	}
	joined := strings.Join(violations, "\n")
	if !strings.Contains(joined, "POST /organization/update: path exists but method not allowed") {
		t.Fatalf("missing method violation: %v", violations)
	}
	if !strings.Contains(joined, "POST /gone/away is not served by the proxy") {
		t.Fatalf("missing path violation: %v", violations)
	}
}

func TestNormalizePathStripsQueryAndVerbs(t *testing.T) {
	if got := normalizePath("/key/info?key=%s"); got != "/key/info" {
		t.Fatalf("got %q", got)
	}
	if got := normalizePath("/credentials/%s"); got != "/credentials/{param}" {
		t.Fatalf("got %q", got)
	}
}
