package litellm

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func TestDataSourceMCPServerRead(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/v1/mcp/server/srv-123" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{
			"server_id": "srv-123",
			"server_name": "github-mcp",
			"alias": "gh",
			"description": "GitHub MCP server",
			"url": "https://mcp.example.com",
			"transport": "http",
			"spec_version": "2024-11-05",
			"auth_type": "bearer",
			"mcp_access_groups": ["dev"],
			"allowed_tools": ["list_repos"],
			"extra_headers": ["x-request-id"],
			"command": "",
			"args": [],
			"env": {"SECRET_TOKEN": "should-never-surface"},
			"static_headers": {"Authorization": "Bearer should-never-surface"},
			"allow_all_keys": true,
			"status": "healthy",
			"last_health_check": "2026-02-01T00:00:00Z",
			"health_check_error": "",
			"created_at": "2026-01-01T00:00:00Z",
			"created_by": "admin",
			"updated_at": "2026-02-01T00:00:00Z",
			"updated_by": "admin"
		}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMMCPServer().Schema, map[string]interface{}{
		"server_id": "srv-123",
	})

	if err := dataSourceLiteLLMMCPServerRead(d, client); err != nil {
		t.Fatalf("read failed: %v", err)
	}

	if d.Id() != "srv-123" {
		t.Fatalf("expected ID 'srv-123', got %q", d.Id())
	}
	checks := map[string]interface{}{
		"server_name":       "github-mcp",
		"alias":             "gh",
		"description":       "GitHub MCP server",
		"url":               "https://mcp.example.com",
		"transport":         "http",
		"spec_version":      "2024-11-05",
		"auth_type":         "bearer",
		"allow_all_keys":    true,
		"status":            "healthy",
		"last_health_check": "2026-02-01T00:00:00Z",
		"created_by":        "admin",
	}
	for attr, want := range checks {
		if got := d.Get(attr); got != want {
			t.Errorf("attr %s: expected %v, got %v", attr, want, got)
		}
	}
	groups := d.Get("mcp_access_groups").([]interface{})
	if len(groups) != 1 || groups[0] != "dev" {
		t.Errorf("unexpected access groups: %v", groups)
	}
	tools := d.Get("allowed_tools").([]interface{})
	if len(tools) != 1 || tools[0] != "list_repos" {
		t.Errorf("unexpected allowed tools: %v", tools)
	}
	headers := d.Get("extra_headers").([]interface{})
	if len(headers) != 1 || headers[0] != "x-request-id" {
		t.Errorf("unexpected extra headers: %v", headers)
	}
}

func TestDataSourceMCPServerReadNotFound(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		w.Write([]byte(`{"detail": {"error": "MCP server not found"}}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMMCPServer().Schema, map[string]interface{}{
		"server_id": "srv-missing",
	})

	if err := dataSourceLiteLLMMCPServerRead(d, client); err == nil {
		t.Fatal("expected error for missing MCP server, got nil")
	}
}

func TestDataSourceMCPServersRead(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/v1/mcp/server" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		if got := r.URL.Query().Get("team_id"); got != "team-1" {
			t.Errorf("expected team_id 'team-1', got %q", got)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`[
			{"server_id": "srv-1", "server_name": "one", "url": "https://one.example.com", "transport": "http", "status": "healthy", "allow_all_keys": false},
			{"server_id": "srv-2", "server_name": "two", "url": "https://two.example.com", "transport": "sse", "status": "unknown", "allow_all_keys": true}
		]`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMMCPServers().Schema, map[string]interface{}{
		"team_id": "team-1",
	})

	if err := dataSourceLiteLLMMCPServersRead(d, client); err != nil {
		t.Fatalf("read failed: %v", err)
	}

	if d.Id() != "team-1" {
		t.Fatalf("expected ID 'team-1', got %q", d.Id())
	}
	ids := d.Get("ids").([]interface{})
	if len(ids) != 2 || ids[0] != "srv-1" || ids[1] != "srv-2" {
		t.Errorf("unexpected ids: %v", ids)
	}
	servers := d.Get("mcp_servers").([]interface{})
	if len(servers) != 2 {
		t.Fatalf("expected 2 servers, got %d", len(servers))
	}
	first := servers[0].(map[string]interface{})
	if first["server_name"] != "one" || first["transport"] != "http" || first["allow_all_keys"] != false {
		t.Errorf("unexpected first server: %v", first)
	}
	second := servers[1].(map[string]interface{})
	if second["status"] != "unknown" || second["allow_all_keys"] != true {
		t.Errorf("unexpected second server: %v", second)
	}
}
