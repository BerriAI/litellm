package litellm

import (
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func TestUnifiedAccessGroupDataSourceRead(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" || r.URL.Path != "/v1/unified_access_group/uag-123" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.Write(unifiedAccessGroupJSON("uag-123"))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMUnifiedAccessGroup().Schema, map[string]interface{}{
		"access_group_id": "uag-123",
	})

	if err := dataSourceLiteLLMUnifiedAccessGroupRead(d, client); err != nil {
		t.Fatalf("data source read failed: %v", err)
	}

	if d.Id() != "uag-123" {
		t.Fatalf("expected ID 'uag-123', got %q", d.Id())
	}
	if d.Get("access_group_name").(string) != "prod-group" {
		t.Fatalf("expected access_group_name 'prod-group', got %v", d.Get("access_group_name"))
	}
	if d.Get("description").(string) != "prod access" {
		t.Fatalf("expected description 'prod access', got %v", d.Get("description"))
	}
	if !reflect.DeepEqual(d.Get("access_model_names"), []interface{}{"gpt-4"}) {
		t.Fatalf("expected access_model_names [gpt-4], got %v", d.Get("access_model_names"))
	}
	if !reflect.DeepEqual(d.Get("assigned_team_ids"), []interface{}{"team-1"}) {
		t.Fatalf("expected assigned_team_ids [team-1], got %v", d.Get("assigned_team_ids"))
	}
}

func TestUnifiedAccessGroupDataSourceReadNotFound(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMUnifiedAccessGroup().Schema, map[string]interface{}{
		"access_group_id": "missing",
	})

	if err := dataSourceLiteLLMUnifiedAccessGroupRead(d, client); err == nil {
		t.Fatal("expected error for missing unified access group, got nil")
	}
}

func TestUnifiedAccessGroupsDataSourceRead(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" || r.URL.Path != "/v1/unified_access_group" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.Write([]byte(`[` +
			`{"access_group_id": "uag-1", "access_group_name": "group-one", "description": "first",` +
			` "access_model_names": ["gpt-4"], "access_mcp_server_ids": [], "access_agent_ids": [],` +
			` "assigned_team_ids": ["team-1"], "assigned_key_ids": [],` +
			` "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-02T00:00:00Z"},` +
			`{"access_group_id": "uag-2", "access_group_name": "group-two",` +
			` "access_model_names": [], "access_mcp_server_ids": ["mcp-1"], "access_agent_ids": [],` +
			` "assigned_team_ids": [], "assigned_key_ids": [],` +
			` "created_at": "2026-01-03T00:00:00Z", "updated_at": "2026-01-04T00:00:00Z"}]`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMUnifiedAccessGroups().Schema, map[string]interface{}{})

	if err := dataSourceLiteLLMUnifiedAccessGroupsRead(d, client); err != nil {
		t.Fatalf("data source read failed: %v", err)
	}

	groups := d.Get("access_groups").([]interface{})
	if len(groups) != 2 {
		t.Fatalf("expected 2 unified access groups, got %d", len(groups))
	}
	first := groups[0].(map[string]interface{})
	if first["access_group_id"] != "uag-1" {
		t.Fatalf("expected first access_group_id 'uag-1', got %v", first["access_group_id"])
	}
	if first["access_group_name"] != "group-one" {
		t.Fatalf("expected first access_group_name 'group-one', got %v", first["access_group_name"])
	}
	if first["description"] != "first" {
		t.Fatalf("expected first description 'first', got %v", first["description"])
	}
	second := groups[1].(map[string]interface{})
	if !reflect.DeepEqual(second["access_mcp_server_ids"], []interface{}{"mcp-1"}) {
		t.Fatalf("expected second access_mcp_server_ids [mcp-1], got %v", second["access_mcp_server_ids"])
	}
	if !reflect.DeepEqual(d.Get("ids"), []interface{}{"uag-1", "uag-2"}) {
		t.Fatalf("expected ids [uag-1 uag-2], got %v", d.Get("ids"))
	}
}
