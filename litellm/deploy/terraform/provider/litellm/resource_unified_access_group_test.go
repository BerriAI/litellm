package litellm

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func unifiedAccessGroupTestData(t *testing.T, raw map[string]interface{}) *schema.ResourceData {
	t.Helper()
	return schema.TestResourceDataRaw(t, resourceLiteLLMUnifiedAccessGroup().Schema, raw)
}

func unifiedAccessGroupJSON(id string) []byte {
	description := "prod access"
	createdBy := "admin"
	body, _ := json.Marshal(unifiedAccessGroupResponse{
		AccessGroupID:      id,
		AccessGroupName:    "prod-group",
		Description:        &description,
		AccessModelNames:   []string{"gpt-4"},
		AccessMCPServerIDs: []string{"mcp-1"},
		AccessAgentIDs:     []string{"agent-1"},
		AssignedTeamIDs:    []string{"team-1"},
		AssignedKeyIDs:     []string{"key-1"},
		CreatedAt:          "2026-01-01T00:00:00Z",
		CreatedBy:          &createdBy,
		UpdatedAt:          "2026-01-02T00:00:00Z",
	})
	return body
}

func TestUnifiedAccessGroupCreate(t *testing.T) {
	var createPayload map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method + " " + r.URL.Path {
		case "POST /v1/unified_access_group":
			if err := json.NewDecoder(r.Body).Decode(&createPayload); err != nil {
				t.Errorf("failed to decode create payload: %v", err)
			}
			w.Write(unifiedAccessGroupJSON("uag-123"))
		case "GET /v1/unified_access_group/uag-123":
			w.Write(unifiedAccessGroupJSON("uag-123"))
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := unifiedAccessGroupTestData(t, map[string]interface{}{
		"access_group_name":  "prod-group",
		"description":        "prod access",
		"access_model_names": []interface{}{"gpt-4"},
		"assigned_team_ids":  []interface{}{"team-1"},
	})

	if err := resourceLiteLLMUnifiedAccessGroupCreate(d, client); err != nil {
		t.Fatalf("create failed: %v", err)
	}

	if createPayload["access_group_name"] != "prod-group" {
		t.Fatalf("expected access_group_name 'prod-group' in payload, got %v", createPayload["access_group_name"])
	}
	if createPayload["description"] != "prod access" {
		t.Fatalf("expected description 'prod access' in payload, got %v", createPayload["description"])
	}
	if !reflect.DeepEqual(createPayload["access_model_names"], []interface{}{"gpt-4"}) {
		t.Fatalf("expected access_model_names [gpt-4] in payload, got %v", createPayload["access_model_names"])
	}
	if !reflect.DeepEqual(createPayload["assigned_team_ids"], []interface{}{"team-1"}) {
		t.Fatalf("expected assigned_team_ids [team-1] in payload, got %v", createPayload["assigned_team_ids"])
	}
	if d.Id() != "uag-123" {
		t.Fatalf("expected ID 'uag-123', got %q", d.Id())
	}
	if d.Get("access_group_id").(string) != "uag-123" {
		t.Fatalf("expected access_group_id 'uag-123', got %v", d.Get("access_group_id"))
	}
	if d.Get("created_by").(string) != "admin" {
		t.Fatalf("expected created_by 'admin', got %v", d.Get("created_by"))
	}
}

func TestUnifiedAccessGroupRead(t *testing.T) {
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
	d := unifiedAccessGroupTestData(t, map[string]interface{}{})
	d.SetId("uag-123")

	if err := resourceLiteLLMUnifiedAccessGroupRead(d, client); err != nil {
		t.Fatalf("read failed: %v", err)
	}

	if d.Get("access_group_name").(string) != "prod-group" {
		t.Fatalf("expected access_group_name 'prod-group', got %v", d.Get("access_group_name"))
	}
	if d.Get("description").(string) != "prod access" {
		t.Fatalf("expected description 'prod access', got %v", d.Get("description"))
	}
	if !reflect.DeepEqual(d.Get("access_mcp_server_ids"), []interface{}{"mcp-1"}) {
		t.Fatalf("expected access_mcp_server_ids [mcp-1], got %v", d.Get("access_mcp_server_ids"))
	}
	if !reflect.DeepEqual(d.Get("assigned_key_ids"), []interface{}{"key-1"}) {
		t.Fatalf("expected assigned_key_ids [key-1], got %v", d.Get("assigned_key_ids"))
	}
	if d.Get("created_at").(string) != "2026-01-01T00:00:00Z" {
		t.Fatalf("expected created_at '2026-01-01T00:00:00Z', got %v", d.Get("created_at"))
	}
}

func TestUnifiedAccessGroupReadNotFound(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := unifiedAccessGroupTestData(t, map[string]interface{}{})
	d.SetId("uag-gone")

	if err := resourceLiteLLMUnifiedAccessGroupRead(d, client); err != nil {
		t.Fatalf("expected nil error on 404, got: %v", err)
	}
	if d.Id() != "" {
		t.Fatalf("expected ID to be cleared on 404, got %q", d.Id())
	}
}

func TestUnifiedAccessGroupUpdate(t *testing.T) {
	var updatePayload map[string]interface{}
	var updatePath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case "PUT":
			updatePath = r.URL.Path
			if err := json.NewDecoder(r.Body).Decode(&updatePayload); err != nil {
				t.Errorf("failed to decode update payload: %v", err)
			}
			w.Write(unifiedAccessGroupJSON("uag-123"))
		case "GET":
			w.Write(unifiedAccessGroupJSON("uag-123"))
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := unifiedAccessGroupTestData(t, map[string]interface{}{
		"access_group_name":  "renamed-group",
		"access_model_names": []interface{}{"gpt-4", "claude-3"},
	})
	d.SetId("uag-123")

	if err := resourceLiteLLMUnifiedAccessGroupUpdate(d, client); err != nil {
		t.Fatalf("update failed: %v", err)
	}

	if updatePath != "/v1/unified_access_group/uag-123" {
		t.Fatalf("expected update path '/v1/unified_access_group/uag-123', got %q", updatePath)
	}
	if updatePayload["access_group_name"] != "renamed-group" {
		t.Fatalf("expected access_group_name 'renamed-group' in payload, got %v", updatePayload["access_group_name"])
	}
	if !reflect.DeepEqual(updatePayload["access_model_names"], []interface{}{"gpt-4", "claude-3"}) {
		t.Fatalf("expected access_model_names [gpt-4 claude-3] in payload, got %v", updatePayload["access_model_names"])
	}
}

func TestUnifiedAccessGroupDelete(t *testing.T) {
	var deleteMethod, deletePath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		deleteMethod = r.Method
		deletePath = r.URL.Path
		w.WriteHeader(http.StatusNoContent)
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := unifiedAccessGroupTestData(t, map[string]interface{}{})
	d.SetId("uag-123")

	if err := resourceLiteLLMUnifiedAccessGroupDelete(d, client); err != nil {
		t.Fatalf("delete failed: %v", err)
	}

	if deleteMethod != "DELETE" || deletePath != "/v1/unified_access_group/uag-123" {
		t.Fatalf("expected DELETE /v1/unified_access_group/uag-123, got %s %s", deleteMethod, deletePath)
	}
	if d.Id() != "" {
		t.Fatalf("expected ID to be cleared after delete, got %q", d.Id())
	}
}
