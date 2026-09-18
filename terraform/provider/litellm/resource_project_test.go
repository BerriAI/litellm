package litellm

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

const projectInfoBody = `{
	"project_id": "proj-123",
	"project_alias": "ml-experiments",
	"description": "ML experimentation project",
	"team_id": "team-1",
	"budget_id": "bud-9",
	"metadata": {"env": "prod", "tags": ["research", "gpu"]},
	"models": ["gpt-4"],
	"spend": 12.5,
	"blocked": false,
	"created_by": "admin",
	"updated_by": "admin",
	"created_at": "2026-01-01T00:00:00",
	"updated_at": "2026-01-02T00:00:00",
	"litellm_budget_table": {
		"max_budget": 100.0,
		"soft_budget": 80.0,
		"max_parallel_requests": 10,
		"tpm_limit": 5000,
		"rpm_limit": 500,
		"budget_duration": "30d"
	}
}`

func TestResourceLiteLLMProjectCreate(t *testing.T) {
	var createPayload map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/project/new":
			if err := json.NewDecoder(r.Body).Decode(&createPayload); err != nil {
				t.Errorf("failed to decode create payload: %v", err)
			}
			w.Write([]byte(projectInfoBody))
		case "/project/info":
			if got := r.URL.Query().Get("project_id"); got != "proj-123" {
				t.Errorf("expected project_id query 'proj-123', got %q", got)
			}
			w.Write([]byte(projectInfoBody))
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resourceLiteLLMProject().Schema, map[string]interface{}{
		"team_id":       "team-1",
		"project_alias": "ml-experiments",
		"description":   "ML experimentation project",
		"models":        []interface{}{"gpt-4"},
		"metadata":      map[string]interface{}{"env": "prod"},
		"tags":          []interface{}{"research", "gpu"},
		"max_budget":    100.0,
		"tpm_limit":     5000,
	})

	if err := resourceLiteLLMProjectCreate(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("create failed: %v", err)
	}

	if d.Id() != "proj-123" {
		t.Fatalf("expected ID 'proj-123', got %q", d.Id())
	}
	if createPayload["team_id"] != "team-1" {
		t.Errorf("expected payload team_id 'team-1', got %v", createPayload["team_id"])
	}
	if createPayload["project_alias"] != "ml-experiments" {
		t.Errorf("expected payload project_alias, got %v", createPayload["project_alias"])
	}
	if !reflect.DeepEqual(createPayload["models"], []interface{}{"gpt-4"}) {
		t.Errorf("expected payload models ['gpt-4'], got %v", createPayload["models"])
	}
	if !reflect.DeepEqual(createPayload["tags"], []interface{}{"research", "gpu"}) {
		t.Errorf("expected payload tags, got %v", createPayload["tags"])
	}
	if createPayload["max_budget"] != 100.0 {
		t.Errorf("expected payload max_budget 100.0, got %v", createPayload["max_budget"])
	}
	if createPayload["tpm_limit"] != float64(5000) {
		t.Errorf("expected payload tpm_limit 5000, got %v", createPayload["tpm_limit"])
	}
	if _, ok := createPayload["project_id"]; ok {
		t.Errorf("create payload must not contain project_id, got %v", createPayload["project_id"])
	}
}

func TestResourceLiteLLMProjectRead(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/project/info" || r.Method != http.MethodGet {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.Write([]byte(projectInfoBody))
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resourceLiteLLMProject().Schema, map[string]interface{}{
		"team_id": "team-1",
	})
	d.SetId("proj-123")

	if err := resourceLiteLLMProjectRead(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("read failed: %v", err)
	}

	checks := map[string]interface{}{
		"project_alias":         "ml-experiments",
		"description":           "ML experimentation project",
		"team_id":               "team-1",
		"budget_id":             "bud-9",
		"spend":                 12.5,
		"max_budget":            100.0,
		"soft_budget":           80.0,
		"max_parallel_requests": 10,
		"tpm_limit":             5000,
		"rpm_limit":             500,
		"budget_duration":       "30d",
		"created_by":            "admin",
		"created_at":            "2026-01-01T00:00:00",
	}
	for key, want := range checks {
		if got := d.Get(key); got != want {
			t.Errorf("expected %s %v, got %v", key, want, got)
		}
	}
	if !reflect.DeepEqual(d.Get("tags"), []interface{}{"research", "gpu"}) {
		t.Errorf("expected tags extracted from metadata, got %v", d.Get("tags"))
	}
	wantMetadata := map[string]interface{}{"env": "prod"}
	if !reflect.DeepEqual(d.Get("metadata"), wantMetadata) {
		t.Errorf("expected metadata without injected tags key, got %v", d.Get("metadata"))
	}
}

func TestResourceLiteLLMProjectRead_404ClearsID(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resourceLiteLLMProject().Schema, map[string]interface{}{
		"team_id": "team-1",
	})
	d.SetId("gone")

	if err := resourceLiteLLMProjectRead(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("expected nil error on 404, got: %v", err)
	}
	if d.Id() != "" {
		t.Fatalf("expected ID cleared on 404, got %q", d.Id())
	}
}

func TestResourceLiteLLMProjectUpdate(t *testing.T) {
	var updatePayload map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/project/update":
			if r.Method != http.MethodPost {
				t.Errorf("expected POST for update, got %s", r.Method)
			}
			if err := json.NewDecoder(r.Body).Decode(&updatePayload); err != nil {
				t.Errorf("failed to decode update payload: %v", err)
			}
			w.Write([]byte(projectInfoBody))
		case "/project/info":
			w.Write([]byte(projectInfoBody))
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resourceLiteLLMProject().Schema, map[string]interface{}{
		"team_id":       "team-1",
		"project_alias": "renamed-project",
		"rpm_limit":     900,
	})
	d.SetId("proj-123")

	if err := resourceLiteLLMProjectUpdate(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("update failed: %v", err)
	}

	if updatePayload["project_id"] != "proj-123" {
		t.Errorf("expected update payload project_id 'proj-123', got %v", updatePayload["project_id"])
	}
	if updatePayload["project_alias"] != "renamed-project" {
		t.Errorf("expected updated project_alias in payload, got %v", updatePayload["project_alias"])
	}
	if updatePayload["rpm_limit"] != float64(900) {
		t.Errorf("expected rpm_limit 900 in payload, got %v", updatePayload["rpm_limit"])
	}
}

func TestResourceLiteLLMProjectDelete(t *testing.T) {
	var deletePayload map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/project/delete" || r.Method != http.MethodDelete {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&deletePayload); err != nil {
			t.Errorf("failed to decode delete payload: %v", err)
		}
		w.Write([]byte(`[]`))
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resourceLiteLLMProject().Schema, map[string]interface{}{
		"team_id": "team-1",
	})
	d.SetId("proj-123")

	if err := resourceLiteLLMProjectDelete(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("delete failed: %v", err)
	}

	if !reflect.DeepEqual(deletePayload["project_ids"], []interface{}{"proj-123"}) {
		t.Errorf("expected delete payload project_ids ['proj-123'], got %v", deletePayload["project_ids"])
	}
	if d.Id() != "" {
		t.Fatalf("expected ID cleared after delete, got %q", d.Id())
	}
}
