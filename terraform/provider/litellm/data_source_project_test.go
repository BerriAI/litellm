package litellm

import (
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func TestDataSourceLiteLLMProjectRead(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/project/info" || r.Method != http.MethodGet {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		if got := r.URL.Query().Get("project_id"); got != "proj-123" {
			t.Errorf("expected project_id query 'proj-123', got %q", got)
		}
		w.Write([]byte(projectInfoBody))
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMProject().Schema, map[string]interface{}{
		"project_id": "proj-123",
	})

	if err := dataSourceLiteLLMProjectRead(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("read failed: %v", err)
	}

	if d.Id() != "proj-123" {
		t.Fatalf("expected ID 'proj-123', got %q", d.Id())
	}
	checks := map[string]interface{}{
		"project_alias":   "ml-experiments",
		"description":     "ML experimentation project",
		"team_id":         "team-1",
		"budget_id":       "bud-9",
		"spend":           12.5,
		"max_budget":      100.0,
		"tpm_limit":       5000,
		"budget_duration": "30d",
		"created_by":      "admin",
	}
	for key, want := range checks {
		if got := d.Get(key); got != want {
			t.Errorf("expected %s %v, got %v", key, want, got)
		}
	}
	if !reflect.DeepEqual(d.Get("models"), []interface{}{"gpt-4"}) {
		t.Errorf("expected models ['gpt-4'], got %v", d.Get("models"))
	}
}

func TestDataSourceLiteLLMProjectRead_NotFound(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMProject().Schema, map[string]interface{}{
		"project_id": "gone",
	})

	if err := dataSourceLiteLLMProjectRead(d, NewClient(srv.URL, "test-key", true)); err == nil {
		t.Fatal("expected error for missing project, got nil")
	}
}

func TestDataSourceLiteLLMProjectsRead(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/project/list" || r.Method != http.MethodGet {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.Write([]byte(`[
			` + projectInfoBody + `,
			{"project_id": "proj-456", "project_alias": "second", "team_id": "team-2", "models": [], "spend": 0.0}
		]`))
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMProjects().Schema, map[string]interface{}{})

	if err := dataSourceLiteLLMProjectsRead(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("read failed: %v", err)
	}

	if !reflect.DeepEqual(d.Get("ids"), []interface{}{"proj-123", "proj-456"}) {
		t.Errorf("expected ids ['proj-123', 'proj-456'], got %v", d.Get("ids"))
	}
	if got := d.Get("projects.#").(int); got != 2 {
		t.Fatalf("expected 2 projects, got %d", got)
	}
	if got := d.Get("projects.0.project_alias").(string); got != "ml-experiments" {
		t.Errorf("expected projects.0.project_alias 'ml-experiments', got %q", got)
	}
	if got := d.Get("projects.0.spend").(float64); got != 12.5 {
		t.Errorf("expected projects.0.spend 12.5, got %v", got)
	}
	if got := d.Get("projects.1.team_id").(string); got != "team-2" {
		t.Errorf("expected projects.1.team_id 'team-2', got %q", got)
	}
}
