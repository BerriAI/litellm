package litellm

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func TestDataSourceTeamRead(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/team/info" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		if got := r.URL.Query().Get("team_id"); got != "team-123" {
			t.Errorf("expected team_id 'team-123', got %q", got)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{
			"team_id": "team-123",
			"team_info": {
				"team_id": "team-123",
				"team_alias": "ml-team",
				"organization_id": "org-1",
				"models": ["gpt-4o"],
				"metadata": {"env": "prod", "tags": ["ml"], "soft_budget_alerting_emails": ["ops@example.com"]},
				"tpm_limit": 5000,
				"rpm_limit": 100,
				"max_budget": 250.5,
				"soft_budget": 200,
				"spend": 42.25,
				"budget_duration": "30d",
				"blocked": true,
				"team_member_permissions": ["/key/generate"],
				"created_at": "2026-01-01T00:00:00Z",
				"updated_at": "2026-02-01T00:00:00Z"
			}
		}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMTeam().Schema, map[string]interface{}{
		"team_id": "team-123",
	})

	if err := dataSourceLiteLLMTeamRead(d, client); err != nil {
		t.Fatalf("read failed: %v", err)
	}

	if d.Id() != "team-123" {
		t.Fatalf("expected ID 'team-123', got %q", d.Id())
	}
	checks := map[string]interface{}{
		"team_alias":      "ml-team",
		"organization_id": "org-1",
		"tpm_limit":       5000,
		"rpm_limit":       100,
		"max_budget":      250.5,
		"soft_budget":     200.0,
		"spend":           42.25,
		"budget_duration": "30d",
		"blocked":         true,
	}
	for attr, want := range checks {
		if got := d.Get(attr); got != want {
			t.Errorf("attr %s: expected %v, got %v", attr, want, got)
		}
	}
	tags := d.Get("tags").([]interface{})
	if len(tags) != 1 || tags[0] != "ml" {
		t.Errorf("unexpected tags: %v", tags)
	}
	emails := d.Get("soft_budget_alerting_emails").([]interface{})
	if len(emails) != 1 || emails[0] != "ops@example.com" {
		t.Errorf("unexpected alerting emails: %v", emails)
	}
	metadata := d.Get("metadata").(map[string]interface{})
	if metadata["env"] != "prod" || len(metadata) != 1 {
		t.Errorf("unexpected metadata: %v", metadata)
	}
	perms := d.Get("team_member_permissions").([]interface{})
	if len(perms) != 1 || perms[0] != "/key/generate" {
		t.Errorf("unexpected permissions: %v", perms)
	}
}

func TestDataSourceTeamsRead(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/team/list" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		if got := r.URL.Query().Get("organization_id"); got != "org-1" {
			t.Errorf("expected organization_id 'org-1', got %q", got)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`[
			{"team_id": "team-1", "team_alias": "alpha", "organization_id": "org-1", "spend": 5, "max_budget": 50, "tpm_limit": 100, "rpm_limit": 10, "models": ["m1"], "blocked": false},
			{"team_id": "team-2", "team_alias": "beta", "organization_id": "org-1", "blocked": true}
		]`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMTeams().Schema, map[string]interface{}{
		"organization_id": "org-1",
	})

	if err := dataSourceLiteLLMTeamsRead(d, client); err != nil {
		t.Fatalf("read failed: %v", err)
	}

	ids := d.Get("ids").([]interface{})
	if len(ids) != 2 || ids[0] != "team-1" || ids[1] != "team-2" {
		t.Errorf("unexpected ids: %v", ids)
	}
	teams := d.Get("teams").([]interface{})
	if len(teams) != 2 {
		t.Fatalf("expected 2 teams, got %d", len(teams))
	}
	first := teams[0].(map[string]interface{})
	if first["team_alias"] != "alpha" || first["max_budget"] != 50.0 || first["tpm_limit"] != 100 {
		t.Errorf("unexpected first team: %v", first)
	}
	second := teams[1].(map[string]interface{})
	if second["blocked"] != true || second["max_budget"] != 0.0 {
		t.Errorf("unexpected second team: %v", second)
	}
}

func TestDataSourceTeamsReadError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"error": "boom"}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMTeams().Schema, map[string]interface{}{})

	if err := dataSourceLiteLLMTeamsRead(d, client); err == nil {
		t.Fatal("expected error on server failure, got nil")
	}
}
