package litellm

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func TestDataSourceOrganizationRead(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/organization/info" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		if got := r.URL.Query().Get("organization_id"); got != "org-123" {
			t.Errorf("expected organization_id 'org-123', got %q", got)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{
			"organization_id": "org-123",
			"organization_alias": "acme-org",
			"budget_id": "budget-1",
			"models": ["gpt-4o"],
			"spend": 77.5,
			"metadata": {"env": "prod"},
			"created_at": "2026-01-01T00:00:00Z",
			"updated_at": "2026-02-01T00:00:00Z",
			"litellm_budget_table": {
				"max_budget": 1000,
				"soft_budget": 800,
				"tpm_limit": 50000,
				"rpm_limit": 500,
				"max_parallel_requests": 20,
				"budget_duration": "30d"
			}
		}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMOrganization().Schema, map[string]interface{}{
		"organization_id": "org-123",
	})

	if err := dataSourceLiteLLMOrganizationRead(d, client); err != nil {
		t.Fatalf("read failed: %v", err)
	}

	if d.Id() != "org-123" {
		t.Fatalf("expected ID 'org-123', got %q", d.Id())
	}
	checks := map[string]interface{}{
		"organization_alias":    "acme-org",
		"budget_id":             "budget-1",
		"spend":                 77.5,
		"max_budget":            1000.0,
		"soft_budget":           800.0,
		"tpm_limit":             50000,
		"rpm_limit":             500,
		"max_parallel_requests": 20,
		"budget_duration":       "30d",
		"created_at":            "2026-01-01T00:00:00Z",
	}
	for attr, want := range checks {
		if got := d.Get(attr); got != want {
			t.Errorf("attr %s: expected %v, got %v", attr, want, got)
		}
	}
	metadata := d.Get("metadata").(map[string]interface{})
	if metadata["env"] != "prod" {
		t.Errorf("unexpected metadata: %v", metadata)
	}
}

func TestDataSourceOrganizationsRead(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/organization/list" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		if got := r.URL.Query().Get("org_alias"); got != "acme" {
			t.Errorf("expected org_alias 'acme', got %q", got)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`[
			{"organization_id": "org-1", "organization_alias": "acme", "spend": 1.5, "litellm_budget_table": {"max_budget": 100, "tpm_limit": 10, "rpm_limit": 5, "budget_duration": "7d"}},
			{"organization_id": "org-2", "organization_alias": "acme-eu", "spend": 0}
		]`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMOrganizations().Schema, map[string]interface{}{
		"org_alias": "acme",
	})

	if err := dataSourceLiteLLMOrganizationsRead(d, client); err != nil {
		t.Fatalf("read failed: %v", err)
	}

	if d.Id() != "acme" {
		t.Fatalf("expected ID 'acme', got %q", d.Id())
	}
	ids := d.Get("ids").([]interface{})
	if len(ids) != 2 || ids[0] != "org-1" || ids[1] != "org-2" {
		t.Errorf("unexpected ids: %v", ids)
	}
	orgs := d.Get("organizations").([]interface{})
	if len(orgs) != 2 {
		t.Fatalf("expected 2 organizations, got %d", len(orgs))
	}
	first := orgs[0].(map[string]interface{})
	if first["organization_alias"] != "acme" || first["max_budget"] != 100.0 || first["budget_duration"] != "7d" {
		t.Errorf("unexpected first organization: %v", first)
	}
	second := orgs[1].(map[string]interface{})
	if second["organization_id"] != "org-2" || second["max_budget"] != 0.0 {
		t.Errorf("unexpected second organization: %v", second)
	}
}
