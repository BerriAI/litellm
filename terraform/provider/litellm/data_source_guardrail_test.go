package litellm

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func TestDataSourceGuardrailRead(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" || r.URL.Path != "/guardrails/gid-1/info" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{
			"guardrail_id": "gid-1",
			"guardrail_name": "guard1",
			"guardrail_info": {"description": "pii guard"},
			"guardrail_definition_location": "db",
			"created_at": "2026-01-01T00:00:00Z",
			"updated_at": "2026-01-02T00:00:00Z"
		}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMGuardrail().Schema, map[string]interface{}{
		"guardrail_id": "gid-1",
	})

	if err := dataSourceLiteLLMGuardrailRead(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if d.Id() != "gid-1" {
		t.Fatalf("expected ID 'gid-1', got %q", d.Id())
	}
	if got := d.Get("guardrail_name").(string); got != "guard1" {
		t.Errorf("expected guardrail_name 'guard1', got %q", got)
	}
	if got := d.Get("guardrail_definition_location").(string); got != "db" {
		t.Errorf("expected guardrail_definition_location 'db', got %q", got)
	}
	info := d.Get("guardrail_info").(map[string]interface{})
	if info["description"] != "pii guard" {
		t.Errorf("expected guardrail_info from API, got: %v", info)
	}
}

func TestDataSourceGuardrailsRead(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" || r.URL.Path != "/guardrails/list" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"guardrails": [
			{"guardrail_id": "gid-1", "guardrail_name": "guard1", "guardrail_definition_location": "db"},
			{"guardrail_id": "gid-2", "guardrail_name": "guard2", "guardrail_definition_location": "config"}
		]}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMGuardrails().Schema, map[string]interface{}{})

	if err := dataSourceLiteLLMGuardrailsRead(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}

	guardrails := d.Get("guardrails").([]interface{})
	if len(guardrails) != 2 {
		t.Fatalf("expected 2 guardrails, got %d", len(guardrails))
	}
	first := guardrails[0].(map[string]interface{})
	if first["guardrail_id"] != "gid-1" || first["guardrail_name"] != "guard1" {
		t.Errorf("unexpected first guardrail: %v", first)
	}
	ids := d.Get("ids").([]interface{})
	if len(ids) != 2 || ids[0] != "gid-1" || ids[1] != "gid-2" {
		t.Errorf("unexpected ids: %v", ids)
	}
}
