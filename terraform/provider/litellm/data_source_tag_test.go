package litellm

import (
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func TestDataSourceLiteLLMTagRead(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/tag/info" || r.Method != http.MethodPost {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.Write([]byte(tagInfoBody("prod")))
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMTag().Schema, map[string]interface{}{"name": "prod"})

	if err := dataSourceLiteLLMTagRead(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("read failed: %v", err)
	}

	if d.Id() != "prod" {
		t.Fatalf("expected ID 'prod', got %q", d.Id())
	}
	checks := map[string]interface{}{
		"description": "Production traffic",
		"budget_id":   "bud-1",
		"max_budget":  50.5,
		"tpm_limit":   1000,
		"created_at":  "2026-01-01T00:00:00",
		"created_by":  "admin",
	}
	for key, want := range checks {
		if got := d.Get(key); got != want {
			t.Errorf("expected %s %v, got %v", key, want, got)
		}
	}
}

func TestDataSourceLiteLLMTagRead_NotFound(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMTag().Schema, map[string]interface{}{"name": "gone"})

	if err := dataSourceLiteLLMTagRead(d, NewClient(srv.URL, "test-key", true)); err == nil {
		t.Fatal("expected error for missing tag, got nil")
	}
}

func TestDataSourceLiteLLMTagsRead(t *testing.T) {
	var gotQuery string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/tag/list" || r.Method != http.MethodGet {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		gotQuery = r.URL.RawQuery
		w.Write([]byte(`[
			{
				"name": "prod",
				"description": "Production traffic",
				"models": ["model-1"],
				"created_at": "2026-01-01T00:00:00",
				"updated_at": "2026-01-02T00:00:00",
				"created_by": "admin",
				"litellm_budget_table": {"budget_id": "bud-1", "max_budget": 50.5}
			},
			{
				"name": "dynamic-tag",
				"description": "This is just a spend tag that was passed dynamically in a request.",
				"models": null,
				"created_at": "2026-02-01T00:00:00",
				"updated_at": "2026-02-02T00:00:00"
			}
		]`))
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMTags().Schema, map[string]interface{}{
		"start_date": "2026-01-01",
		"end_date":   "2026-03-01",
	})

	if err := dataSourceLiteLLMTagsRead(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("read failed: %v", err)
	}

	if gotQuery != "start_date=2026-01-01&end_date=2026-03-01" {
		t.Errorf("expected date filter query params, got %q", gotQuery)
	}
	if !reflect.DeepEqual(d.Get("ids"), []interface{}{"prod", "dynamic-tag"}) {
		t.Errorf("expected ids ['prod', 'dynamic-tag'], got %v", d.Get("ids"))
	}
	if got := d.Get("tags.#").(int); got != 2 {
		t.Fatalf("expected 2 tags, got %d", got)
	}
	if got := d.Get("tags.0.name").(string); got != "prod" {
		t.Errorf("expected tags.0.name 'prod', got %q", got)
	}
	if got := d.Get("tags.0.max_budget").(float64); got != 50.5 {
		t.Errorf("expected tags.0.max_budget 50.5, got %v", got)
	}
	if got := d.Get("tags.1.name").(string); got != "dynamic-tag" {
		t.Errorf("expected tags.1.name 'dynamic-tag', got %q", got)
	}
	if got := d.Get("tags.1.budget_id").(string); got != "" {
		t.Errorf("expected empty budget_id for dynamic tag, got %q", got)
	}
}
