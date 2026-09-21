package litellm

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func TestDataSourceKeyRead(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/key/info" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		if got := r.URL.Query().Get("key"); got != "43d0a3c1b9dc2739952a8ffc4ee4f41ea34da6587cbc717c3a51185b9fac611c" {
			t.Errorf("expected key query param to be the token hash, got %q", got)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{
			"key": "sk-raw-secret",
			"info": {
				"token": "hashed-token-123",
				"key_name": "sk-...cret",
				"key_alias": "ci-key",
				"spend": 12.5,
				"max_budget": 100,
				"models": ["gpt-4o", "claude-3"],
				"user_id": "user-1",
				"team_id": "team-1",
				"org_id": "org-1",
				"tpm_limit": 1000,
				"rpm_limit": 60,
				"max_parallel_requests": 5,
				"budget_duration": "30d",
				"metadata": {"env": "prod", "tags": ["alpha", "beta"]},
				"blocked": true,
				"expires": "2027-01-01T00:00:00Z",
				"created_at": "2026-01-01T00:00:00Z",
				"updated_at": "2026-02-01T00:00:00Z"
			}
		}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMKey().Schema, map[string]interface{}{
		"key": "sk-raw-secret",
	})

	if err := dataSourceLiteLLMKeyRead(d, client); err != nil {
		t.Fatalf("read failed: %v", err)
	}

	if d.Id() != "hashed-token-123" {
		t.Fatalf("expected ID 'hashed-token-123', got %q", d.Id())
	}
	checks := map[string]interface{}{
		"token_id":              "hashed-token-123",
		"key_name":              "sk-...cret",
		"key_alias":             "ci-key",
		"spend":                 12.5,
		"max_budget":            100.0,
		"user_id":               "user-1",
		"team_id":               "team-1",
		"organization_id":       "org-1",
		"tpm_limit":             1000,
		"rpm_limit":             60,
		"max_parallel_requests": 5,
		"budget_duration":       "30d",
		"blocked":               true,
		"expires":               "2027-01-01T00:00:00Z",
	}
	for attr, want := range checks {
		if got := d.Get(attr); got != want {
			t.Errorf("attr %s: expected %v, got %v", attr, want, got)
		}
	}
	models := d.Get("models").([]interface{})
	if len(models) != 2 || models[0] != "gpt-4o" {
		t.Errorf("unexpected models: %v", models)
	}
	tags := d.Get("tags").([]interface{})
	if len(tags) != 2 || tags[0] != "alpha" {
		t.Errorf("unexpected tags: %v", tags)
	}
	metadata := d.Get("metadata").(map[string]interface{})
	if metadata["env"] != "prod" {
		t.Errorf("unexpected metadata: %v", metadata)
	}
	if _, hasTags := metadata["tags"]; hasTags {
		t.Errorf("non-string metadata value should not be in the metadata map: %v", metadata)
	}
}

func TestDataSourceKeyReadNotFound(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		w.Write([]byte(`{"detail": {"error": "key not found"}}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMKey().Schema, map[string]interface{}{
		"key": "sk-missing",
	})

	if err := dataSourceLiteLLMKeyRead(d, client); err == nil {
		t.Fatal("expected error for missing key, got nil")
	}
}

func TestDataSourceKeysRead(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/key/list" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		query := r.URL.Query()
		if query.Get("return_full_object") != "true" {
			t.Errorf("expected return_full_object=true, got %q", query.Get("return_full_object"))
		}
		if query.Get("team_id") != "team-1" {
			t.Errorf("expected team_id=team-1, got %q", query.Get("team_id"))
		}
		if query.Get("page") != "2" || query.Get("size") != "10" {
			t.Errorf("expected page=2 size=10, got page=%q size=%q", query.Get("page"), query.Get("size"))
		}
		if query.Get("include_team_keys") != "true" {
			t.Errorf("expected include_team_keys=true, got %q", query.Get("include_team_keys"))
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{
			"keys": [
				{"token": "tok-1", "key_alias": "a", "team_id": "team-1", "spend": 1.5, "max_budget": 10, "models": ["m1"], "blocked": false},
				{"token": "tok-2", "key_alias": "b", "team_id": "team-1", "spend": 0, "blocked": true}
			],
			"total_count": 2,
			"current_page": 2,
			"total_pages": 1
		}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMKeys().Schema, map[string]interface{}{
		"team_id":           "team-1",
		"page":              2,
		"size":              10,
		"include_team_keys": true,
	})

	if err := dataSourceLiteLLMKeysRead(d, client); err != nil {
		t.Fatalf("read failed: %v", err)
	}

	if d.Id() == "" {
		t.Fatal("expected data source ID to be set")
	}
	if got := d.Get("total_count").(int); got != 2 {
		t.Errorf("expected total_count 2, got %d", got)
	}
	ids := d.Get("ids").([]interface{})
	if len(ids) != 2 || ids[0] != "tok-1" || ids[1] != "tok-2" {
		t.Errorf("unexpected ids: %v", ids)
	}
	keys := d.Get("keys").([]interface{})
	if len(keys) != 2 {
		t.Fatalf("expected 2 keys, got %d", len(keys))
	}
	first := keys[0].(map[string]interface{})
	if first["token_id"] != "tok-1" || first["key_alias"] != "a" || first["max_budget"] != 10.0 {
		t.Errorf("unexpected first key: %v", first)
	}
	second := keys[1].(map[string]interface{})
	if second["blocked"] != true || second["max_budget"] != 0.0 {
		t.Errorf("unexpected second key: %v", second)
	}
}

// Regression for the security review finding: the singular key data source
// must query /key/info by the SHA-256 token hash, never the raw sk- value.
func TestDataSourceKeyQueriesByTokenHash(t *testing.T) {
	var gotQuery string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotQuery = r.URL.Query().Get("key")
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"key": "hash", "info": {"token": "hash", "key_alias": "a"}}`))
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMKey().Schema, map[string]interface{}{"key": "sk-test-123"})
	if err := dataSourceLiteLLMKeyRead(d, NewClient(srv.URL, "master-key", true)); err != nil {
		t.Fatalf("read failed: %v", err)
	}
	if gotQuery != keyBlockTestHash {
		t.Fatalf("query key = %q, want the token hash %q", gotQuery, keyBlockTestHash)
	}
}
