package litellm

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func TestDataSourceModelReadSingleObject(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/v1/model/info" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		if got := r.URL.Query().Get("litellm_model_id"); got != "model-abc" {
			t.Errorf("expected litellm_model_id 'model-abc', got %q", got)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{
			"data": {
				"model_name": "gpt-4o-alias",
				"litellm_params": {
					"model": "openai/gpt-4o",
					"custom_llm_provider": "openai",
					"api_base": "https://api.openai.com/v1",
					"api_version": "2024-06-01",
					"api_key": "sk-should-never-surface",
					"tpm": 100000,
					"rpm": 500
				},
				"model_info": {
					"id": "model-abc",
					"db_model": true,
					"base_model": "gpt-4o",
					"tier": "paid",
					"mode": "chat",
					"team_id": "team-1"
				}
			}
		}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMModel().Schema, map[string]interface{}{
		"model_id": "model-abc",
	})

	if err := dataSourceLiteLLMModelRead(d, client); err != nil {
		t.Fatalf("read failed: %v", err)
	}

	if d.Id() != "model-abc" {
		t.Fatalf("expected ID 'model-abc', got %q", d.Id())
	}
	checks := map[string]interface{}{
		"model_name":          "gpt-4o-alias",
		"model":               "openai/gpt-4o",
		"custom_llm_provider": "openai",
		"model_api_base":      "https://api.openai.com/v1",
		"api_version":         "2024-06-01",
		"tpm":                 100000,
		"rpm":                 500,
		"base_model":          "gpt-4o",
		"tier":                "paid",
		"mode":                "chat",
		"team_id":             "team-1",
		"db_model":            true,
	}
	for attr, want := range checks {
		if got := d.Get(attr); got != want {
			t.Errorf("attr %s: expected %v, got %v", attr, want, got)
		}
	}
}

func TestDataSourceModelReadListShape(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{
			"data": [{
				"model_name": "claude-alias",
				"litellm_params": {"model": "anthropic/claude-opus-4", "custom_llm_provider": "anthropic"},
				"model_info": {"id": "model-xyz", "mode": "chat"}
			}]
		}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMModel().Schema, map[string]interface{}{
		"model_id": "model-xyz",
	})

	if err := dataSourceLiteLLMModelRead(d, client); err != nil {
		t.Fatalf("read failed: %v", err)
	}
	if d.Id() != "model-xyz" {
		t.Fatalf("expected ID 'model-xyz', got %q", d.Id())
	}
	if got := d.Get("model").(string); got != "anthropic/claude-opus-4" {
		t.Errorf("expected model 'anthropic/claude-opus-4', got %q", got)
	}
}

func TestDataSourceModelsRead(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/v1/model/info" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		if got := r.URL.Query().Get("teamId"); got != "team-1" {
			t.Errorf("expected teamId 'team-1', got %q", got)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{
			"data": [
				{"model_name": "a", "litellm_params": {"model": "openai/a", "custom_llm_provider": "openai"}, "model_info": {"id": "id-1", "db_model": true}},
				{"model_name": "b", "litellm_params": {"model": "anthropic/b", "custom_llm_provider": "anthropic"}, "model_info": {"id": "id-2"}}
			]
		}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMModels().Schema, map[string]interface{}{
		"team_id": "team-1",
	})

	if err := dataSourceLiteLLMModelsRead(d, client); err != nil {
		t.Fatalf("read failed: %v", err)
	}

	if d.Id() != "team-1" {
		t.Fatalf("expected ID 'team-1', got %q", d.Id())
	}
	ids := d.Get("ids").([]interface{})
	if len(ids) != 2 || ids[0] != "id-1" || ids[1] != "id-2" {
		t.Errorf("unexpected ids: %v", ids)
	}
	models := d.Get("models").([]interface{})
	if len(models) != 2 {
		t.Fatalf("expected 2 models, got %d", len(models))
	}
	first := models[0].(map[string]interface{})
	if first["model_name"] != "a" || first["custom_llm_provider"] != "openai" || first["db_model"] != true {
		t.Errorf("unexpected first model: %v", first)
	}
}
