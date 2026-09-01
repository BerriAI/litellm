package litellm

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func TestResourceLiteLLMModelCreateSendsCacheReadPricing(t *testing.T) {
	var payload map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodPost && r.URL.Path == endpointModelNew:
			if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
				t.Fatalf("decode request: %v", err)
			}
			json.NewEncoder(w).Encode(payload)
		case r.Method == http.MethodGet && r.URL.Path == endpointModelInfo:
			json.NewEncoder(w).Encode(payload)
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resourceLiteLLMModel().Schema, map[string]interface{}{
		"model_name":          "cached-model",
		"custom_llm_provider": "openai",
		"base_model":          "cached-model",
		"cache_read_input_cost_per_million_tokens": 0.6,
	})
	if err := resourceLiteLLMModelCreate(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("create model: %v", err)
	}

	modelInfo := payload["model_info"].(map[string]interface{})
	if got := modelInfo["cache_read_input_token_cost"]; got != 0.0000006 {
		t.Fatalf("cache read cost: want 0.0000006, got %v", got)
	}
}
