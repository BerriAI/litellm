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
		if r.URL.Path == endpointModelNew {
			if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
				t.Fatalf("decode request: %v", err)
			}
			_ = json.NewEncoder(w).Encode(payload)
			return
		}
		_ = json.NewEncoder(w).Encode(s04WrappedModelResponse(payload, r.URL.Query().Get("modelId")))
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
	if got := d.Get("cache_read_input_cost_per_million_tokens"); got != 0.6 {
		t.Fatalf("cache read state: want 0.6, got %v", got)
	}
}

func TestCacheReadPricingLifecycle(t *testing.T) {
	for _, tc := range []struct {
		name       string
		configured bool
		price      float64
	}{
		{"omitted", false, 0}, {"zero", true, 0}, {"positive", true, 0.6},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var payload map[string]interface{}
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				if r.Method == http.MethodPost || r.Method == http.MethodPatch {
					if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
						t.Error(err)
					}
					json.NewEncoder(w).Encode(payload)
					return
				}
				json.NewEncoder(w).Encode(s04WrappedModelResponse(payload, r.URL.Query().Get("modelId")))
			}))
			defer srv.Close()
			raw := map[string]interface{}{"model_name": "cache-test", "custom_llm_provider": "openai", "base_model": "cache-test"}
			if tc.configured {
				raw["cache_read_input_cost_per_million_tokens"] = tc.price
			}
			d := schema.TestResourceDataRaw(t, resourceLiteLLMModel().Schema, raw)
			client := NewClient(srv.URL, "test-key", false)
			if err := resourceLiteLLMModelCreate(d, client); err != nil {
				t.Fatal(err)
			}
			info := payload["model_info"].(map[string]interface{})
			got, present := info["cache_read_input_token_cost"]
			if present != tc.configured || (present && got != tc.price/1000000) {
				t.Fatalf("price = %v, present = %t", got, present)
			}
			if !tc.configured {
				return
			}
			d.Set("cache_read_input_cost_per_million_tokens", 0.0)
			if err := resourceLiteLLMModelUpdate(d, client); err != nil {
				t.Fatal(err)
			}
			if got := payload["model_info"].(map[string]interface{})["cache_read_input_token_cost"]; got != 0.0 {
				t.Fatalf("update zero = %v", got)
			}
			payload["model_info"].(map[string]interface{})["cache_read_input_token_cost"] = 0.000002
			if err := resourceLiteLLMModelRead(d, client); err != nil {
				t.Fatal(err)
			}
			if d.Get("cache_read_input_cost_per_million_tokens") != 2.0 {
				t.Fatal("refresh did not read changed pricing")
			}
			imported := schema.TestResourceDataRaw(t, resourceLiteLLMModel().Schema, nil)
			imported.SetId(d.Id())
			if err := resourceLiteLLMModelRead(imported, client); err != nil {
				t.Fatal(err)
			}
			if imported.Get("cache_read_input_cost_per_million_tokens") != 2.0 {
				t.Fatal("import did not recover pricing")
			}
		})
	}
}
