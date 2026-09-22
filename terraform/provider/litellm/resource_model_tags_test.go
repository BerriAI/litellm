package litellm

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func newModelTestServer(t *testing.T, captured *map[string]interface{}, infoBody string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case endpointModelNew, endpointModelUpdate:
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, captured)
			w.Write([]byte(infoBody))
		case endpointModelInfo:
			w.Write([]byte(infoBody))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
}

func newModelResourceData(t *testing.T, raw map[string]interface{}) *schema.ResourceData {
	t.Helper()
	return schema.TestResourceDataRaw(t, resourceLiteLLMModel().Schema, raw)
}

func TestModelCreateSendsTags(t *testing.T) {
	var captured map[string]interface{}
	infoBody := `{
		"model_name": "gpt-4.1",
		"litellm_params": {
			"model": "azure/gpt-4.1",
			"custom_llm_provider": "azure",
			"tags": ["advisor"]
		},
		"model_info": {
			"id": "model-1",
			"db_model": true,
			"base_model": "gpt-4.1",
			"tier": "free",
			"mode": "chat"
		}
	}`
	srv := newModelTestServer(t, &captured, infoBody)
	defer srv.Close()

	d := newModelResourceData(t, map[string]interface{}{
		"model_name":          "gpt-4.1",
		"custom_llm_provider": "azure",
		"base_model":          "gpt-4.1",
		"tags":                []interface{}{"advisor"},
	})

	if err := resourceLiteLLMModelCreate(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("create failed: %v", err)
	}

	litellmParams, ok := captured["litellm_params"].(map[string]interface{})
	if !ok {
		t.Fatalf("payload litellm_params = %v, want a map", captured["litellm_params"])
	}
	wantTags := []interface{}{"advisor"}
	if got := litellmParams["tags"]; !reflect.DeepEqual(got, wantTags) {
		t.Fatalf("payload litellm_params.tags = %v, want %v", got, wantTags)
	}
}

func TestModelReadPopulatesTagsFromResponse(t *testing.T) {
	var captured map[string]interface{}
	infoBody := `{
		"model_name": "gpt-4.1",
		"litellm_params": {
			"model": "azure/gpt-4.1",
			"custom_llm_provider": "azure",
			"tags": ["advisor", "callbot"]
		},
		"model_info": {
			"id": "model-1",
			"db_model": true,
			"base_model": "gpt-4.1",
			"tier": "free",
			"mode": "chat"
		}
	}`
	srv := newModelTestServer(t, &captured, infoBody)
	defer srv.Close()

	d := newModelResourceData(t, map[string]interface{}{
		"model_name":          "gpt-4.1",
		"custom_llm_provider": "azure",
		"base_model":          "gpt-4.1",
	})
	d.SetId("model-1")

	if err := resourceLiteLLMModelRead(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("read failed: %v", err)
	}

	wantTags := []interface{}{"advisor", "callbot"}
	if got := d.Get("tags"); !reflect.DeepEqual(got, wantTags) {
		t.Fatalf("state tags = %v, want %v", got, wantTags)
	}
}

func TestModelUpdateOmitsTagsWhenUnset(t *testing.T) {
	var captured map[string]interface{}
	infoBody := `{
		"model_name": "gpt-4.1",
		"litellm_params": {
			"model": "azure/gpt-4.1",
			"custom_llm_provider": "azure"
		},
		"model_info": {
			"id": "model-1",
			"db_model": true,
			"base_model": "gpt-4.1",
			"tier": "free",
			"mode": "chat"
		}
	}`
	srv := newModelTestServer(t, &captured, infoBody)
	defer srv.Close()

	d := newModelResourceData(t, map[string]interface{}{
		"model_name":          "gpt-4.1",
		"custom_llm_provider": "azure",
		"base_model":          "gpt-4.1",
	})
	d.SetId("model-1")

	if err := resourceLiteLLMModelUpdate(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("update failed: %v", err)
	}

	litellmParams, ok := captured["litellm_params"].(map[string]interface{})
	if !ok {
		t.Fatalf("payload litellm_params = %v, want a map", captured["litellm_params"])
	}
	if _, ok := litellmParams["tags"]; ok {
		t.Fatalf("payload litellm_params.tags = %v, want omitted when unset", litellmParams["tags"])
	}
}
