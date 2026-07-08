package litellm

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func TestVectorStoreReadDoesNotPersistServerLitellmParams(t *testing.T) {
	resp := VectorStoreResponse{
		VectorStoreID:     "vs-123",
		VectorStoreName:   "kb",
		CustomLLMProvider: "openai",
		LiteLLMParams: map[string]interface{}{
			"api_key":  "sk-from-server",
			"api_base": "https://upstream.example.com",
		},
	}
	body, _ := json.Marshal(resp)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write(body)
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, resourceLiteLLMVectorStore().Schema, map[string]interface{}{
		"vector_store_name":   "kb",
		"custom_llm_provider": "openai",
		"litellm_params": map[string]interface{}{
			"vector_store_id": "vs-123",
		},
	})
	d.SetId("vs-123")

	if err := resourceLiteLLMVectorStoreRead(d, client); err != nil {
		t.Fatalf("read failed: %v", err)
	}

	got := d.Get("litellm_params").(map[string]interface{})
	if _, leaked := got["api_key"]; leaked {
		t.Fatalf("server-returned api_key persisted into state: %v", got)
	}
	if got["vector_store_id"] != "vs-123" {
		t.Fatalf("config litellm_params not preserved: %v", got)
	}
	if d.Get("vector_store_name").(string) != "kb" {
		t.Fatalf("read did not populate non-sensitive fields")
	}
}
