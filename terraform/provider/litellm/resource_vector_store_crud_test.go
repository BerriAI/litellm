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
	body, _ := json.Marshal(map[string]interface{}{"vector_store": resp})

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

// /vector_store/new 400s without a vector_store_id, and the attribute is
// computed, so create has to mint one and key the resource on it rather than on
// the store's name.
func TestVectorStoreCreateSendsGeneratedIDAndUsesItAsResourceID(t *testing.T) {
	var createBody map[string]interface{}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		if r.URL.Path == "/vector_store/new" {
			json.NewDecoder(r.Body).Decode(&createBody)
			id, _ := createBody["vector_store_id"].(string)
			if id == "" {
				w.WriteHeader(http.StatusBadRequest)
				w.Write([]byte(`{"detail":"vector_store_id and custom_llm_provider are required"}`))
				return
			}
			json.NewEncoder(w).Encode(map[string]interface{}{
				"status":       "success",
				"vector_store": VectorStoreResponse{VectorStoreID: id, VectorStoreName: "kb", CustomLLMProvider: "openai"},
			})
			return
		}

		var info VectorStoreInfoRequest
		json.NewDecoder(r.Body).Decode(&info)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"vector_store": VectorStoreResponse{
				VectorStoreID:     info.VectorStoreID,
				VectorStoreName:   "kb",
				CustomLLMProvider: "openai",
				CreatedAt:         "2026-01-01T00:00:00Z",
			},
		})
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resourceLiteLLMVectorStore().Schema, map[string]interface{}{
		"vector_store_name":   "kb",
		"custom_llm_provider": "openai",
	})

	if err := resourceLiteLLMVectorStoreCreate(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("create failed: %v", err)
	}

	sentID, _ := createBody["vector_store_id"].(string)
	if sentID == "" {
		t.Fatal("create payload omitted vector_store_id, which the proxy rejects")
	}
	if d.Id() != sentID {
		t.Errorf("resource id = %q, want the created store id %q", d.Id(), sentID)
	}
	if d.Get("vector_store_id").(string) != sentID {
		t.Errorf("vector_store_id = %q, want %q", d.Get("vector_store_id").(string), sentID)
	}
	if d.Get("created_at").(string) != "2026-01-01T00:00:00Z" {
		t.Errorf("create did not refresh computed fields from the API: %v", d.Get("created_at"))
	}
}
