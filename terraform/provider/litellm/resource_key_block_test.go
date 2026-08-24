package litellm

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func newKeyBlockTestResourceData(t *testing.T, key string) *schema.ResourceData {
	t.Helper()
	return schema.TestResourceDataRaw(t, resourceLiteLLMKeyBlock().Schema, map[string]interface{}{
		"key": key,
	})
}

func TestResourceLiteLLMKeyBlockCreate(t *testing.T) {
	var blockPayload map[string]interface{}
	mux := http.NewServeMux()
	mux.HandleFunc("/key/block", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if err := json.NewDecoder(r.Body).Decode(&blockPayload); err != nil {
			t.Fatalf("failed to decode block payload: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"blocked":true}`))
	})
	mux.HandleFunc("/key/info", func(w http.ResponseWriter, r *http.Request) {
		if got := r.URL.Query().Get("key"); got != "sk-test-123" {
			t.Errorf("expected key query 'sk-test-123', got %q", got)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"key":"sk-test-123","info":{"blocked":true}}`))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newKeyBlockTestResourceData(t, "sk-test-123")

	if err := resourceLiteLLMKeyBlockCreate(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if d.Id() != "sk-test-123" {
		t.Fatalf("expected ID 'sk-test-123', got %q", d.Id())
	}
	if blockPayload["key"] != "sk-test-123" {
		t.Fatalf("expected block payload key 'sk-test-123', got %+v", blockPayload)
	}
	if !d.Get("blocked").(bool) {
		t.Fatal("expected blocked=true in state")
	}
}

func TestResourceLiteLLMKeyBlockRead_UnblockedClearsID(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"key":"sk-test-123","info":{"blocked":false}}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newKeyBlockTestResourceData(t, "sk-test-123")
	d.SetId("sk-test-123")

	if err := resourceLiteLLMKeyBlockRead(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if d.Id() != "" {
		t.Fatalf("expected ID cleared for unblocked key, got %q", d.Id())
	}
}

func TestResourceLiteLLMKeyBlockRead_404ClearsID(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newKeyBlockTestResourceData(t, "sk-test-123")
	d.SetId("sk-test-123")

	if err := resourceLiteLLMKeyBlockRead(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if d.Id() != "" {
		t.Fatalf("expected ID cleared on 404, got %q", d.Id())
	}
}

func TestResourceLiteLLMKeyBlockDelete(t *testing.T) {
	var gotPath string
	var unblockPayload map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		json.NewDecoder(r.Body).Decode(&unblockPayload)
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"blocked":false}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newKeyBlockTestResourceData(t, "sk-test-123")
	d.SetId("sk-test-123")

	if err := resourceLiteLLMKeyBlockDelete(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if gotPath != "/key/unblock" {
		t.Fatalf("expected path /key/unblock, got %s", gotPath)
	}
	if unblockPayload["key"] != "sk-test-123" {
		t.Fatalf("expected unblock payload key 'sk-test-123', got %+v", unblockPayload)
	}
	if d.Id() != "" {
		t.Fatalf("expected ID cleared after delete, got %q", d.Id())
	}
}
