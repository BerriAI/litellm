package litellm

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

// SHA-256 of "sk-test-123", the token hash the proxy stores for that key.
const keyBlockTestHash = "e0dbaa0c6455768bf812d8345ec96a2677d1e3bf17dbb0020b115c80092811e6"

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
		if got := r.URL.Query().Get("key"); got != keyBlockTestHash {
			t.Errorf("expected key query to be the token hash %q, got %q", keyBlockTestHash, got)
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
	if d.Id() != keyBlockTestHash {
		t.Fatalf("expected ID to be the token hash %q, got %q", keyBlockTestHash, d.Id())
	}
	if blockPayload["key"] != keyBlockTestHash {
		t.Fatalf("expected block payload to carry the token hash, got %+v", blockPayload)
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
	d.SetId(keyBlockTestHash)

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
	d.SetId(keyBlockTestHash)

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
	d.SetId(keyBlockTestHash)

	if err := resourceLiteLLMKeyBlockDelete(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if gotPath != "/key/unblock" {
		t.Fatalf("expected path /key/unblock, got %s", gotPath)
	}
	if unblockPayload["key"] != keyBlockTestHash {
		t.Fatalf("expected unblock payload to carry the token hash, got %+v", unblockPayload)
	}
	if d.Id() != "" {
		t.Fatalf("expected ID cleared after delete, got %q", d.Id())
	}
}

// Regression for the security review finding: a raw sk- key must never leave
// the provider in a URL, request body, or resource ID; only its SHA-256 token
// hash may.
func TestKeyBlockNeverSendsRawKey(t *testing.T) {
	var seen []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		seen = append(seen, r.URL.String()+" "+string(body))
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"key":"x","info":{"blocked":true}}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "master-key", true)
	d := newKeyBlockTestResourceData(t, "sk-test-123")
	if err := resourceLiteLLMKeyBlockCreate(d, client); err != nil {
		t.Fatalf("create failed: %v", err)
	}
	if err := resourceLiteLLMKeyBlockRead(d, client); err != nil {
		t.Fatalf("read failed: %v", err)
	}
	if err := resourceLiteLLMKeyBlockDelete(d, client); err != nil {
		t.Fatalf("delete failed: %v", err)
	}

	for _, req := range seen {
		if strings.Contains(req, "sk-test-123") {
			t.Fatalf("raw key leaked to the API: %s", req)
		}
	}
}
