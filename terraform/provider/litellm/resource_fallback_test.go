package litellm

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func newFallbackTestResourceData(t *testing.T, model string, fallbackModels []interface{}, fallbackType string) *schema.ResourceData {
	t.Helper()
	d := schema.TestResourceDataRaw(t, resourceLiteLLMFallback().Schema, map[string]interface{}{
		"model":           model,
		"fallback_models": fallbackModels,
		"fallback_type":   fallbackType,
	})
	return d
}

func fallbackGetHandler(t *testing.T, wantPath string, resp FallbackGetResponse) http.HandlerFunc {
	t.Helper()
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			t.Errorf("expected GET, got %s", r.Method)
		}
		if r.URL.Path != wantPath {
			t.Errorf("expected path %s, got %s", wantPath, r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}
}

func TestResourceLiteLLMFallbackCreate(t *testing.T) {
	var createPayload map[string]interface{}
	mux := http.NewServeMux()
	mux.HandleFunc("/fallback", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if err := json.NewDecoder(r.Body).Decode(&createPayload); err != nil {
			t.Fatalf("failed to decode create payload: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model":"gpt-4","fallback_models":["claude-3","gpt-3.5-turbo"],"fallback_type":"general","message":"ok"}`))
	})
	mux.Handle("/fallback/gpt-4", fallbackGetHandler(t, "/fallback/gpt-4", FallbackGetResponse{
		Model:          "gpt-4",
		FallbackModels: []string{"claude-3", "gpt-3.5-turbo"},
		FallbackType:   "general",
	}))
	srv := httptest.NewServer(mux)
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newFallbackTestResourceData(t, "gpt-4", []interface{}{"claude-3", "gpt-3.5-turbo"}, "general")

	if err := resourceLiteLLMFallbackCreate(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if d.Id() != "gpt-4" {
		t.Fatalf("expected ID 'gpt-4', got %q", d.Id())
	}
	want := map[string]interface{}{
		"model":           "gpt-4",
		"fallback_models": []interface{}{"claude-3", "gpt-3.5-turbo"},
		"fallback_type":   "general",
	}
	if !reflect.DeepEqual(createPayload, want) {
		t.Fatalf("unexpected create payload: %+v, want %+v", createPayload, want)
	}
}

func TestResourceLiteLLMFallbackRead_MapsFields(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/fallback/gpt-4" {
			t.Errorf("expected path /fallback/gpt-4, got %s", r.URL.Path)
		}
		if got := r.URL.Query().Get("fallback_type"); got != "context_window" {
			t.Errorf("expected fallback_type query 'context_window', got %q", got)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model":"gpt-4","fallback_models":["claude-3"],"fallback_type":"context_window"}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newFallbackTestResourceData(t, "gpt-4", []interface{}{"stale-model"}, "context_window")
	d.SetId("gpt-4")

	if err := resourceLiteLLMFallbackRead(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	got := d.Get("fallback_models").([]interface{})
	if !reflect.DeepEqual(got, []interface{}{"claude-3"}) {
		t.Fatalf("expected fallback_models [claude-3], got %+v", got)
	}
	if d.Get("fallback_type").(string) != "context_window" {
		t.Fatalf("expected fallback_type 'context_window', got %q", d.Get("fallback_type"))
	}
	if d.Get("model").(string) != "gpt-4" {
		t.Fatalf("expected model 'gpt-4', got %q", d.Get("model"))
	}
}

func TestResourceLiteLLMFallbackRead_404ClearsID(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newFallbackTestResourceData(t, "gpt-4", []interface{}{"claude-3"}, "general")
	d.SetId("gpt-4")

	if err := resourceLiteLLMFallbackRead(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if d.Id() != "" {
		t.Fatalf("expected ID cleared on 404, got %q", d.Id())
	}
}

func TestResourceLiteLLMFallbackUpdate_SendsChangedModels(t *testing.T) {
	var updatePayload map[string]interface{}
	mux := http.NewServeMux()
	mux.HandleFunc("/fallback", func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&updatePayload)
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model":"gpt-4","fallback_models":["new-model"],"fallback_type":"general","message":"ok"}`))
	})
	mux.Handle("/fallback/gpt-4", fallbackGetHandler(t, "/fallback/gpt-4", FallbackGetResponse{
		Model:          "gpt-4",
		FallbackModels: []string{"new-model"},
		FallbackType:   "general",
	}))
	srv := httptest.NewServer(mux)
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newFallbackTestResourceData(t, "gpt-4", []interface{}{"new-model"}, "general")
	d.SetId("gpt-4")

	if err := resourceLiteLLMFallbackUpdate(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if !reflect.DeepEqual(updatePayload["fallback_models"], []interface{}{"new-model"}) {
		t.Fatalf("expected updated fallback_models [new-model], got %+v", updatePayload["fallback_models"])
	}
}

func TestResourceLiteLLMFallbackDelete(t *testing.T) {
	var gotMethod, gotPath, gotType string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod = r.Method
		gotPath = r.URL.Path
		gotType = r.URL.Query().Get("fallback_type")
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model":"gpt-4","fallback_type":"general","message":"deleted"}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newFallbackTestResourceData(t, "gpt-4", []interface{}{"claude-3"}, "general")
	d.SetId("gpt-4")

	if err := resourceLiteLLMFallbackDelete(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if gotMethod != http.MethodDelete || gotPath != "/fallback/gpt-4" || gotType != "general" {
		t.Fatalf("expected DELETE /fallback/gpt-4?fallback_type=general, got %s %s?fallback_type=%s",
			gotMethod, gotPath, gotType)
	}
	if d.Id() != "" {
		t.Fatalf("expected ID cleared after delete, got %q", d.Id())
	}
}
