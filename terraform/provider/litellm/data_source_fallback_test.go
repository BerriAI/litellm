package litellm

import (
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func TestDataSourceLiteLLMFallbackRead(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/fallback/gpt-4" {
			t.Errorf("expected path /fallback/gpt-4, got %s", r.URL.Path)
		}
		if got := r.URL.Query().Get("fallback_type"); got != "general" {
			t.Errorf("expected fallback_type query 'general', got %q", got)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model":"gpt-4","fallback_models":["claude-3","gpt-3.5-turbo"],"fallback_type":"general"}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMFallback().Schema, map[string]interface{}{
		"model":         "gpt-4",
		"fallback_type": "general",
	})

	if err := dataSourceLiteLLMFallbackRead(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if d.Id() != "gpt-4" {
		t.Fatalf("expected ID 'gpt-4', got %q", d.Id())
	}
	got := d.Get("fallback_models").([]interface{})
	if !reflect.DeepEqual(got, []interface{}{"claude-3", "gpt-3.5-turbo"}) {
		t.Fatalf("expected fallback_models [claude-3 gpt-3.5-turbo], got %+v", got)
	}
}

func TestDataSourceLiteLLMFallbackRead_NotFoundErrors(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMFallback().Schema, map[string]interface{}{
		"model":         "missing-model",
		"fallback_type": "general",
	})

	err := dataSourceLiteLLMFallbackRead(d, client)
	if err == nil {
		t.Fatal("expected error for missing fallback, got nil")
	}
	if !strings.Contains(err.Error(), "missing-model") {
		t.Fatalf("expected error to name the model, got: %v", err)
	}
}
