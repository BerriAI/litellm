package litellm

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func TestDataSourcePromptRead(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" || r.URL.Path != "/prompts/p1/info" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(promptInfoJSON("p1")))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMPrompt().Schema, map[string]interface{}{
		"prompt_id": "p1",
	})

	if err := dataSourceLiteLLMPromptRead(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if d.Id() != "p1" {
		t.Fatalf("expected ID 'p1', got %q", d.Id())
	}
	if got := d.Get("prompt_integration").(string); got != "langfuse" {
		t.Errorf("expected prompt_integration 'langfuse', got %q", got)
	}
	if got := d.Get("prompt_type").(string); got != "db" {
		t.Errorf("expected prompt_type 'db', got %q", got)
	}
	if got := d.Get("version").(int); got != 3 {
		t.Errorf("expected version 3, got %d", got)
	}
	envs := d.Get("environments").([]interface{})
	if len(envs) != 1 || envs[0] != "development" {
		t.Errorf("unexpected environments: %v", envs)
	}
}

func TestDataSourcePromptsRead_WithEnvironmentFilter(t *testing.T) {
	var gotQuery string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" || r.URL.Path != "/prompts/list" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		gotQuery = r.URL.RawQuery
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"prompts": [
			{
				"prompt_id": "p1",
				"litellm_params": {"prompt_integration": "langfuse"},
				"prompt_info": {"prompt_type": "db"},
				"version": 2,
				"environment": "production"
			}
		]}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMPrompts().Schema, map[string]interface{}{
		"environment": "production",
	})

	if err := dataSourceLiteLLMPromptsRead(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if gotQuery != "environment=production" {
		t.Fatalf("expected environment filter in query, got %q", gotQuery)
	}

	prompts := d.Get("prompts").([]interface{})
	if len(prompts) != 1 {
		t.Fatalf("expected 1 prompt, got %d", len(prompts))
	}
	first := prompts[0].(map[string]interface{})
	if first["prompt_id"] != "p1" || first["prompt_integration"] != "langfuse" ||
		first["prompt_type"] != "db" || first["version"] != 2 || first["environment"] != "production" {
		t.Errorf("unexpected prompt item: %v", first)
	}
	ids := d.Get("ids").([]interface{})
	if len(ids) != 1 || ids[0] != "p1" {
		t.Errorf("unexpected ids: %v", ids)
	}
}
