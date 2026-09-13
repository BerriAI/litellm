package litellm

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func newPromptTestData(t *testing.T, raw map[string]interface{}) *schema.ResourceData {
	t.Helper()
	return schema.TestResourceDataRaw(t, resourceLiteLLMPrompt().Schema, raw)
}

func promptInfoJSON(promptID string) string {
	body, _ := json.Marshal(map[string]interface{}{
		"prompt_spec": map[string]interface{}{
			"prompt_id": promptID,
			"litellm_params": map[string]interface{}{
				"prompt_integration":             "langfuse",
				"api_base":                       "https://langfuse.example.com",
				"ignore_prompt_manager_model":    true,
				"provider_specific_query_params": map[string]interface{}{"label": "prod"},
			},
			"prompt_info": map[string]interface{}{"prompt_type": "db"},
			"version":     3,
			"environment": "development",
			"created_at":  "2026-01-01T00:00:00Z",
			"updated_at":  "2026-01-02T00:00:00Z",
		},
		"environments": []string{"development"},
	})
	return string(body)
}

func TestPromptCreate_SendsPayloadAndSetsID(t *testing.T) {
	var createPayload map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == "POST" && r.URL.Path == "/prompts":
			if err := json.NewDecoder(r.Body).Decode(&createPayload); err != nil {
				t.Errorf("failed to decode create payload: %v", err)
			}
			w.Write([]byte(`{"prompt_id": "p1"}`))
		case r.Method == "GET" && r.URL.Path == "/prompts/p1/info":
			w.Write([]byte(promptInfoJSON("p1")))
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newPromptTestData(t, map[string]interface{}{
		"prompt_id":          "p1",
		"prompt_integration": "langfuse",
		"api_key":            "sk-langfuse",
		"litellm_params":     `{"prompt_id": "external-prompt", "prompt_directory": "/prompts"}`,
		"prompt_type":        "db",
	})

	if err := resourceLiteLLMPromptCreate(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if d.Id() != "p1" {
		t.Fatalf("expected ID 'p1', got %q", d.Id())
	}

	if createPayload["prompt_id"] != "p1" {
		t.Errorf("expected prompt_id 'p1', got %v", createPayload["prompt_id"])
	}
	params, ok := createPayload["litellm_params"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected litellm_params object, got: %v", createPayload["litellm_params"])
	}
	if params["prompt_integration"] != "langfuse" || params["api_key"] != "sk-langfuse" {
		t.Errorf("unexpected litellm_params: %v", params)
	}
	if params["prompt_id"] != "external-prompt" || params["prompt_directory"] != "/prompts" {
		t.Errorf("expected merged extra litellm_params, got: %v", params)
	}
	info, ok := createPayload["prompt_info"].(map[string]interface{})
	if !ok || info["prompt_type"] != "db" {
		t.Errorf("expected prompt_info with prompt_type 'db', got: %v", createPayload["prompt_info"])
	}
}

func TestPromptRead_MapsFieldsAndKeepsAPIKey(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" || r.URL.Path != "/prompts/p1/info" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(promptInfoJSON("p1")))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newPromptTestData(t, map[string]interface{}{
		"prompt_id":          "p1",
		"prompt_integration": "old-integration",
		"api_key":            "sk-configured",
	})
	d.SetId("p1")

	if err := resourceLiteLLMPromptRead(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if got := d.Get("prompt_integration").(string); got != "langfuse" {
		t.Errorf("expected prompt_integration 'langfuse', got %q", got)
	}
	if got := d.Get("api_base").(string); got != "https://langfuse.example.com" {
		t.Errorf("expected api_base from API, got %q", got)
	}
	if got := d.Get("ignore_prompt_manager_model").(bool); !got {
		t.Error("expected ignore_prompt_manager_model true from API")
	}
	if got := d.Get("provider_specific_query_params").(string); got != `{"label":"prod"}` {
		t.Errorf("expected provider_specific_query_params JSON, got %q", got)
	}
	if got := d.Get("prompt_type").(string); got != "db" {
		t.Errorf("expected prompt_type 'db', got %q", got)
	}
	if got := d.Get("api_key").(string); got != "sk-configured" {
		t.Errorf("expected configured api_key to stay authoritative, got %q", got)
	}
}

func TestPromptRead_NotFound400ClearsID(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`{"detail": "Prompt p-gone not found"}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newPromptTestData(t, map[string]interface{}{
		"prompt_id":          "p-gone",
		"prompt_integration": "langfuse",
	})
	d.SetId("p-gone")

	if err := resourceLiteLLMPromptRead(d, client); err != nil {
		t.Fatalf("expected nil error on not-found 400, got: %v", err)
	}
	if d.Id() != "" {
		t.Fatalf("expected ID to be cleared, got %q", d.Id())
	}
}

func TestPromptRead_Other400ReturnsError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`{"detail": "invalid environment"}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newPromptTestData(t, map[string]interface{}{
		"prompt_id":          "p1",
		"prompt_integration": "langfuse",
	})
	d.SetId("p1")

	if err := resourceLiteLLMPromptRead(d, client); err == nil {
		t.Fatal("expected error for non-not-found 400, got nil")
	}
	if d.Id() != "p1" {
		t.Fatalf("expected ID to be kept, got %q", d.Id())
	}
}

func TestPromptUpdate_SendsPUTToPromptEndpoint(t *testing.T) {
	var updateMethod, updatePath string
	var updatePayload map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method == "PUT" {
			updateMethod, updatePath = r.Method, r.URL.Path
			json.NewDecoder(r.Body).Decode(&updatePayload)
			w.Write([]byte(`{"prompt_id": "p1"}`))
			return
		}
		w.Write([]byte(promptInfoJSON("p1")))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newPromptTestData(t, map[string]interface{}{
		"prompt_id":          "p1",
		"prompt_integration": "langfuse",
		"api_base":           "https://new-base.example.com",
	})
	d.SetId("p1")

	if err := resourceLiteLLMPromptUpdate(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if updateMethod != "PUT" || updatePath != "/prompts/p1" {
		t.Fatalf("expected PUT /prompts/p1, got %s %s", updateMethod, updatePath)
	}
	params := updatePayload["litellm_params"].(map[string]interface{})
	if params["api_base"] != "https://new-base.example.com" {
		t.Errorf("expected updated api_base in payload, got %v", params["api_base"])
	}
}

func TestPromptDelete_CallsDeleteEndpoint(t *testing.T) {
	var deleteMethod, deletePath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		deleteMethod, deletePath = r.Method, r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"message": "deleted"}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newPromptTestData(t, map[string]interface{}{
		"prompt_id":          "p1",
		"prompt_integration": "langfuse",
	})
	d.SetId("p1")

	if err := resourceLiteLLMPromptDelete(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if deleteMethod != "DELETE" || deletePath != "/prompts/p1" {
		t.Fatalf("expected DELETE /prompts/p1, got %s %s", deleteMethod, deletePath)
	}
	if d.Id() != "" {
		t.Fatalf("expected ID to be cleared after delete, got %q", d.Id())
	}
}
