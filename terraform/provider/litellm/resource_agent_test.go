package litellm

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

const testAgentCardJSON = `{"name": "Hello Agent", "url": "http://agent.local:9999/", "version": "1.0.0"}`

func newAgentTestResourceData(t *testing.T) *schema.ResourceData {
	t.Helper()
	return schema.TestResourceDataRaw(t, resourceLiteLLMAgent().Schema, map[string]interface{}{
		"agent_name":        "my-agent",
		"agent_card_params": testAgentCardJSON,
		"litellm_params":    `{"model": "gpt-5.2", "api_key": "sk-secret"}`,
		"extra_headers":     []interface{}{"x-request-id"},
		"tpm_limit":         1000,
	})
}

func agentReadResponseBody() []byte {
	body, _ := json.Marshal(map[string]interface{}{
		"agent_id":   "agent-123",
		"agent_name": "my-agent",
		"agent_card_params": map[string]interface{}{
			"name":                "Hello Agent",
			"url":                 "http://agent.local:9999/",
			"version":             "1.0.0",
			"supportedInterfaces": []string{"http://proxy/a2a/agent-123"},
		},
		"litellm_params": map[string]interface{}{"model": "gpt-5.2", "api_key": "sk-1****"},
		"extra_headers":  []string{"x-request-id"},
		"tpm_limit":      1000,
		"spend":          1.5,
		"created_at":     "2026-01-01T00:00:00",
		"updated_at":     "2026-01-02T00:00:00",
		"created_by":     "admin",
		"updated_by":     "admin",
	})
	return body
}

func TestResourceLiteLLMAgentCreate(t *testing.T) {
	var createPayload map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/v1/agents":
			if err := json.NewDecoder(r.Body).Decode(&createPayload); err != nil {
				t.Errorf("failed to decode create payload: %v", err)
			}
			w.Write([]byte(`{"agent_id": "agent-123", "agent_name": "my-agent", "agent_card_params": {}}`))
		case r.Method == http.MethodGet && r.URL.Path == "/v1/agents/agent-123":
			w.Write(agentReadResponseBody())
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newAgentTestResourceData(t)

	if err := resourceLiteLLMAgentCreate(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if d.Id() != "agent-123" {
		t.Fatalf("expected ID 'agent-123', got %q", d.Id())
	}

	if createPayload["agent_name"] != "my-agent" {
		t.Errorf("expected agent_name 'my-agent' in payload, got %v", createPayload["agent_name"])
	}
	card, ok := createPayload["agent_card_params"].(map[string]interface{})
	if !ok || card["url"] != "http://agent.local:9999/" {
		t.Errorf("expected agent_card_params sent as JSON object with url, got %v", createPayload["agent_card_params"])
	}
	params, ok := createPayload["litellm_params"].(map[string]interface{})
	if !ok || params["api_key"] != "sk-secret" {
		t.Errorf("expected litellm_params sent as JSON object, got %v", createPayload["litellm_params"])
	}
	if createPayload["tpm_limit"] != float64(1000) {
		t.Errorf("expected tpm_limit 1000 in payload, got %v", createPayload["tpm_limit"])
	}

	if d.Get("created_at").(string) != "2026-01-01T00:00:00" {
		t.Errorf("expected created_at from read-back, got %q", d.Get("created_at").(string))
	}
	if got := d.Get("agent_card_params").(string); got != testAgentCardJSON {
		t.Errorf("expected configured agent_card_params to stay authoritative, got %q", got)
	}
	if got := d.Get("litellm_params").(string); got != `{"model": "gpt-5.2", "api_key": "sk-secret"}` {
		t.Errorf("expected litellm_params to keep configured value, got %q", got)
	}
}

func TestResourceLiteLLMAgentCreateInvalidCardJSON(t *testing.T) {
	d := schema.TestResourceDataRaw(t, resourceLiteLLMAgent().Schema, map[string]interface{}{
		"agent_name":        "my-agent",
		"agent_card_params": "not-json",
	})
	client := NewClient("http://unused.invalid", "test-key", true)

	if err := resourceLiteLLMAgentCreate(d, client); err == nil {
		t.Fatal("expected error for invalid agent_card_params JSON, got nil")
	}
}

func TestResourceLiteLLMAgentReadPopulatesStateOnImport(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/v1/agents/agent-123" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write(agentReadResponseBody())
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, resourceLiteLLMAgent().Schema, map[string]interface{}{})
	d.SetId("agent-123")

	if err := resourceLiteLLMAgentRead(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if d.Get("agent_name").(string) != "my-agent" {
		t.Errorf("expected agent_name 'my-agent', got %q", d.Get("agent_name").(string))
	}
	var card map[string]interface{}
	if err := json.Unmarshal([]byte(d.Get("agent_card_params").(string)), &card); err != nil {
		t.Fatalf("agent_card_params not populated as JSON on import: %v", err)
	}
	if card["name"] != "Hello Agent" {
		t.Errorf("expected card name 'Hello Agent', got %v", card["name"])
	}
	if d.Get("tpm_limit").(int) != 1000 {
		t.Errorf("expected tpm_limit 1000, got %d", d.Get("tpm_limit").(int))
	}
	headers := d.Get("extra_headers").([]interface{})
	if len(headers) != 1 || headers[0] != "x-request-id" {
		t.Errorf("expected extra_headers ['x-request-id'], got %v", headers)
	}
	if d.Get("litellm_params").(string) != "" {
		t.Errorf("expected litellm_params to never be read back, got %q", d.Get("litellm_params").(string))
	}
}

func TestResourceLiteLLMAgentRead404ClearsID(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newAgentTestResourceData(t)
	d.SetId("agent-123")

	if err := resourceLiteLLMAgentRead(d, client); err != nil {
		t.Fatalf("expected nil error on 404, got: %v", err)
	}
	if d.Id() != "" {
		t.Fatalf("expected ID cleared on 404, got %q", d.Id())
	}
}

func TestResourceLiteLLMAgentUpdate(t *testing.T) {
	var updateMethod, updatePath string
	var updatePayload map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method == http.MethodGet {
			w.Write(agentReadResponseBody())
			return
		}
		updateMethod = r.Method
		updatePath = r.URL.Path
		if err := json.NewDecoder(r.Body).Decode(&updatePayload); err != nil {
			t.Errorf("failed to decode update payload: %v", err)
		}
		w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newAgentTestResourceData(t)
	d.SetId("agent-123")

	if err := resourceLiteLLMAgentUpdate(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if updateMethod != http.MethodPatch {
		t.Errorf("expected PATCH, got %s", updateMethod)
	}
	if updatePath != "/v1/agents/agent-123" {
		t.Errorf("expected path '/v1/agents/agent-123', got %q", updatePath)
	}
	if updatePayload["agent_name"] != "my-agent" {
		t.Errorf("expected agent_name in update payload, got %v", updatePayload["agent_name"])
	}
	if updatePayload["tpm_limit"] != float64(1000) {
		t.Errorf("expected tpm_limit 1000 in update payload, got %v", updatePayload["tpm_limit"])
	}
}

func TestResourceLiteLLMAgentDelete(t *testing.T) {
	var deleteMethod, deletePath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		deleteMethod = r.Method
		deletePath = r.URL.Path
		w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newAgentTestResourceData(t)
	d.SetId("agent-123")

	if err := resourceLiteLLMAgentDelete(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if deleteMethod != http.MethodDelete {
		t.Errorf("expected DELETE, got %s", deleteMethod)
	}
	if deletePath != "/v1/agents/agent-123" {
		t.Errorf("expected path '/v1/agents/agent-123', got %q", deletePath)
	}
	if d.Id() != "" {
		t.Fatalf("expected ID cleared after delete, got %q", d.Id())
	}
}
