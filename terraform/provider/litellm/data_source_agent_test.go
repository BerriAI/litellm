package litellm

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func TestDataSourceLiteLLMAgentRead(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/v1/agents/agent-123" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write(agentReadResponseBody())
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMAgent().Schema, map[string]interface{}{
		"agent_id": "agent-123",
	})

	if err := dataSourceLiteLLMAgentRead(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if d.Id() != "agent-123" {
		t.Fatalf("expected ID 'agent-123', got %q", d.Id())
	}
	if d.Get("agent_name").(string) != "my-agent" {
		t.Errorf("expected agent_name 'my-agent', got %q", d.Get("agent_name").(string))
	}
	var card map[string]interface{}
	if err := json.Unmarshal([]byte(d.Get("agent_card_params").(string)), &card); err != nil {
		t.Fatalf("agent_card_params not populated as JSON: %v", err)
	}
	if card["url"] != "http://agent.local:9999/" {
		t.Errorf("expected card url, got %v", card["url"])
	}
	if d.Get("spend").(float64) != 1.5 {
		t.Errorf("expected spend 1.5, got %v", d.Get("spend"))
	}
	if d.Get("tpm_limit").(int) != 1000 {
		t.Errorf("expected tpm_limit 1000, got %d", d.Get("tpm_limit").(int))
	}
}

func TestDataSourceLiteLLMAgentsRead(t *testing.T) {
	var gotQuery string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/v1/agents" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		gotQuery = r.URL.RawQuery
		w.Header().Set("Content-Type", "application/json")
		body, _ := json.Marshal([]map[string]interface{}{
			{"agent_id": "agent-1", "agent_name": "first", "tpm_limit": 100, "spend": 0.5},
			{"agent_id": "agent-2", "agent_name": "second"},
		})
		w.Write(body)
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMAgents().Schema, map[string]interface{}{
		"health_check": true,
	})

	if err := dataSourceLiteLLMAgentsRead(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if gotQuery != "health_check=true" {
		t.Errorf("expected health_check=true query, got %q", gotQuery)
	}

	ids := d.Get("ids").([]interface{})
	if len(ids) != 2 || ids[0] != "agent-1" || ids[1] != "agent-2" {
		t.Fatalf("expected ids [agent-1 agent-2], got %v", ids)
	}
	agents := d.Get("agents").([]interface{})
	if len(agents) != 2 {
		t.Fatalf("expected 2 agents, got %d", len(agents))
	}
	first := agents[0].(map[string]interface{})
	if first["agent_name"] != "first" || first["tpm_limit"] != 100 || first["spend"] != 0.5 {
		t.Errorf("unexpected first agent entry: %v", first)
	}
	if d.Id() == "" {
		t.Fatal("expected data source ID to be set")
	}
}
