package litellm

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func TestDataSourceBudgetRead_MapsFields(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/budget/info" || r.Method != http.MethodPost {
			t.Errorf("expected POST /budget/info, got %s %s", r.Method, r.URL.Path)
		}
		var payload map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("failed to decode info payload: %v", err)
		}
		budgets, ok := payload["budgets"].([]interface{})
		if !ok || len(budgets) != 1 || budgets[0] != "bud-ds" {
			t.Errorf("expected budgets ['bud-ds'], got %v", payload["budgets"])
		}
		w.Write(budgetInfoBody("bud-ds"))
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMBudget().Schema, map[string]interface{}{
		"budget_id": "bud-ds",
	})

	if err := dataSourceLiteLLMBudgetRead(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("read failed: %v", err)
	}

	if d.Id() != "bud-ds" {
		t.Fatalf("expected ID 'bud-ds', got %q", d.Id())
	}
	if got := d.Get("max_budget").(float64); got != 100.0 {
		t.Errorf("expected max_budget 100.0, got %v", got)
	}
	if got := d.Get("budget_duration").(string); got != "30d" {
		t.Errorf("expected budget_duration '30d', got %q", got)
	}
	if got := d.Get("budget_reset_at").(string); got != "2026-09-01T00:00:00Z" {
		t.Errorf("expected budget_reset_at set, got %q", got)
	}
}

func TestDataSourceBudgetsRead_MapsList(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/budget/list" || r.Method != http.MethodGet {
			t.Errorf("expected GET /budget/list, got %s %s", r.Method, r.URL.Path)
		}
		body, _ := json.Marshal([]map[string]interface{}{
			{
				"budget_id":        "bud-1",
				"max_budget":       10.0,
				"tpm_limit":        500,
				"model_max_budget": map[string]interface{}{"gpt-4o": map[string]interface{}{"max_budget": 1.0}},
			},
			{
				"budget_id":   "bud-2",
				"soft_budget": 5.0,
			},
		})
		w.Write(body)
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMBudgets().Schema, map[string]interface{}{})

	if err := dataSourceLiteLLMBudgetsRead(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("read failed: %v", err)
	}

	budgets := d.Get("budgets").([]interface{})
	if len(budgets) != 2 {
		t.Fatalf("expected 2 budgets, got %d", len(budgets))
	}
	first := budgets[0].(map[string]interface{})
	if got := first["budget_id"].(string); got != "bud-1" {
		t.Errorf("expected first budget_id 'bud-1', got %q", got)
	}
	if got := first["max_budget"].(float64); got != 10.0 {
		t.Errorf("expected first max_budget 10.0, got %v", got)
	}
	if got := first["tpm_limit"].(int); got != 500 {
		t.Errorf("expected first tpm_limit 500, got %d", got)
	}
	var mmb map[string]interface{}
	if err := json.Unmarshal([]byte(first["model_max_budget"].(string)), &mmb); err != nil {
		t.Fatalf("model_max_budget is not valid JSON: %v", err)
	}
	if _, ok := mmb["gpt-4o"]; !ok {
		t.Errorf("expected gpt-4o key in model_max_budget, got %v", mmb)
	}
	second := budgets[1].(map[string]interface{})
	if got := second["soft_budget"].(float64); got != 5.0 {
		t.Errorf("expected second soft_budget 5.0, got %v", got)
	}
	ids := d.Get("ids").([]interface{})
	if len(ids) != 2 || ids[0] != "bud-1" || ids[1] != "bud-2" {
		t.Errorf("expected ids [bud-1 bud-2], got %v", ids)
	}
}
