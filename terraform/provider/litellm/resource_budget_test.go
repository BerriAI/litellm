package litellm

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func budgetInfoBody(budgetID string) []byte {
	body, _ := json.Marshal([]map[string]interface{}{{
		"budget_id":             budgetID,
		"max_budget":            100.0,
		"soft_budget":           80.0,
		"max_parallel_requests": 10,
		"tpm_limit":             1000,
		"rpm_limit":             60,
		"budget_duration":       "30d",
		"model_max_budget":      map[string]interface{}{"gpt-4o": map[string]interface{}{"max_budget": 5.0}},
		"budget_reset_at":       "2026-09-01T00:00:00Z",
	}})
	return body
}

func TestResourceBudgetCreate_ServerGeneratedID(t *testing.T) {
	var createPayload map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/budget/new":
			if r.Method != http.MethodPost {
				t.Errorf("expected POST /budget/new, got %s", r.Method)
			}
			if err := json.NewDecoder(r.Body).Decode(&createPayload); err != nil {
				t.Fatalf("failed to decode create payload: %v", err)
			}
			w.Write([]byte(`{"budget_id": "bud-generated", "max_budget": 100.0}`))
		case "/budget/info":
			var infoPayload map[string]interface{}
			if err := json.NewDecoder(r.Body).Decode(&infoPayload); err != nil {
				t.Fatalf("failed to decode info payload: %v", err)
			}
			budgets, ok := infoPayload["budgets"].([]interface{})
			if !ok || len(budgets) != 1 || budgets[0] != "bud-generated" {
				t.Errorf("expected budgets ['bud-generated'], got %v", infoPayload["budgets"])
			}
			w.Write(budgetInfoBody("bud-generated"))
		default:
			t.Errorf("unexpected request to %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resourceLiteLLMBudget().Schema, map[string]interface{}{
		"max_budget":       100.0,
		"soft_budget":      80.0,
		"tpm_limit":        1000,
		"model_max_budget": `{"gpt-4o": {"max_budget": 5.0}}`,
	})

	if err := resourceLiteLLMBudgetCreate(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("create failed: %v", err)
	}

	if d.Id() != "bud-generated" {
		t.Fatalf("expected ID 'bud-generated', got %q", d.Id())
	}
	if _, ok := createPayload["budget_id"]; ok {
		t.Errorf("budget_id must be omitted when not configured, got %v", createPayload["budget_id"])
	}
	if got := createPayload["max_budget"]; got != 100.0 {
		t.Errorf("expected max_budget 100.0 in payload, got %v", got)
	}
	if got := createPayload["soft_budget"]; got != 80.0 {
		t.Errorf("expected soft_budget 80.0 in payload, got %v", got)
	}
	mmb, ok := createPayload["model_max_budget"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected model_max_budget object in payload, got %v", createPayload["model_max_budget"])
	}
	if _, ok := mmb["gpt-4o"]; !ok {
		t.Errorf("expected gpt-4o key in model_max_budget, got %v", mmb)
	}
	if got := d.Get("budget_reset_at").(string); got != "2026-09-01T00:00:00Z" {
		t.Errorf("expected budget_reset_at from read, got %q", got)
	}
}

func TestResourceBudgetCreate_ConfiguredID(t *testing.T) {
	var createPayload map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/budget/new":
			if err := json.NewDecoder(r.Body).Decode(&createPayload); err != nil {
				t.Fatalf("failed to decode create payload: %v", err)
			}
			w.Write([]byte(`{"budget_id": "my-budget"}`))
		case "/budget/info":
			w.Write(budgetInfoBody("my-budget"))
		default:
			t.Errorf("unexpected request to %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resourceLiteLLMBudget().Schema, map[string]interface{}{
		"budget_id":  "my-budget",
		"max_budget": 100.0,
	})

	if err := resourceLiteLLMBudgetCreate(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("create failed: %v", err)
	}

	if d.Id() != "my-budget" {
		t.Fatalf("expected ID 'my-budget', got %q", d.Id())
	}
	if got := createPayload["budget_id"]; got != "my-budget" {
		t.Errorf("expected budget_id 'my-budget' in payload, got %v", got)
	}
}

func TestResourceBudgetRead_MapsFields(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(budgetInfoBody("bud-1"))
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resourceLiteLLMBudget().Schema, map[string]interface{}{})
	d.SetId("bud-1")

	if err := resourceLiteLLMBudgetRead(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("read failed: %v", err)
	}

	if got := d.Get("max_budget").(float64); got != 100.0 {
		t.Errorf("expected max_budget 100.0, got %v", got)
	}
	if got := d.Get("soft_budget").(float64); got != 80.0 {
		t.Errorf("expected soft_budget 80.0, got %v", got)
	}
	if got := d.Get("max_parallel_requests").(int); got != 10 {
		t.Errorf("expected max_parallel_requests 10, got %d", got)
	}
	if got := d.Get("tpm_limit").(int); got != 1000 {
		t.Errorf("expected tpm_limit 1000, got %d", got)
	}
	if got := d.Get("rpm_limit").(int); got != 60 {
		t.Errorf("expected rpm_limit 60, got %d", got)
	}
	if got := d.Get("budget_duration").(string); got != "30d" {
		t.Errorf("expected budget_duration '30d', got %q", got)
	}
	var mmb map[string]interface{}
	if err := json.Unmarshal([]byte(d.Get("model_max_budget").(string)), &mmb); err != nil {
		t.Fatalf("model_max_budget in state is not valid JSON: %v", err)
	}
	if _, ok := mmb["gpt-4o"]; !ok {
		t.Errorf("expected gpt-4o key in model_max_budget state, got %v", mmb)
	}
}

func TestResourceBudgetRead_EmptyListClearsID(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`[]`))
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resourceLiteLLMBudget().Schema, map[string]interface{}{})
	d.SetId("gone-budget")

	if err := resourceLiteLLMBudgetRead(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("expected nil error on empty response, got: %v", err)
	}
	if d.Id() != "" {
		t.Fatalf("expected ID to be cleared, got %q", d.Id())
	}
}

func TestResourceBudgetRead_404ClearsID(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resourceLiteLLMBudget().Schema, map[string]interface{}{})
	d.SetId("gone-budget")

	if err := resourceLiteLLMBudgetRead(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("expected nil error on 404, got: %v", err)
	}
	if d.Id() != "" {
		t.Fatalf("expected ID to be cleared on 404, got %q", d.Id())
	}
}

func TestResourceBudgetUpdate_SendsPayload(t *testing.T) {
	var updatePayload map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/budget/update":
			if r.Method != http.MethodPost {
				t.Errorf("expected POST /budget/update, got %s", r.Method)
			}
			if err := json.NewDecoder(r.Body).Decode(&updatePayload); err != nil {
				t.Fatalf("failed to decode update payload: %v", err)
			}
			w.Write([]byte(`{"budget_id": "bud-1"}`))
		case "/budget/info":
			w.Write(budgetInfoBody("bud-1"))
		default:
			t.Errorf("unexpected request to %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resourceLiteLLMBudget().Schema, map[string]interface{}{
		"max_budget": 200.0,
		"rpm_limit":  120,
	})
	d.SetId("bud-1")

	if err := resourceLiteLLMBudgetUpdate(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("update failed: %v", err)
	}

	if got := updatePayload["budget_id"]; got != "bud-1" {
		t.Errorf("expected budget_id 'bud-1' in payload, got %v", got)
	}
	if got := updatePayload["max_budget"]; got != 200.0 {
		t.Errorf("expected max_budget 200.0 in payload, got %v", got)
	}
	if got := updatePayload["rpm_limit"]; got != 120.0 {
		t.Errorf("expected rpm_limit 120 in payload, got %v", got)
	}
}

func TestResourceBudgetDelete_SendsID(t *testing.T) {
	var deletePayload map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/budget/delete" || r.Method != http.MethodPost {
			t.Errorf("expected POST /budget/delete, got %s %s", r.Method, r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&deletePayload); err != nil {
			t.Fatalf("failed to decode delete payload: %v", err)
		}
		w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resourceLiteLLMBudget().Schema, map[string]interface{}{})
	d.SetId("bud-del")

	if err := resourceLiteLLMBudgetDelete(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("delete failed: %v", err)
	}

	if got := deletePayload["id"]; got != "bud-del" {
		t.Fatalf("expected id 'bud-del' in payload, got %v", got)
	}
	if d.Id() != "" {
		t.Fatalf("expected ID to be cleared after delete, got %q", d.Id())
	}
}
