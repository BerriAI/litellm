package litellm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
	"github.com/hashicorp/terraform-plugin-sdk/v2/terraform"
)

func tagInfoBody(name string) string {
	return `{"` + name + `": {
		"name": "` + name + `",
		"description": "Production traffic",
		"models": ["model-1", "model-2"],
		"created_at": "2026-01-01T00:00:00",
		"updated_at": "2026-01-02T00:00:00",
		"created_by": "admin",
		"litellm_budget_table": {
			"budget_id": "bud-1",
			"max_budget": 50.5,
			"soft_budget": 40.0,
			"max_parallel_requests": 5,
			"tpm_limit": 1000,
			"rpm_limit": 100,
			"budget_duration": "30d"
		}
	}}`
}

func TestResourceLiteLLMTagCreate(t *testing.T) {
	var createPayload map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/tag/new":
			if err := json.NewDecoder(r.Body).Decode(&createPayload); err != nil {
				t.Errorf("failed to decode create payload: %v", err)
			}
			w.Write([]byte(`{"message": "created"}`))
		case "/tag/info":
			w.Write([]byte(tagInfoBody("prod")))
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resourceLiteLLMTag().Schema, map[string]interface{}{
		"name":             "prod",
		"description":      "Production traffic",
		"models":           []interface{}{"model-1", "model-2"},
		"max_budget":       50.5,
		"tpm_limit":        1000,
		"model_max_budget": `{"gpt-4": {"budget_limit": 10}}`,
	})

	if err := resourceLiteLLMTagCreate(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("create failed: %v", err)
	}

	if d.Id() != "prod" {
		t.Fatalf("expected ID 'prod', got %q", d.Id())
	}
	if createPayload["name"] != "prod" {
		t.Errorf("expected payload name 'prod', got %v", createPayload["name"])
	}
	if createPayload["description"] != "Production traffic" {
		t.Errorf("expected payload description, got %v", createPayload["description"])
	}
	if !reflect.DeepEqual(createPayload["models"], []interface{}{"model-1", "model-2"}) {
		t.Errorf("expected payload models, got %v", createPayload["models"])
	}
	if createPayload["max_budget"] != 50.5 {
		t.Errorf("expected payload max_budget 50.5, got %v", createPayload["max_budget"])
	}
	if createPayload["tpm_limit"] != float64(1000) {
		t.Errorf("expected payload tpm_limit 1000, got %v", createPayload["tpm_limit"])
	}
	modelMaxBudget, ok := createPayload["model_max_budget"].(map[string]interface{})
	if !ok || modelMaxBudget["gpt-4"] == nil {
		t.Errorf("expected model_max_budget sent as JSON object, got %v", createPayload["model_max_budget"])
	}
	if got := d.Get("budget_id").(string); got != "bud-1" {
		t.Errorf("expected budget_id 'bud-1' from read, got %q", got)
	}
}

func TestResourceLiteLLMTagCreate_InvalidModelMaxBudget(t *testing.T) {
	d := schema.TestResourceDataRaw(t, resourceLiteLLMTag().Schema, map[string]interface{}{
		"name":             "prod",
		"model_max_budget": "not-json",
	})

	if err := resourceLiteLLMTagCreate(d, NewClient("http://127.0.0.1:1", "test-key", true)); err == nil {
		t.Fatal("expected error for invalid model_max_budget JSON, got nil")
	}
}

func TestResourceLiteLLMTagRead(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/tag/info" {
			t.Errorf("unexpected request path: %s", r.URL.Path)
		}
		var payload map[string]interface{}
		json.NewDecoder(r.Body).Decode(&payload)
		if !reflect.DeepEqual(payload["names"], []interface{}{"prod"}) {
			t.Errorf("expected names ['prod'], got %v", payload["names"])
		}
		w.Write([]byte(tagInfoBody("prod")))
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resourceLiteLLMTag().Schema, map[string]interface{}{"name": "prod"})
	d.SetId("prod")

	if err := resourceLiteLLMTagRead(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("read failed: %v", err)
	}

	checks := map[string]interface{}{
		"description":           "Production traffic",
		"budget_id":             "bud-1",
		"max_budget":            50.5,
		"soft_budget":           40.0,
		"max_parallel_requests": 5,
		"tpm_limit":             1000,
		"rpm_limit":             100,
		"budget_duration":       "30d",
	}
	for key, want := range checks {
		if got := d.Get(key); got != want {
			t.Errorf("expected %s %v, got %v", key, want, got)
		}
	}
	if !reflect.DeepEqual(d.Get("models"), []interface{}{"model-1", "model-2"}) {
		t.Errorf("expected models in state, got %v", d.Get("models"))
	}
}

const tagInfoWithModelMaxBudget = `{"prod": {
	"name": "prod",
	"litellm_budget_table": {
		"budget_id": "bud-created-by-proxy",
		"model_max_budget": {"gpt-4": {"max_budget": 25.0, "budget_duration": "1d"}}
	}
}}`

// readTagFromProxy returns the state a Read leaves behind for a tag configured with config.
func readTagFromProxy(t *testing.T, config map[string]interface{}) *schema.ResourceData {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(tagInfoWithModelMaxBudget))
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resourceLiteLLMTag().Schema, config)
	d.SetId("prod")
	if err := resourceLiteLLMTagRead(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("read failed: %v", err)
	}
	return d
}

func planAttributes(t *testing.T, prior *schema.ResourceData, config map[string]interface{}) map[string]*terraform.ResourceAttrDiff {
	t.Helper()
	diff, err := resourceLiteLLMTag().Diff(context.Background(), prior.State(), terraform.NewResourceConfigRaw(config), nil)
	if err != nil {
		t.Fatalf("diff failed: %v", err)
	}
	if diff == nil {
		return nil
	}
	return diff.Attributes
}

func TestResourceLiteLLMTagRead_PicksUpModelMaxBudgetChangedOnTheProxy(t *testing.T) {
	d := readTagFromProxy(t, map[string]interface{}{
		"name":             "prod",
		"model_max_budget": `{"gpt-4": {"max_budget": 10, "budget_duration": "1d"}}`,
	})

	if !budgetSuppressEquivalentJSON("", d.Get("model_max_budget").(string), `{"gpt-4": {"max_budget": 25, "budget_duration": "1d"}}`, nil) {
		t.Errorf("expected the proxy's per-model budget in state, got %s", d.Get("model_max_budget"))
	}
}

func TestResourceLiteLLMTag_ModelMaxBudgetClearedOnTheProxyPlansItBack(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"prod": {"name": "prod", "litellm_budget_table": {"budget_id": "bud-1", "model_max_budget": null}}}`))
	}))
	defer srv.Close()
	config := map[string]interface{}{
		"name":             "prod",
		"model_max_budget": `{"gpt-4": {"max_budget": 10, "budget_duration": "1d"}}`,
	}
	prior := schema.TestResourceDataRaw(t, resourceLiteLLMTag().Schema, config)
	prior.SetId("prod")
	if err := resourceLiteLLMTagRead(prior, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("read failed: %v", err)
	}

	attrs := planAttributes(t, prior, config)
	if attrs["model_max_budget"] == nil || attrs["model_max_budget"].Old != "" {
		t.Errorf("expected the plan to set model_max_budget again after it was cleared on the proxy, got %+v", attrs["model_max_budget"])
	}
}

func TestResourceLiteLLMTag_PlanAfterInlineBudgetCreateKeepsTheProxyBudgetID(t *testing.T) {
	config := map[string]interface{}{
		"name":             "prod",
		"model_max_budget": `{"gpt-4": {"max_budget": 25, "budget_duration": "1d"}}`,
	}
	prior := readTagFromProxy(t, config)

	if attrs := planAttributes(t, prior, config); attrs["budget_id"] != nil {
		t.Errorf("expected no budget_id change when the config leaves it unset, got %+v", attrs["budget_id"])
	}
}

func TestResourceLiteLLMTag_EquivalentModelMaxBudgetJSONPlansNoChange(t *testing.T) {
	prior := readTagFromProxy(t, map[string]interface{}{"name": "prod"})

	attrs := planAttributes(t, prior, map[string]interface{}{
		"name":             "prod",
		"model_max_budget": `{"gpt-4": {"max_budget": 25.0, "budget_duration": "1d"}}`,
	})
	if attrs["model_max_budget"] != nil {
		t.Errorf("expected key order and number formatting to plan no change, got %+v", attrs["model_max_budget"])
	}
}

func TestResourceLiteLLMTagRead_404ClearsID(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resourceLiteLLMTag().Schema, map[string]interface{}{"name": "gone"})
	d.SetId("gone")

	if err := resourceLiteLLMTagRead(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("expected nil error on 404, got: %v", err)
	}
	if d.Id() != "" {
		t.Fatalf("expected ID cleared on 404, got %q", d.Id())
	}
}

// The proxy wraps its internal 404 into a 500 whose detail mentions "Tags not found".
func TestResourceLiteLLMTagRead_WrappedNotFoundClearsID(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"detail": "404: Tags not found: ['gone']"}`))
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resourceLiteLLMTag().Schema, map[string]interface{}{"name": "gone"})
	d.SetId("gone")

	if err := resourceLiteLLMTagRead(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("expected nil error on wrapped not-found, got: %v", err)
	}
	if d.Id() != "" {
		t.Fatalf("expected ID cleared on wrapped not-found, got %q", d.Id())
	}
}

func TestResourceLiteLLMTagUpdate(t *testing.T) {
	var updatePayload map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/tag/update":
			if err := json.NewDecoder(r.Body).Decode(&updatePayload); err != nil {
				t.Errorf("failed to decode update payload: %v", err)
			}
			w.Write([]byte(`{"message": "updated"}`))
		case "/tag/info":
			w.Write([]byte(tagInfoBody("prod")))
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resourceLiteLLMTag().Schema, map[string]interface{}{
		"name":        "prod",
		"description": "Updated description",
		"rpm_limit":   200,
	})
	d.SetId("prod")

	if err := resourceLiteLLMTagUpdate(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("update failed: %v", err)
	}

	if updatePayload["name"] != "prod" {
		t.Errorf("expected update payload name 'prod', got %v", updatePayload["name"])
	}
	if updatePayload["description"] != "Updated description" {
		t.Errorf("expected updated description in payload, got %v", updatePayload["description"])
	}
	if updatePayload["rpm_limit"] != float64(200) {
		t.Errorf("expected rpm_limit 200 in payload, got %v", updatePayload["rpm_limit"])
	}
}

func TestResourceLiteLLMTagDelete(t *testing.T) {
	var deletePayload map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/tag/delete" || r.Method != http.MethodPost {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&deletePayload); err != nil {
			t.Errorf("failed to decode delete payload: %v", err)
		}
		w.Write([]byte(`{"message": "deleted"}`))
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resourceLiteLLMTag().Schema, map[string]interface{}{"name": "prod"})
	d.SetId("prod")

	if err := resourceLiteLLMTagDelete(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("delete failed: %v", err)
	}

	if deletePayload["name"] != "prod" {
		t.Errorf("expected delete payload name 'prod', got %v", deletePayload["name"])
	}
	if d.Id() != "" {
		t.Fatalf("expected ID cleared after delete, got %q", d.Id())
	}
}
