package litellm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func newKeyResourceData(t *testing.T, raw map[string]interface{}) *schema.ResourceData {
	t.Helper()
	return schema.TestResourceDataRaw(t, resourceKey().Schema, raw)
}

func TestMapResourceDataToKeyNewFields(t *testing.T) {
	d := newKeyResourceData(t, map[string]interface{}{
		"budget_id":                  "budget-1",
		"enforced_params":            []interface{}{"user"},
		"allowed_routes":             []interface{}{"/chat/completions"},
		"allowed_passthrough_routes": []interface{}{"/vertex-ai"},
		"rpm_limit_type":             "guaranteed_throughput",
		"tpm_limit_type":             "best_effort_throughput",
		"prompts":                    []interface{}{"prompt-1"},
		"organization_id":            "org-1",
		"project_id":                 "proj-1",
	})

	key := &Key{}
	mapResourceDataToKey(d, key)

	if key.BudgetID != "budget-1" {
		t.Errorf("BudgetID = %q, want budget-1", key.BudgetID)
	}
	if len(key.EnforcedParams) != 1 || key.EnforcedParams[0] != "user" {
		t.Errorf("EnforcedParams = %v, want [user]", key.EnforcedParams)
	}
	if len(key.AllowedRoutes) != 1 || key.AllowedRoutes[0] != "/chat/completions" {
		t.Errorf("AllowedRoutes = %v", key.AllowedRoutes)
	}
	if len(key.AllowedPassthroughRoutes) != 1 || key.AllowedPassthroughRoutes[0] != "/vertex-ai" {
		t.Errorf("AllowedPassthroughRoutes = %v", key.AllowedPassthroughRoutes)
	}
	if key.RPMLimitType != "guaranteed_throughput" {
		t.Errorf("RPMLimitType = %q", key.RPMLimitType)
	}
	if key.TPMLimitType != "best_effort_throughput" {
		t.Errorf("TPMLimitType = %q", key.TPMLimitType)
	}
	if len(key.Prompts) != 1 || key.Prompts[0] != "prompt-1" {
		t.Errorf("Prompts = %v", key.Prompts)
	}
	if key.OrganizationID != "org-1" {
		t.Errorf("OrganizationID = %q", key.OrganizationID)
	}
	if key.ProjectID != "proj-1" {
		t.Errorf("ProjectID = %q", key.ProjectID)
	}
}

func TestUpdateKeySendsNewFields(t *testing.T) {
	var captured map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &captured)
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"key": "sk-test"}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	_, err := client.UpdateKey(&Key{
		Key:                      "sk-test",
		BudgetID:                 "budget-1",
		EnforcedParams:           []string{"user"},
		AllowedRoutes:            []string{"/chat/completions"},
		AllowedPassthroughRoutes: []string{"/vertex-ai"},
		RPMLimitType:             "guaranteed_throughput",
		TPMLimitType:             "dynamic",
		Prompts:                  []string{"prompt-1"},
		OrganizationID:           "org-1",
	})
	if err != nil {
		t.Fatalf("UpdateKey returned error: %v", err)
	}

	want := map[string]interface{}{
		"budget_id":       "budget-1",
		"rpm_limit_type":  "guaranteed_throughput",
		"tpm_limit_type":  "dynamic",
		"organization_id": "org-1",
	}
	for k, v := range want {
		if captured[k] != v {
			t.Errorf("update payload %s = %v, want %v", k, captured[k], v)
		}
	}
	for _, k := range []string{"enforced_params", "allowed_routes", "allowed_passthrough_routes", "prompts"} {
		list, ok := captured[k].([]interface{})
		if !ok || len(list) != 1 {
			t.Errorf("update payload %s = %v, want single-element list", k, captured[k])
		}
	}
}

func TestUpdateKeyOmitsUnsetNewFields(t *testing.T) {
	var captured map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &captured)
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"key": "sk-test"}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	if _, err := client.UpdateKey(&Key{Key: "sk-test"}); err != nil {
		t.Fatalf("UpdateKey returned error: %v", err)
	}

	for _, k := range []string{
		"budget_id", "enforced_params", "allowed_routes", "allowed_passthrough_routes",
		"rpm_limit_type", "tpm_limit_type", "prompts", "organization_id",
	} {
		if _, present := captured[k]; present {
			t.Errorf("update payload unexpectedly contains %s", k)
		}
	}
}

func TestParseKeyResponseNewFields(t *testing.T) {
	client := NewClient("http://localhost:4000", "test-key", true)
	resp := map[string]interface{}{
		"key":                        "sk-test",
		"budget_id":                  "budget-1",
		"enforced_params":            []interface{}{"user"},
		"allowed_routes":             []interface{}{"/chat/completions"},
		"allowed_passthrough_routes": []interface{}{"/vertex-ai"},
		"rpm_limit_type":             "guaranteed_throughput",
		"tpm_limit_type":             "best_effort_throughput",
		"prompts":                    []interface{}{"prompt-1"},
		"organization_id":            "org-1",
		"project_id":                 "proj-1",
	}

	key, err := client.parseKeyResponse(resp)
	if err != nil {
		t.Fatalf("parseKeyResponse returned error: %v", err)
	}
	if key.BudgetID != "budget-1" || key.OrganizationID != "org-1" || key.ProjectID != "proj-1" {
		t.Errorf("string fields not parsed: %+v", key)
	}
	if key.RPMLimitType != "guaranteed_throughput" || key.TPMLimitType != "best_effort_throughput" {
		t.Errorf("limit types not parsed: %+v", key)
	}
	if len(key.EnforcedParams) != 1 || len(key.AllowedRoutes) != 1 || len(key.AllowedPassthroughRoutes) != 1 || len(key.Prompts) != 1 {
		t.Errorf("list fields not parsed: %+v", key)
	}
}

// A config-supplied key value must be forwarded to /key/generate; previously
// it was silently dropped and the proxy generated a random key instead.
func TestCreateKeySendsConfigSuppliedKey(t *testing.T) {
	var captured map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/key/generate" {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &captured)
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"key": "sk-custom", "token_id": "hash-1"}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"key": "sk-custom", "token_id": "hash-1"}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newKeyResourceData(t, map[string]interface{}{"key": "sk-custom"})

	diags := resourceKeyCreate(context.Background(), d, client)
	if diags.HasError() {
		t.Fatalf("create returned error: %v", diags)
	}
	if captured["key"] != "sk-custom" {
		t.Errorf("create payload key = %v, want sk-custom", captured["key"])
	}
	if d.Id() != "hash-1" {
		t.Errorf("resource ID = %q, want hash-1", d.Id())
	}
}

// The proxy 400s on budget_duration: "", so an unset duration must be
// omitted from the update payload entirely.
func TestUpdateKeyOmitsEmptyBudgetDuration(t *testing.T) {
	var captured map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &captured)
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"key": "sk-test"}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	if _, err := client.UpdateKey(&Key{Key: "sk-test"}); err != nil {
		t.Fatalf("UpdateKey returned error: %v", err)
	}
	if _, present := captured["budget_duration"]; present {
		t.Errorf("update payload contains empty budget_duration: %v", captured["budget_duration"])
	}

	if _, err := client.UpdateKey(&Key{Key: "sk-test", BudgetDuration: "30d"}); err != nil {
		t.Fatalf("UpdateKey returned error: %v", err)
	}
	if captured["budget_duration"] != "30d" {
		t.Errorf("budget_duration = %v, want 30d", captured["budget_duration"])
	}
}

// /key/info nests the key's fields under "info"; GetKey must unwrap that
// envelope or reads map nothing back into state.
func TestGetKeyUnwrapsInfoEnvelope(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{
			"key": "hash-1",
			"info": {
				"key_alias": "envelope-alias",
				"models": ["gpt-4o-mini"],
				"budget_id": "budget-1",
				"team_id": "team-1",
				"rpm_limit": 100
			}
		}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	key, err := client.GetKey("hash-1")
	if err != nil {
		t.Fatalf("GetKey returned error: %v", err)
	}
	if key.KeyAlias != "envelope-alias" {
		t.Errorf("KeyAlias = %q, want envelope-alias (info envelope not unwrapped)", key.KeyAlias)
	}
	if key.BudgetID != "budget-1" || key.TeamID != "team-1" {
		t.Errorf("nested fields not parsed: %+v", key)
	}
	if key.RPMLimit == nil || *key.RPMLimit != 100 {
		t.Errorf("RPMLimit not parsed: %+v", key.RPMLimit)
	}
}
