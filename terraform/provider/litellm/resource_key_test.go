package litellm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
	"github.com/hashicorp/terraform-plugin-sdk/v2/terraform"
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

// fakeKeyProxy serves /key/info from stored metadata and applies /key/update
// the way the proxy does: an absent "metadata" keeps the stored map, a
// present one replaces it wholesale.
type fakeKeyProxy struct {
	metadata map[string]interface{}
	updates  []map[string]interface{}
}

func (p *fakeKeyProxy) handler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/key/info":
			json.NewEncoder(w).Encode(map[string]interface{}{
				"key":  "hash-1",
				"info": map[string]interface{}{"key_alias": "alias-1", "models": []string{"gpt-4o-mini"}, "metadata": p.metadata},
			})
		case "/key/update":
			var body map[string]interface{}
			json.NewDecoder(r.Body).Decode(&body)
			p.updates = append(p.updates, body)
			if m, ok := body["metadata"].(map[string]interface{}); ok {
				p.metadata = m
			}
			json.NewEncoder(w).Encode(map[string]interface{}{"key": "hash-1", "metadata": p.metadata})
		default:
			http.NotFound(w, r)
		}
	}
}

func applyKeyUpdate(t *testing.T, client *Client, stateAttrs map[string]string, config map[string]interface{}) *terraform.InstanceState {
	t.Helper()
	r := resourceKey()
	state := &terraform.InstanceState{ID: "hash-1", Attributes: stateAttrs}
	diff, err := r.Diff(context.Background(), state, terraform.NewResourceConfigRaw(config), client)
	if err != nil {
		t.Fatalf("Diff returned error: %v", err)
	}
	if diff == nil {
		t.Fatalf("expected a non-empty diff between %v and %v", stateAttrs, config)
	}
	newState, diags := r.Apply(context.Background(), state, diff, client)
	if diags.HasError() {
		t.Fatalf("Apply returned error: %v", diags)
	}
	return newState
}

func TestKeyUpdateWithoutMetadataChangePreservesServerMetadata(t *testing.T) {
	proxy := &fakeKeyProxy{metadata: map[string]interface{}{"a": "1", "server_side": "x", "model_rpm_limit": map[string]interface{}{"gpt-4o-mini": float64(5)}}}
	srv := httptest.NewServer(proxy.handler())
	defer srv.Close()
	client := NewClient(srv.URL, "test-key", true)

	newState := applyKeyUpdate(t, client,
		map[string]string{"key_alias": "alias-1", "max_budget": "10", "metadata.%": "1", "metadata.a": "1"},
		map[string]interface{}{"key_alias": "alias-1", "max_budget": 20, "metadata": map[string]interface{}{"a": "1"}},
	)

	if len(proxy.updates) != 1 {
		t.Fatalf("expected one /key/update call, got %d", len(proxy.updates))
	}
	for _, field := range []string{"metadata", "model_rpm_limit", "model_tpm_limit"} {
		if _, present := proxy.updates[0][field]; present {
			t.Errorf("unchanged %q was sent on /key/update: %v", field, proxy.updates[0][field])
		}
	}
	if proxy.metadata["server_side"] != "x" {
		t.Errorf("server-side metadata lost: %v", proxy.metadata)
	}
	if got := newState.Attributes["metadata.%"]; got != "1" {
		t.Errorf("state metadata should hold only the declared entry, got %v", newState.Attributes)
	}
	if got := newState.Attributes["metadata.a"]; got != "1" {
		t.Errorf("metadata.a = %q, want 1", got)
	}
}

func TestKeyUpdateWithMetadataChangeMergesOverServerMetadata(t *testing.T) {
	proxy := &fakeKeyProxy{metadata: map[string]interface{}{"a": "1", "b": "2", "server_side": "x"}}
	srv := httptest.NewServer(proxy.handler())
	defer srv.Close()
	client := NewClient(srv.URL, "test-key", true)

	applyKeyUpdate(t, client,
		map[string]string{"key_alias": "alias-1", "metadata.%": "2", "metadata.a": "1", "metadata.b": "2"},
		map[string]interface{}{"key_alias": "alias-1", "metadata": map[string]interface{}{"a": "2", "c": "3"}},
	)

	want := map[string]interface{}{"a": "2", "c": "3", "server_side": "x"}
	if !reflect.DeepEqual(proxy.metadata, want) {
		t.Errorf("metadata after update = %v, want %v", proxy.metadata, want)
	}
}

func TestKeyUpdateSendsChangedModelLimits(t *testing.T) {
	proxy := &fakeKeyProxy{metadata: map[string]interface{}{}}
	srv := httptest.NewServer(proxy.handler())
	defer srv.Close()
	client := NewClient(srv.URL, "test-key", true)

	applyKeyUpdate(t, client,
		map[string]string{"key_alias": "alias-1", "model_rpm_limit.%": "1", "model_rpm_limit.gpt-4o-mini": "5"},
		map[string]interface{}{"key_alias": "alias-1", "model_rpm_limit": map[string]interface{}{"gpt-4o-mini": 7}},
	)

	got, ok := proxy.updates[0]["model_rpm_limit"].(map[string]interface{})
	if !ok || got["gpt-4o-mini"] != float64(7) {
		t.Errorf("changed model_rpm_limit not sent: %v", proxy.updates[0])
	}
}

func TestKeyReadKeepsOnlyDeclaredMetadata(t *testing.T) {
	proxy := &fakeKeyProxy{metadata: map[string]interface{}{"a": "1", "server_side": "x"}}
	srv := httptest.NewServer(proxy.handler())
	defer srv.Close()
	client := NewClient(srv.URL, "test-key", true)

	d := newKeyResourceData(t, map[string]interface{}{"metadata": map[string]interface{}{"a": "1"}})
	d.SetId("hash-1")
	if diags := resourceKeyRead(context.Background(), d, client); diags.HasError() {
		t.Fatalf("Read returned error: %v", diags)
	}

	want := map[string]interface{}{"a": "1"}
	if got := d.Get("metadata"); !reflect.DeepEqual(got, want) {
		t.Errorf("metadata in state = %v, want %v", got, want)
	}
}
