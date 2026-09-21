package litellm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sync/atomic"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/diag"
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

// The proxy validates each model_max_budget entry as a BudgetConfig object and
// 500s on a bare number, so the JSON string must reach /key/generate as nested
// objects and the proxy's response must map back to equivalent JSON in state.
func TestCreateKeySendsModelMaxBudgetAsBudgetObjects(t *testing.T) {
	var captured map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/key/generate" {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &captured)
			w.Write([]byte(`{"key": "sk-test", "token_id": "hash-1"}`))
			return
		}
		w.Write([]byte(`{"key": "hash-1", "info": {"model_max_budget": {"gpt-4o-mini": {"budget_limit": 50, "time_period": "30d", "rpm_limit": 60}}}}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newKeyResourceData(t, map[string]interface{}{
		"model_max_budget": `{"gpt-4o-mini": {"budget_limit": 50, "time_period": "30d"}}`,
	})

	if diags := resourceKeyCreate(context.Background(), d, client); diags.HasError() {
		t.Fatalf("create returned error: %v", diags)
	}

	budgets, ok := captured["model_max_budget"].(map[string]interface{})
	if !ok {
		t.Fatalf("create payload model_max_budget = %v, want object", captured["model_max_budget"])
	}
	cfg, ok := budgets["gpt-4o-mini"].(map[string]interface{})
	if !ok {
		t.Fatalf("model_max_budget[gpt-4o-mini] = %v, want BudgetConfig object", budgets["gpt-4o-mini"])
	}
	if cfg["budget_limit"] != float64(50) || cfg["time_period"] != "30d" {
		t.Errorf("BudgetConfig = %v, want budget_limit 50 and time_period 30d", cfg)
	}

	var state map[string]interface{}
	if err := json.Unmarshal([]byte(d.Get("model_max_budget").(string)), &state); err != nil {
		t.Fatalf("state model_max_budget %q is not JSON: %v", d.Get("model_max_budget"), err)
	}
	if got, _ := state["gpt-4o-mini"].(map[string]interface{}); got["budget_limit"] != float64(50) || got["rpm_limit"] != float64(60) {
		t.Errorf("state model_max_budget = %v, want the BudgetConfig read back from /key/info", state)
	}
}

// Schema version 0 stored model_max_budget as map(number); that state cannot
// decode into the version 1 string attribute, so the upgrader must drop it.
func TestKeyStateUpgradeV0DropsMapModelMaxBudget(t *testing.T) {
	upgraded, err := resourceKey().StateUpgraders[0].Upgrade(context.Background(), map[string]interface{}{
		"id":               "hash-1",
		"key_alias":        "legacy",
		"model_max_budget": map[string]interface{}{"gpt-4o-mini": 50.0},
	}, nil)
	if err != nil {
		t.Fatalf("upgrade returned error: %v", err)
	}
	if _, present := upgraded["model_max_budget"]; present {
		t.Errorf("upgraded state still carries map model_max_budget: %v", upgraded["model_max_budget"])
	}
	if upgraded["key_alias"] != "legacy" {
		t.Errorf("upgrade dropped unrelated attribute: %v", upgraded)
	}
}

func TestKeyModelMaxBudgetValidationRequiresBudgetObjects(t *testing.T) {
	validate := resourceKey().Schema["model_max_budget"].ValidateFunc
	for _, valid := range []string{
		`{}`,
		`{"gpt-4o-mini": {"budget_limit": 50, "time_period": "30d"}}`,
		`{"gpt-4o-mini": {"max_budget": 50, "rpm_limit": 60}, "gpt-4o": {"budget_duration": "1d", "tpm_limit": 1000}}`,
	} {
		if _, errs := validate(valid, "model_max_budget"); len(errs) != 0 {
			t.Errorf("validate(%s) = %v, want accepted", valid, errs)
		}
	}
	for _, invalid := range []string{
		`null`,
		`[]`,
		`"gpt-4o-mini"`,
		`50`,
		`{"gpt-4o-mini": 50}`,
		`{"gpt-4o-mini": null}`,
		`{"gpt-4o-mini": [50]}`,
		`{"gpt-4o-mini": {}}`,
		`{"gpt-4o-mini": {"budget_limt": 50}}`,
		`{"gpt-4o-mini": {"budget_limit": 50, "max_tokens": 100}}`,
		`not json`,
	} {
		if _, errs := validate(invalid, "model_max_budget"); len(errs) == 0 {
			t.Errorf("validate(%s) accepted a value that would send no per-model budget", invalid)
		}
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

func TestResourceKeyUpdateFailureKeepsPriorState(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/key/update" {
			w.WriteHeader(http.StatusBadRequest)
			w.Write([]byte(`{"error":{"message":"Invalid budget_duration 'bad'"}}`))
			return
		}
		w.Write([]byte(`{"key":"hash-1","info":{"key_alias":"demo","models":["fake-model"]}}`))
	}))
	defer srv.Close()

	res := resourceKey()
	priorData := newKeyResourceData(t, map[string]interface{}{
		"key_alias": "demo",
		"models":    []interface{}{"fake-model"},
	})
	priorData.SetId("hash-1")
	prior := priorData.State()
	config := terraform.NewResourceConfigRaw(map[string]interface{}{
		"key_alias":       "demo",
		"models":          []interface{}{"fake-model"},
		"budget_duration": "bad",
	})
	diff, err := res.Diff(context.Background(), prior, config, nil)
	if err != nil {
		t.Fatalf("diff failed: %v", err)
	}

	newState, diags := res.Apply(context.Background(), prior, diff, NewClient(srv.URL, "test-key", true))
	if !diags.HasError() {
		t.Fatal("apply succeeded, want the proxy's 400 surfaced as an error")
	}
	if got, ok := newState.Attributes["budget_duration"]; ok {
		t.Errorf("failed update persisted budget_duration=%q into state, want it absent", got)
	}
	if newState.Attributes["key_alias"] != "demo" {
		t.Errorf("prior key_alias lost from state: %v", newState.Attributes)
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

func TestGetKeyReadsFieldsStoredInMetadata(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{
			"key": "hash-1",
			"info": {
				"models": ["gpt-4o-mini"],
				"metadata": {
					"team": "core-infra",
					"model_rpm_limit": {"gpt-4o-mini": 7},
					"model_tpm_limit": {"gpt-4o-mini": 10000},
					"guardrails": ["pii-guard"],
					"tags": ["prod"],
					"enforced_params": ["user"],
					"allowed_passthrough_routes": ["/v1/foo"],
					"rpm_limit_type": "guaranteed_throughput",
					"tpm_limit_type": "dynamic",
					"prompts": ["p1"]
				}
			}
		}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	key, err := client.GetKey("hash-1")
	if err != nil {
		t.Fatalf("GetKey returned error: %v", err)
	}
	if got, ok := key.ModelRPMLimit["gpt-4o-mini"].(float64); !ok || got != 7 {
		t.Errorf("ModelRPMLimit = %v, want gpt-4o-mini=7 read from metadata", key.ModelRPMLimit)
	}
	if got, ok := key.ModelTPMLimit["gpt-4o-mini"].(float64); !ok || got != 10000 {
		t.Errorf("ModelTPMLimit = %v, want gpt-4o-mini=10000 read from metadata", key.ModelTPMLimit)
	}
	if len(key.Guardrails) != 1 || key.Guardrails[0] != "pii-guard" {
		t.Errorf("Guardrails = %v, want [pii-guard]", key.Guardrails)
	}
	if len(key.Tags) != 1 || key.Tags[0] != "prod" {
		t.Errorf("Tags = %v, want [prod]", key.Tags)
	}
	if len(key.EnforcedParams) != 1 || key.EnforcedParams[0] != "user" {
		t.Errorf("EnforcedParams = %v, want [user]", key.EnforcedParams)
	}
	if len(key.AllowedPassthroughRoutes) != 1 || key.AllowedPassthroughRoutes[0] != "/v1/foo" {
		t.Errorf("AllowedPassthroughRoutes = %v, want [/v1/foo]", key.AllowedPassthroughRoutes)
	}
	if key.RPMLimitType != "guaranteed_throughput" || key.TPMLimitType != "dynamic" {
		t.Errorf("limit types = %q/%q, want guaranteed_throughput/dynamic", key.RPMLimitType, key.TPMLimitType)
	}
	if len(key.Prompts) != 1 || key.Prompts[0] != "p1" {
		t.Errorf("Prompts = %v, want [p1]", key.Prompts)
	}
	if key.Metadata["team"] != "core-infra" {
		t.Errorf("Metadata = %v, want team=core-infra preserved", key.Metadata)
	}
}

func TestGetKeyPrefersTopLevelOverMetadataCopy(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{
			"key": "hash-1",
			"info": {
				"tags": ["top-level"],
				"guardrails": null,
				"metadata": {
					"tags": ["from-metadata"],
					"guardrails": ["from-metadata"]
				}
			}
		}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	key, err := client.GetKey("hash-1")
	if err != nil {
		t.Fatalf("GetKey returned error: %v", err)
	}
	if len(key.Tags) != 1 || key.Tags[0] != "top-level" {
		t.Errorf("Tags = %v, want [top-level]", key.Tags)
	}
	if len(key.Guardrails) != 1 || key.Guardrails[0] != "from-metadata" {
		t.Errorf("Guardrails = %v, want [from-metadata] (null top-level must not shadow)", key.Guardrails)
	}
}

func TestResourceKeyReadDropsMissingKeyFromState(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		w.Write([]byte(`{"error":{"message":"Key not found in database","type":"not_found_error","param":"key","code":"404"}}`))
	}))
	defer srv.Close()

	d := newKeyResourceData(t, map[string]interface{}{"key_alias": "stale"})
	d.SetId("deleted-out-of-band")

	diags := resourceKeyRead(context.Background(), d, NewClient(srv.URL, "test-key", true))
	if diags.HasError() {
		t.Fatalf("read of a missing key must not error, got: %v", diags)
	}
	if d.Id() != "" {
		t.Errorf("Id = %q, want empty so Terraform plans a recreate", d.Id())
	}
}

func TestResourceKeyReadStillFailsOnNon404Errors(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"error":{"message":"db down"}}`))
	}))
	defer srv.Close()

	d := newKeyResourceData(t, map[string]interface{}{"key_alias": "live"})
	d.SetId("still-exists")

	diags := resourceKeyRead(context.Background(), d, NewClient(srv.URL, "test-key", true))
	if !diags.HasError() {
		t.Fatal("a 500 from /key/info must surface as an error, not be treated as a deleted key")
	}
	if d.Id() != "still-exists" {
		t.Errorf("Id = %q, want unchanged on a transient error", d.Id())
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

func TestKeyUpdateSendsChangedDuration(t *testing.T) {
	proxy := &fakeKeyProxy{metadata: map[string]interface{}{}}
	srv := httptest.NewServer(proxy.handler())
	defer srv.Close()
	client := NewClient(srv.URL, "test-key", true)

	applyKeyUpdate(t, client,
		map[string]string{"key_alias": "alias-1", "duration": "30d"},
		map[string]interface{}{"key_alias": "alias-1", "duration": "90d"},
	)

	if got := proxy.updates[0]["duration"]; got != "90d" {
		t.Errorf("update payload duration = %v, want 90d", got)
	}
}

func TestKeyUpdateOmitsUnchangedDuration(t *testing.T) {
	proxy := &fakeKeyProxy{metadata: map[string]interface{}{}}
	srv := httptest.NewServer(proxy.handler())
	defer srv.Close()
	client := NewClient(srv.URL, "test-key", true)

	applyKeyUpdate(t, client,
		map[string]string{"key_alias": "alias-1", "duration": "30d"},
		map[string]interface{}{"key_alias": "alias-2", "duration": "30d"},
	)

	if got := proxy.updates[0]["key_alias"]; got != "alias-2" {
		t.Fatalf("update payload key_alias = %v, want alias-2", got)
	}
	if v, present := proxy.updates[0]["duration"]; present {
		t.Errorf("update payload unexpectedly contains duration = %v", v)
	}
}

// newKeyUpdateResourceData builds a *schema.ResourceData reflecting a real
// state -> config diff for team_id (unlike schema.TestResourceDataRaw, which
// has no notion of prior state), so d.HasChange("team_id") behaves the way it
// does during a real Update call.
func newKeyUpdateResourceData(t *testing.T, id, oldTeamID, newTeamID string) *schema.ResourceData {
	t.Helper()
	state := &terraform.InstanceState{ID: id, Attributes: map[string]string{"team_id": oldTeamID}}
	diff := &terraform.InstanceDiff{Attributes: map[string]*terraform.ResourceAttrDiff{
		"team_id": {Old: oldTeamID, New: newTeamID},
	}}
	d, err := schema.InternalMap(resourceKey().Schema).Data(state, diff)
	if err != nil {
		t.Fatalf("building ResourceData returned error: %v", err)
	}
	return d
}

// keyRecoveryProxy fakes the two responses the cascade-delete recovery path
// turns on: what POST /key/update returns, and whether GET /key/info still
// finds the key afterwards.
type keyRecoveryProxy struct {
	updateStatus  int
	updateBody    string
	staleKeyGone  bool
	updateCalls   int32
	generateCalls int32
}

const keyNotFoundBody = `{"error":{"message":"Key not found.","type":"not_found_error","param":"key","code":"404"}}`

func (p *keyRecoveryProxy) handler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/key/update":
			atomic.AddInt32(&p.updateCalls, 1)
			w.WriteHeader(p.updateStatus)
			io.WriteString(w, p.updateBody)
		case "/key/generate":
			atomic.AddInt32(&p.generateCalls, 1)
			io.WriteString(w, `{"key": "sk-new", "token_id": "new-token"}`)
		case "/key/info":
			requested := r.URL.Query().Get("key")
			if p.staleKeyGone && requested != "new-token" {
				w.WriteHeader(http.StatusNotFound)
				io.WriteString(w, keyNotFoundBody)
				return
			}
			json.NewEncoder(w).Encode(map[string]interface{}{
				"key":  requested,
				"info": map[string]interface{}{"team_id": "team-b"},
			})
		default:
			http.NotFound(w, r)
		}
	}
}

func runKeyUpdate(t *testing.T, p *keyRecoveryProxy, d *schema.ResourceData) diag.Diagnostics {
	t.Helper()
	srv := httptest.NewServer(p.handler())
	defer srv.Close()
	return resourceKeyUpdate(context.Background(), d, NewClient(srv.URL, "test-key", true))
}

// Reassigning a key between two teams that both still exist is a plain
// in-place /key/update and must not be turned into a destroy/recreate.
func TestResourceKeyUpdateTeamReassignmentStaysInPlace(t *testing.T) {
	proxy := &keyRecoveryProxy{updateStatus: http.StatusOK, updateBody: `{"key": "hash-1"}`}
	d := newKeyUpdateResourceData(t, "hash-1", "team-a", "team-b")

	if diags := runKeyUpdate(t, proxy, d); diags.HasError() {
		t.Fatalf("update returned error: %v", diags)
	}
	if got := atomic.LoadInt32(&proxy.generateCalls); got != 0 {
		t.Errorf("a benign team reassignment must not recreate the key, got %d /key/generate calls", got)
	}
	if d.Id() != "hash-1" {
		t.Errorf("Id = %q, want hash-1 unchanged", d.Id())
	}
}

// The reported bug: the key was cascade-deleted along with its old team, so
// /key/update 404s and the apply must recover by recreating it.
func TestResourceKeyUpdateRecreatesCascadeDeletedKey(t *testing.T) {
	proxy := &keyRecoveryProxy{updateStatus: http.StatusNotFound, updateBody: keyNotFoundBody, staleKeyGone: true}
	d := newKeyUpdateResourceData(t, "stale-token", "team-a", "team-b")

	if diags := runKeyUpdate(t, proxy, d); diags.HasError() {
		t.Fatalf("a cascade-deleted key must be recreated, not error: %v", diags)
	}
	if got := atomic.LoadInt32(&proxy.updateCalls); got != 1 {
		t.Errorf("expected 1 /key/update attempt before recovering, got %d", got)
	}
	if got := atomic.LoadInt32(&proxy.generateCalls); got != 1 {
		t.Errorf("expected exactly 1 /key/generate recreate, got %d", got)
	}
	if d.Id() != "new-token" {
		t.Errorf("Id = %q, want the recreated key's new-token", d.Id())
	}
}

// /key/update 404s for reasons other than a missing key, a rejected
// project_id among them. Recovering on the status code alone would orphan a
// key that is still live on the proxy, so the key's absence must be confirmed.
func TestResourceKeyUpdateNotFoundWithLiveKeyFailsLoudly(t *testing.T) {
	proxy := &keyRecoveryProxy{
		updateStatus: http.StatusNotFound,
		updateBody:   `{"error":{"message":"Project not found, project_id=proj-1"}}`,
	}
	d := newKeyUpdateResourceData(t, "hash-1", "team-a", "team-b")

	if diags := runKeyUpdate(t, proxy, d); !diags.HasError() {
		t.Fatal("a 404 on a key that still exists must stay an error")
	}
	if got := atomic.LoadInt32(&proxy.generateCalls); got != 0 {
		t.Errorf("expected no recreate while the key is still live, got %d /key/generate calls", got)
	}
	if d.Id() != "hash-1" {
		t.Errorf("Id = %q, want hash-1 untouched on a hard failure", d.Id())
	}
}

// A key gone for some reason unrelated to a team move still fails loudly.
func TestResourceKeyUpdateNotFoundWithoutTeamChangeFailsLoudly(t *testing.T) {
	proxy := &keyRecoveryProxy{updateStatus: http.StatusNotFound, updateBody: keyNotFoundBody, staleKeyGone: true}
	d := newKeyUpdateResourceData(t, "gone-token", "team-a", "team-a")

	if diags := runKeyUpdate(t, proxy, d); !diags.HasError() {
		t.Fatal("expected an error when team_id did not change")
	}
	if got := atomic.LoadInt32(&proxy.generateCalls); got != 0 {
		t.Errorf("expected no recreate when team_id is unchanged, got %d /key/generate calls", got)
	}
	if d.Id() != "gone-token" {
		t.Errorf("Id = %q, want gone-token untouched on a hard failure", d.Id())
	}
}

// A transient failure must never be mistaken for a cascade-deleted key.
func TestResourceKeyUpdateServerErrorDoesNotRecreate(t *testing.T) {
	proxy := &keyRecoveryProxy{
		updateStatus: http.StatusInternalServerError,
		updateBody:   `{"error":{"message":"Internal Server Error"}}`,
		staleKeyGone: true,
	}
	d := newKeyUpdateResourceData(t, "hash-1", "team-a", "team-b")

	if diags := runKeyUpdate(t, proxy, d); !diags.HasError() {
		t.Fatal("expected a 500 to surface as an error")
	}
	if got := atomic.LoadInt32(&proxy.generateCalls); got != 0 {
		t.Errorf("expected no recreate for a transient error, got %d /key/generate calls", got)
	}
}

// The metadata pre-read fails before /key/update is ever reached when the key
// is gone, so that path needs the same recovery.
func TestResourceKeyUpdateRecreatesCascadeDeletedKeyWithMetadataChange(t *testing.T) {
	proxy := &keyRecoveryProxy{updateStatus: http.StatusOK, updateBody: `{"key": "hash-1"}`, staleKeyGone: true}
	state := &terraform.InstanceState{ID: "stale-token", Attributes: map[string]string{
		"team_id":       "team-a",
		"metadata.%":    "1",
		"metadata.tier": "gold",
	}}
	diff := &terraform.InstanceDiff{Attributes: map[string]*terraform.ResourceAttrDiff{
		"team_id":       {Old: "team-a", New: "team-b"},
		"metadata.tier": {Old: "gold", New: "silver"},
	}}
	d, err := schema.InternalMap(resourceKey().Schema).Data(state, diff)
	if err != nil {
		t.Fatalf("building ResourceData returned error: %v", err)
	}

	if diags := runKeyUpdate(t, proxy, d); diags.HasError() {
		t.Fatalf("a cascade-deleted key must be recreated, not error: %v", diags)
	}
	if got := atomic.LoadInt32(&proxy.updateCalls); got != 0 {
		t.Errorf("expected the metadata pre-read to short-circuit /key/update, got %d calls", got)
	}
	if got := atomic.LoadInt32(&proxy.generateCalls); got != 1 {
		t.Errorf("expected exactly 1 /key/generate recreate, got %d", got)
	}
	if d.Id() != "new-token" {
		t.Errorf("Id = %q, want the recreated key's new-token", d.Id())
	}
}
