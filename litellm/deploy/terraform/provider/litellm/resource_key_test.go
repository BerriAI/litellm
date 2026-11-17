package litellm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/hashicorp/go-cty/cty"
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
		"key_type":                   "llm_api",
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

	if key.KeyType != "llm_api" {
		t.Errorf("KeyType = %q, want llm_api", key.KeyType)
	}
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
		"key_type":                   "llm_api",
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
	if key.KeyType != "llm_api" {
		t.Errorf("KeyType = %q, want llm_api", key.KeyType)
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
	d := newKeyResourceData(t, map[string]interface{}{
		"key":      "sk-custom",
		"key_type": "llm_api",
	})

	diags := resourceKeyCreate(context.Background(), d, client)
	if diags.HasError() {
		t.Fatalf("create returned error: %v", diags)
	}
	if captured["key"] != "sk-custom" {
		t.Errorf("create payload key = %v, want sk-custom", captured["key"])
	}
	if captured["key_type"] != "llm_api" {
		t.Errorf("create payload key_type = %v, want llm_api", captured["key_type"])
	}
	if d.Id() != "hash-1" {
		t.Errorf("resource ID = %q, want hash-1", d.Id())
	}
}

func TestKeyTypeRejectsUnknownValue(t *testing.T) {
	_, errs := resourceKey().Schema["key_type"].ValidateFunc("unrestricted", "key_type")
	if len(errs) == 0 {
		t.Fatal("key_type accepted an unknown value")
	}
}

func TestKeyTypeChangeForcesReplacement(t *testing.T) {
	res := resourceKey()
	priorData := newKeyResourceData(t, map[string]interface{}{"key_type": "default"})
	priorData.SetId("hash-1")
	config := terraform.NewResourceConfigRaw(map[string]interface{}{"key_type": "llm_api"})
	diff, err := res.Diff(context.Background(), priorData.State(), config, nil)
	if err != nil {
		t.Fatalf("diff failed: %v", err)
	}
	if diff == nil || !diff.RequiresNew() {
		t.Fatalf("changing key_type must force replacement, diff = %+v", diff)
	}
}

func TestKeyTypePresetRoutesDoNotDrift(t *testing.T) {
	cases := map[string]struct {
		read   *Key
		config map[string]interface{}
	}{
		"llm_api preset":    {read: &Key{KeyType: "llm_api", AllowedRoutes: []string{"llm_api_routes"}}, config: map[string]interface{}{"key_type": "llm_api"}},
		"default no routes": {read: &Key{KeyType: "default"}, config: map[string]interface{}{}},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			res := resourceKey()
			priorData := newKeyResourceData(t, map[string]interface{}{})
			priorData.SetId("hash-1")
			if err := priorData.Set("server_metadata", serverKeyMetadata(tc.read.Metadata)); err != nil {
				t.Fatalf("set server_metadata: %v", err)
			}
			mapKeyToResourceData(priorData, tc.read)
			diff, err := res.Diff(context.Background(), priorData.State(), terraform.NewResourceConfigRaw(tc.config), nil)
			if err != nil {
				t.Fatalf("diff failed: %v", err)
			}
			if diff != nil && !diff.Empty() {
				t.Fatalf("server-derived allowed_routes must not drift, diff = %+v", diff)
			}
		})
	}
}

// A config that omits allowed_routes must stay KNOWN at plan time: marking it
// computed makes it "known after apply", which fails any plan that consumes
// the attribute (e.g. for_each = toset(coalesce(..., []))) before the key
// exists.
func TestAllowedRoutesOmittedIsKnownAtPlan(t *testing.T) {
	res := resourceKey()
	prior := &terraform.InstanceState{}
	prior.RawConfig = keyRawConfig(t, nil)
	config := terraform.NewResourceConfigRaw(map[string]interface{}{"key_alias": "example"})
	diff, err := res.Diff(context.Background(), prior, config, nil)
	if err != nil {
		t.Fatalf("diff failed: %v", err)
	}
	for k, attr := range diff.Attributes {
		if !strings.HasPrefix(k, "allowed_routes") {
			continue
		}
		if attr.NewComputed {
			t.Fatalf("omitted allowed_routes is computed (unknown) at plan time: %+v", attr)
		}
		t.Errorf("omitted allowed_routes produced a plan diff %q = %+v, want none", k, attr)
	}
}

// Suppressing unconfigured routes must not swallow real config changes: a
// config that shrinks the declared list still has to diff.
func TestAllowedRoutesConfiguredShrinkStillDiffs(t *testing.T) {
	res := resourceKey()
	priorData := newKeyResourceData(t, map[string]interface{}{
		"allowed_routes": []interface{}{"/a", "/b", "/c"},
	})
	priorData.SetId("hash-1")
	prior := priorData.State()
	prior.RawConfig = keyRawConfig(t, []string{"/a", "/b"})
	config := terraform.NewResourceConfigRaw(map[string]interface{}{
		"allowed_routes": []interface{}{"/a", "/b"},
	})
	diff, err := res.Diff(context.Background(), prior, config, nil)
	if err != nil {
		t.Fatalf("diff failed: %v", err)
	}
	if diff == nil || diff.Attributes["allowed_routes.#"] == nil {
		t.Fatalf("config shrinking allowed_routes must still diff, diff = %+v", diff)
	}
}

// The diff for server-derived routes in state is suppressed via the raw
// config, the same signal real terraform runs carry.
func TestAllowedRoutesUnconfiguredDoesNotDriftWithRawConfig(t *testing.T) {
	res := resourceKey()
	priorData := newKeyResourceData(t, map[string]interface{}{
		"key_alias":      "old",
		"allowed_routes": []interface{}{"/v1/models"},
	})
	priorData.SetId("hash-1")
	prior := priorData.State()
	prior.RawConfig = keyRawConfig(t, nil)
	config := terraform.NewResourceConfigRaw(map[string]interface{}{"key_alias": "old"})
	diff, err := res.Diff(context.Background(), prior, config, nil)
	if err != nil {
		t.Fatalf("diff failed: %v", err)
	}
	for k := range diff.Attributes {
		if strings.HasPrefix(k, "allowed_routes") {
			t.Fatalf("unconfigured allowed_routes drifted at plan: %q = %+v", k, diff.Attributes[k])
		}
	}
}

func keyRawConfig(t *testing.T, routes []string) cty.Value {
	t.Helper()
	return cty.ObjectVal(map[string]cty.Value{
		"key_alias":      cty.StringVal("example"),
		"allowed_routes": keyRawConfigRoutes(t, routes),
	})
}

// An update must never POST allowed_routes the config does not declare: the
// value d.Get returns is whatever the last refresh stored (stale with
// -refresh=false), and /key/update would overwrite externally managed routes
// with it. The alias-only rename below must leave the field out.
func TestResourceKeyUpdateOmitsUnconfiguredAllowedRoutes(t *testing.T) {
	var captured map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &captured)
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/key/update" {
			w.Write([]byte(`{"key": "hash-1"}`))
			return
		}
		w.Write([]byte(`{"key":"hash-1","info":{"key_alias":"renamed","allowed_routes":["/v1/models"]}}`))
	}))
	defer srv.Close()

	res := resourceKey()
	priorData := newKeyResourceData(t, map[string]interface{}{
		"key_alias":      "old",
		"allowed_routes": []interface{}{"/chat/completions"},
	})
	priorData.SetId("hash-1")
	prior := priorData.State()
	config := terraform.NewResourceConfigRaw(map[string]interface{}{"key_alias": "renamed"})
	diff, err := res.Diff(context.Background(), prior, config, nil)
	if err != nil {
		t.Fatalf("diff failed: %v", err)
	}
	diff.RawConfig = keyRawConfig(t, nil)

	_, diags := res.Apply(context.Background(), prior, diff, NewClient(srv.URL, "test-key", true))
	if diags.HasError() {
		t.Fatalf("apply failed: %+v", diags)
	}
	if _, present := captured["allowed_routes"]; present {
		t.Fatalf("update POSTed allowed_routes %v although the config does not declare it", captured["allowed_routes"])
	}
	if captured["key_alias"] != "renamed" {
		t.Errorf("update payload key_alias = %v, want renamed", captured["key_alias"])
	}
}

// Declaring allowed_routes keeps owning them: an update still re-asserts the
// configured routes, matching the pre-key_type behavior.
func TestResourceKeyUpdateSendsConfiguredAllowedRoutes(t *testing.T) {
	var captured map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &captured)
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/key/update" {
			w.Write([]byte(`{"key": "hash-1"}`))
			return
		}
		w.Write([]byte(`{"key":"hash-1","info":{"key_alias":"renamed","allowed_routes":["/v1/models"]}}`))
	}))
	defer srv.Close()

	res := resourceKey()
	priorData := newKeyResourceData(t, map[string]interface{}{
		"key_alias":      "old",
		"allowed_routes": []interface{}{"/v1/models"},
	})
	priorData.SetId("hash-1")
	prior := priorData.State()
	config := terraform.NewResourceConfigRaw(map[string]interface{}{
		"key_alias":      "renamed",
		"allowed_routes": []interface{}{"/v1/models"},
	})
	diff, err := res.Diff(context.Background(), prior, config, nil)
	if err != nil {
		t.Fatalf("diff failed: %v", err)
	}
	diff.RawConfig = keyRawConfig(t, []string{"/v1/models"})

	_, diags := res.Apply(context.Background(), prior, diff, NewClient(srv.URL, "test-key", true))
	if diags.HasError() {
		t.Fatalf("apply failed: %+v", diags)
	}
	routes, ok := captured["allowed_routes"].([]interface{})
	if !ok || len(routes) != 1 || routes[0] != "/v1/models" {
		t.Fatalf("update payload allowed_routes = %v, want [/v1/models]", captured["allowed_routes"])
	}
}

// /key/generate replaces declared routes with the key_type preset while
// /key/update stores them verbatim, so create must re-assert the declared
// list; the restore body may only touch allowed_routes plus the two fields
// /key/update requires non-null.
func TestCreateKeyRestoresDeclaredRoutesOverPreset(t *testing.T) {
	cases := map[string]struct {
		generateRoutes []interface{}
		suppliedKey    string
		permissions    map[string]interface{}
		generateStores bool
		generateBudget map[string]interface{}
		restoreFails   bool
		wantUpdate     bool
	}{
		"preset overwrote declared":        {generateRoutes: []interface{}{"llm_api_routes"}, wantUpdate: true},
		"declared permissions are echoed":  {generateRoutes: []interface{}{"llm_api_routes"}, permissions: map[string]interface{}{"get_server_info": "true"}, generateStores: true, wantUpdate: true},
		"declared budgets are echoed":      {generateRoutes: []interface{}{"llm_api_routes"}, generateBudget: map[string]interface{}{"x": true}, wantUpdate: true},
		"generate honored declared":        {generateRoutes: []interface{}{"/v1/models"}, wantUpdate: false},
		"restore update rejected":          {generateRoutes: []interface{}{"llm_api_routes"}, restoreFails: true, wantUpdate: true},
		"rejected restore keeps supplied":  {generateRoutes: []interface{}{"llm_api_routes"}, suppliedKey: "sk-custom", restoreFails: true, wantUpdate: true},
		"supplied key still gets restored": {generateRoutes: []interface{}{"llm_api_routes"}, suppliedKey: "sk-custom", wantUpdate: true},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			var generateBody, updateBody, deleteBody map[string]interface{}
			updateCalled, deleteCalled := false, false
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, _ := io.ReadAll(r.Body)
				w.Header().Set("Content-Type", "application/json")
				switch r.URL.Path {
				case "/key/generate":
					json.Unmarshal(body, &generateBody)
					extra := ""
					if tc.generateStores {
						extra = `, "permissions": {"get_server_info": true}`
					}
					if tc.generateBudget != nil {
						extra += `, "model_max_budget": {"gpt-4o-mini": {"budget_limit": 5, "time_period": "30d"}}`
					}
					w.Write([]byte(`{"key": "sk-new", "token_id": "hash-1", "allowed_routes": ["` + tc.generateRoutes[0].(string) + `"]` + extra + `}`))
				case "/key/update":
					updateCalled = true
					json.Unmarshal(body, &updateBody)
					if tc.restoreFails {
						w.WriteHeader(http.StatusBadRequest)
						w.Write([]byte(`{"error":{"message":"rejected"}}`))
						return
					}
					w.Write([]byte(`{"key": "hash-1"}`))
				case "/key/delete":
					deleteCalled = true
					json.Unmarshal(body, &deleteBody)
					w.Write([]byte(`{}`))
				default:
					w.Write([]byte(`{"key":"hash-1","info":{"key_alias":"typed","key_type":"llm_api","allowed_routes":["/v1/models"]}}`))
				}
			}))
			defer srv.Close()

			raw := map[string]interface{}{
				"key_alias":      "typed",
				"key_type":       "llm_api",
				"allowed_routes": []interface{}{"/v1/models"},
			}
			if tc.permissions != nil {
				raw["permissions"] = tc.permissions
			}
			if tc.generateBudget != nil {
				raw["model_max_budget"] = `{"gpt-4o-mini": {"budget_limit": 5, "time_period": "30d"}}`
			}
			if tc.suppliedKey != "" {
				raw["key"] = tc.suppliedKey
			}
			d := newKeyResourceData(t, raw)
			diags := resourceKeyCreate(context.Background(), d, NewClient(srv.URL, "test-key", true))
			if tc.restoreFails {
				if !diags.HasError() {
					t.Fatal("create succeeded although the restore update was rejected")
				}
				if tc.suppliedKey != "" {
					if deleteCalled {
						t.Fatal("failed restore deleted a config-supplied key, which /key/generate may have upserted onto an existing credential")
					}
					return
				}
				if !deleteCalled {
					t.Fatal("failed restore did not delete the created key; a retried apply would orphan it")
				}
				if keys, _ := deleteBody["keys"].([]interface{}); len(keys) != 1 || keys[0] != "hash-1" {
					t.Errorf("delete payload keys = %v, want [hash-1]", deleteBody["keys"])
				}
				if d.Id() != "" {
					t.Errorf("state recorded id %q for a key the provider deleted", d.Id())
				}
				return
			}
			if diags.HasError() {
				t.Fatalf("create failed: %+v", diags)
			}
			if generateBody["key_type"] != "llm_api" {
				t.Errorf("generate payload key_type = %v, want llm_api", generateBody["key_type"])
			}
			if tc.generateBudget != nil {
				want := map[string]interface{}{"gpt-4o-mini": map[string]interface{}{"budget_limit": float64(5), "time_period": "30d"}}
				if fmt.Sprint(generateBody["model_max_budget"]) != fmt.Sprint(want) {
					t.Errorf("generate payload model_max_budget = %v, want the declared %v", generateBody["model_max_budget"], want)
				}
			}
			if tc.wantUpdate {
				if !updateCalled {
					t.Fatal("create did not re-assert declared allowed_routes after the preset overwrote them")
				}
				wantPerms := map[string]interface{}{}
				if tc.generateStores {
					wantPerms = map[string]interface{}{"get_server_info": true}
				}
				wantBudget := map[string]interface{}{}
				if tc.generateBudget != nil {
					wantBudget = map[string]interface{}{"gpt-4o-mini": map[string]interface{}{"budget_limit": float64(5), "time_period": "30d"}}
				}
				wantBody := map[string]interface{}{
					"key":              "hash-1",
					"allowed_routes":   []interface{}{"/v1/models"},
					"permissions":      wantPerms,
					"model_max_budget": wantBudget,
				}
				if len(updateBody) != len(wantBody) {
					t.Fatalf("restore payload = %v, want exactly %v (anything else rewrites fields the config did not declare)", updateBody, wantBody)
				}
				for k, v := range wantBody {
					if fmt.Sprint(updateBody[k]) != fmt.Sprint(v) {
						t.Errorf("restore payload %s = %v (%T), want %v (%T)", k, updateBody[k], updateBody[k], v, v)
					}
				}
			} else if updateCalled {
				t.Fatal("create re-asserted routes although the generate already stored the declared list")
			}
			if deleteCalled {
				t.Fatal("create deleted a key although nothing failed")
			}
			if got := d.Get("allowed_routes").([]interface{}); len(got) != 1 || got[0] != "/v1/models" {
				t.Errorf("state allowed_routes = %v, want [/v1/models]", got)
			}
		})
	}
}

func keyRawConfigRoutes(t *testing.T, routes []string) cty.Value {
	t.Helper()
	if routes == nil {
		return cty.NullVal(cty.List(cty.String))
	}
	vals := make([]cty.Value, len(routes))
	for i, r := range routes {
		vals[i] = cty.StringVal(r)
	}
	return cty.ListVal(vals)
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
				"key_type": "llm_api",
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
	if key.KeyType != "llm_api" {
		t.Errorf("KeyType = %q, want llm_api", key.KeyType)
	}
	if key.BudgetID != "budget-1" || key.TeamID != "team-1" {
		t.Errorf("nested fields not parsed: %+v", key)
	}
	if key.RPMLimit == nil || *key.RPMLimit != 100 {
		t.Errorf("RPMLimit not parsed: %+v", key.RPMLimit)
	}
}

func TestGetKeyReturnsNilForDeletedStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"key":"hash-1","info":{"token":"hash-1","key_alias":"gone","status":"deleted","deleted_at":"2026-10-05T00:00:00Z","deleted_by":"admin"}}`))
	}))
	defer srv.Close()

	key, err := NewClient(srv.URL, "test-key", true).GetKey("hash-1")
	if err != nil {
		t.Fatalf("GetKey returned error: %v", err)
	}
	if key != nil {
		t.Errorf("GetKey returned %+v, want nil for deleted status", key)
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

func TestResourceKeyReadDropsDeletedKeyFromState(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"key":"hash-1","info":{"token":"hash-1","key_alias":"gone","status":"deleted","deleted_at":"2026-10-05T00:00:00Z","deleted_by":"admin"}}`))
	}))
	defer srv.Close()

	d := newKeyResourceData(t, map[string]interface{}{"key_alias": "gone"})
	d.SetId("hash-1")

	diags := resourceKeyRead(context.Background(), d, NewClient(srv.URL, "test-key", true))
	if diags.HasError() {
		t.Fatalf("read of a deleted key must not error, got: %v", diags)
	}
	if d.Id() != "" {
		t.Errorf("Id = %q, want empty so Terraform plans a recreate", d.Id())
	}
}

func TestResourceKeyReadKeepsKeyWithLiveStatus(t *testing.T) {
	for _, status := range []string{"active", "revoked", "expired"} {
		t.Run(status, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				fmt.Fprintf(w, `{"key":"hash-1","info":{"token":"hash-1","status":%q}}`, status)
			}))
			defer srv.Close()

			d := newKeyResourceData(t, map[string]interface{}{})
			d.SetId("hash-1")

			diags := resourceKeyRead(context.Background(), d, NewClient(srv.URL, "test-key", true))
			if diags.HasError() {
				t.Fatalf("read of a %s key returned error: %v", status, diags)
			}
			if d.Id() != "hash-1" {
				t.Errorf("Id = %q, want unchanged for %s status", d.Id(), status)
			}
		})
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

func TestResourceKeyDeleteTreatsAlreadyDeletedKeyAsGone(t *testing.T) {
	for _, infoStatus := range []int{http.StatusOK, http.StatusNotFound} {
		t.Run(http.StatusText(infoStatus), func(t *testing.T) {
			infoCalls := 0
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				switch r.URL.Path {
				case "/key/delete":
					w.WriteHeader(http.StatusNotFound)
					w.Write([]byte(`{"error":{"message":"{'error': 'No keys found'}","type":"internal_server_error","param":null,"code":"404"}}`))
				case "/key/info":
					infoCalls++
					if infoStatus == http.StatusNotFound {
						w.WriteHeader(http.StatusNotFound)
						w.Write([]byte(keyNotFoundBody))
						return
					}
					w.Write([]byte(`{"key":"hash-1","info":{"token":"hash-1","status":"deleted"}}`))
				default:
					http.NotFound(w, r)
				}
			}))
			defer srv.Close()

			d := newKeyResourceData(t, map[string]interface{}{})
			d.SetId("hash-1")

			diags := resourceKeyDelete(context.Background(), d, NewClient(srv.URL, "test-key", true))
			if diags.HasError() {
				t.Fatalf("deleting an already deleted key must not error, got: %v", diags)
			}
			if d.Id() != "" {
				t.Errorf("Id = %q, want empty after confirming the key is gone", d.Id())
			}
			if infoCalls != 1 {
				t.Errorf("expected one /key/info confirmation, got %d calls", infoCalls)
			}
		})
	}
}

func TestResourceKeyDelete404WithLiveKeyFails(t *testing.T) {
	infoCalls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/key/delete":
			w.WriteHeader(http.StatusNotFound)
			w.Write([]byte(`{"error":{"message":"{'error': 'No keys found'}","type":"internal_server_error","param":null,"code":"404"}}`))
		case "/key/info":
			infoCalls++
			w.Write([]byte(`{"key":"hash-1","info":{"token":"hash-1","status":"active"}}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	d := newKeyResourceData(t, map[string]interface{}{})
	d.SetId("hash-1")

	diags := resourceKeyDelete(context.Background(), d, NewClient(srv.URL, "test-key", true))
	if !diags.HasError() {
		t.Fatal("a /key/delete 404 with a live key must remain an error")
	}
	if d.Id() != "hash-1" {
		t.Errorf("Id = %q, want unchanged when /key/info confirms a live key", d.Id())
	}
	if infoCalls != 1 {
		t.Errorf("expected one /key/info confirmation, got %d calls", infoCalls)
	}
}

func TestResourceKeyDeleteServerErrorFailsWithoutInfoLookup(t *testing.T) {
	infoCalls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/key/delete" {
			w.WriteHeader(http.StatusInternalServerError)
			w.Write([]byte(`{"error":{"message":"server error"}}`))
			return
		}
		if r.URL.Path == "/key/info" {
			infoCalls++
		}
	}))
	defer srv.Close()

	d := newKeyResourceData(t, map[string]interface{}{})
	d.SetId("hash-1")

	diags := resourceKeyDelete(context.Background(), d, NewClient(srv.URL, "test-key", true))
	if !diags.HasError() {
		t.Fatal("a /key/delete 500 must remain an error")
	}
	if d.Id() != "hash-1" {
		t.Errorf("Id = %q, want unchanged after a server error", d.Id())
	}
	if infoCalls != 0 {
		t.Errorf("expected no /key/info lookup after a /key/delete 500, got %d calls", infoCalls)
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
	if got := newState.Attributes["server_metadata.server_side"]; got != "x" {
		t.Errorf("server_metadata.server_side = %q, want x", got)
	}
	if _, present := proxy.updates[0]["server_metadata"]; present {
		t.Errorf("computed server_metadata was sent on /key/update: %v", proxy.updates[0]["server_metadata"])
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

func TestKeyReadExposesUndeclaredMetadataInServerMetadata(t *testing.T) {
	proxy := &fakeKeyProxy{metadata: map[string]interface{}{
		"a":               "1",
		"server_side":     "x",
		"model_rpm_limit": map[string]interface{}{"gpt-4o-mini": float64(5)},
		"nested":          map[string]interface{}{"k": "v"},
	}}
	srv := httptest.NewServer(proxy.handler())
	defer srv.Close()
	client := NewClient(srv.URL, "test-key", true)

	d := newKeyResourceData(t, map[string]interface{}{"metadata": map[string]interface{}{"a": "1"}})
	d.SetId("hash-1")
	if diags := resourceKeyRead(context.Background(), d, client); diags.HasError() {
		t.Fatalf("Read returned error: %v", diags)
	}

	if got, want := d.Get("metadata"), map[string]interface{}{"a": "1"}; !reflect.DeepEqual(got, want) {
		t.Errorf("metadata in state = %v, want %v", got, want)
	}
	want := map[string]interface{}{"a": "1", "server_side": "x", "nested": `{"k":"v"}`}
	if got := d.Get("server_metadata"); !reflect.DeepEqual(got, want) {
		t.Errorf("server_metadata in state = %v, want %v", got, want)
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
	updateStatus    int
	updateBody      string
	staleKeyGone    bool
	staleKeyDeleted bool
	updateCalls     int32
	generateCalls   int32
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
				if p.staleKeyDeleted {
					io.WriteString(w, `{"key":"stale-token","info":{"token":"stale-token","status":"deleted","deleted_at":"2026-10-05T00:00:00Z","deleted_by":"admin"}}`)
					return
				}
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

func TestResourceKeyUpdateRecreatesCascadeDeletedKeyWithDeletedStatus(t *testing.T) {
	proxy := &keyRecoveryProxy{
		updateStatus:    http.StatusNotFound,
		updateBody:      keyNotFoundBody,
		staleKeyGone:    true,
		staleKeyDeleted: true,
	}
	d := newKeyUpdateResourceData(t, "stale-token", "team-a", "team-b")

	if diags := runKeyUpdate(t, proxy, d); diags.HasError() {
		t.Fatalf("a cascade-deleted key must be recreated, not error: %v", diags)
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
