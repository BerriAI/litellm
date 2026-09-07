package litellm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

var s04ModelInfo = map[string]interface{}{
	"team_id":                   "team-terraform-litellm",
	"max_input_tokens":          200000,
	"max_output_tokens":         32000,
	"input_modalities":          []interface{}{"text"},
	"output_modalities":         []interface{}{"text"},
	"supports_reasoning":        true,
	"supports_function_calling": true,
}

func s04JSONEqual(got, want interface{}) bool {
	if wantInt, ok := want.(int); ok {
		gotFloat, ok := got.(float64)
		return ok && gotFloat == float64(wantInt)
	}
	return reflect.DeepEqual(got, want)
}

func assertS04ModelInfoState(t *testing.T, d *schema.ResourceData, label string) {
	t.Helper()
	for name, want := range s04ModelInfo {
		if got := d.Get(name); !reflect.DeepEqual(got, want) {
			t.Errorf("%s %s = %#v, want %#v", label, name, got, want)
		}
	}
}

func s04WrappedModelResponse(response map[string]interface{}, id string) map[string]interface{} {
	wrappedModel := make(map[string]interface{}, len(response))
	for key, value := range response {
		wrappedModel[key] = value
	}
	modelInfo, _ := response["model_info"].(map[string]interface{})
	wrappedInfo := make(map[string]interface{}, len(modelInfo)+1)
	for key, value := range modelInfo {
		wrappedInfo[key] = value
	}
	wrappedInfo["id"] = id
	wrappedModel["model_info"] = wrappedInfo
	return map[string]interface{}{"data": []interface{}{wrappedModel}}
}

func requireS04ModelSchema(t *testing.T) *schema.Resource {
	t.Helper()
	resource := resourceLiteLLMModel()
	want := map[string]schema.ValueType{
		"team_id":                   schema.TypeString,
		"max_input_tokens":          schema.TypeInt,
		"max_output_tokens":         schema.TypeInt,
		"input_modalities":          schema.TypeList,
		"output_modalities":         schema.TypeList,
		"supports_reasoning":        schema.TypeBool,
		"supports_function_calling": schema.TypeBool,
	}
	for name, valueType := range want {
		field, ok := resource.Schema[name]
		if !ok {
			t.Errorf("S04 RED: model schema missing %q", name)
			continue
		}
		if field.Type != valueType {
			t.Errorf("S04 RED: model schema %q type = %v, want %v", name, field.Type, valueType)
		}
	}
	if field := resource.Schema["team_id"]; field != nil && (!field.Optional || !field.Computed || field.Required) {
		t.Error("team_id must remain Optional+Computed for upstream compatibility; the homelab module enforces ownership")
	}
	return resource
}

func TestS04ModelImporterPassesThroughStableID(t *testing.T) {
	resource := resourceLiteLLMModel()
	if resource.Importer == nil || resource.Importer.StateContext == nil {
		t.Fatal("S04 RED: model resource has no importer")
	}
	d := schema.TestResourceDataRaw(t, resource.Schema, map[string]interface{}{})
	d.SetId("model-tf-canary-big-pickle")
	states, err := resource.Importer.StateContext(context.Background(), d, nil)
	if err != nil {
		t.Fatalf("import stable model ID: %v", err)
	}
	if len(states) != 1 || states[0].Id() != "model-tf-canary-big-pickle" {
		t.Fatalf("import did not preserve model ID: %#v", states)
	}
}

func TestS04ModelCreateSerializesTeamAndModelInfo(t *testing.T) {
	resource := requireS04ModelSchema(t)
	for name := range s04ModelInfo {
		if resource.Schema[name] == nil {
			t.Skip("remaining request assertions require the RED schema fields")
		}
	}

	var request map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == endpointModelNew {
			if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
				t.Fatalf("decode model request: %v", err)
			}
			_ = json.NewEncoder(w).Encode(request)
			return
		}
		_ = json.NewEncoder(w).Encode(s04WrappedModelResponse(request, r.URL.Query().Get("modelId")))
	}))
	defer srv.Close()

	raw := map[string]interface{}{
		"model_name": "tf-canary-big-pickle", "custom_llm_provider": "openai",
		"base_model": "big-pickle", "mode": "chat",
	}
	for name, value := range s04ModelInfo {
		raw[name] = value
	}
	d := schema.TestResourceDataRaw(t, resource.Schema, raw)
	if err := createOrUpdateModel(d, NewClient(srv.URL, "test-key", false), false); err != nil {
		t.Fatalf("create model: %v", err)
	}
	got, _ := request["model_info"].(map[string]interface{})
	for name, want := range s04ModelInfo {
		if !s04JSONEqual(got[name], want) {
			t.Errorf("model_info.%s = %#v, want %#v", name, got[name], want)
		}
	}
}

func TestS04ModelUpdateSerializesFalseCapabilities(t *testing.T) {
	resource := requireS04ModelSchema(t)
	var request map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/model/model-tf-canary-big-pickle/update" && r.Method == http.MethodPatch {
			if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
				t.Fatalf("decode model update: %v", err)
			}
			_ = json.NewEncoder(w).Encode(request)
			return
		}
		_ = json.NewEncoder(w).Encode(s04WrappedModelResponse(request, r.URL.Query().Get("modelId")))
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resource.Schema, map[string]interface{}{
		"model_name": "tf-canary-big-pickle", "custom_llm_provider": "openai",
		"base_model": "big-pickle", "mode": "chat", "team_id": "team-terraform-litellm",
		"supports_reasoning": false, "supports_function_calling": false,
		"max_input_tokens": 0, "max_output_tokens": 0,
	})
	d.SetId("model-tf-canary-big-pickle")
	if err := createOrUpdateModel(d, NewClient(srv.URL, "test-key", false), true); err != nil {
		t.Fatalf("update model: %v", err)
	}
	got, _ := request["model_info"].(map[string]interface{})
	for _, name := range []string{"supports_reasoning", "supports_function_calling"} {
		value, exists := got[name]
		if !exists || value != false {
			t.Errorf("model_info.%s = %#v (exists %t), want explicit false", name, value, exists)
		}
	}
	for _, name := range []string{"max_input_tokens", "max_output_tokens"} {
		if value, exists := got[name]; !exists || value != float64(0) {
			t.Errorf("model_info.%s = %#v (exists %t), want explicit zero", name, value, exists)
		}
	}
	for _, name := range []string{"input_modalities", "output_modalities"} {
		if value, exists := got[name]; exists {
			t.Errorf("model_info.%s unexpectedly serialized as %#v; empty preserves remote metadata", name, value)
		}
	}
}

func TestS04RoutingBaseStripsOneExactProviderPrefix(t *testing.T) {
	if got := modelRoutingBase("openai", "openai/vendor/big-pickle"); got != "vendor/big-pickle" {
		t.Fatalf("routing base = %q, want vendor/big-pickle", got)
	}
	if got := modelRoutingBase("openai", "other/openai/big-pickle"); got != "other/openai/big-pickle" {
		t.Fatalf("nonmatching routing base mangled to %q", got)
	}
}

func TestS04LegacyModelUpdateOmitsUnconfiguredMetadata(t *testing.T) {
	resource := requireS04ModelSchema(t)
	var request map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/model/legacy-model-id/update" && r.Method == http.MethodPatch {
			if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
				t.Fatalf("decode legacy model update: %v", err)
			}
			_ = json.NewEncoder(w).Encode(request)
			return
		}
		_ = json.NewEncoder(w).Encode(s04WrappedModelResponse(request, r.URL.Query().Get("modelId")))
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resource.Schema, map[string]interface{}{
		"model_name": "legacy-model", "custom_llm_provider": "openai",
		"base_model": "legacy", "mode": "chat",
	})
	d.SetId("legacy-model-id")
	if err := createOrUpdateModel(d, NewClient(srv.URL, "test-key", false), true); err != nil {
		t.Fatalf("update legacy model: %v", err)
	}
	got, _ := request["model_info"].(map[string]interface{})
	for _, name := range []string{
		"max_input_tokens", "max_output_tokens", "input_modalities", "output_modalities",
		"supports_reasoning", "supports_function_calling",
	} {
		if value, exists := got[name]; exists {
			t.Errorf("legacy model_info.%s unexpectedly serialized as %#v", name, value)
		}
	}
}

func TestS04ModelReadPreservesMetadataOmittedByAPI(t *testing.T) {
	resource := requireS04ModelSchema(t)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		response := map[string]interface{}{
			"model_name": "tf-canary-big-pickle",
			"litellm_params": map[string]interface{}{
				"custom_llm_provider": "openai", "model": "openai/big-pickle",
			},
			"model_info": map[string]interface{}{"team_id": "team-terraform-litellm"},
		}
		_ = json.NewEncoder(w).Encode(s04WrappedModelResponse(response, r.URL.Query().Get("modelId")))
	}))
	defer srv.Close()

	raw := map[string]interface{}{
		"model_name": "tf-canary-big-pickle", "custom_llm_provider": "openai",
		"base_model": "big-pickle", "mode": "chat",
	}
	for name, value := range s04ModelInfo {
		raw[name] = value
	}
	d := schema.TestResourceDataRaw(t, resource.Schema, raw)
	d.SetId("model-tf-canary-big-pickle")
	if err := resourceLiteLLMModelRead(d, NewClient(srv.URL, "test-key", false)); err != nil {
		t.Fatalf("read model with omitted metadata: %v", err)
	}
	assertS04ModelInfoState(t, d, "omitted API metadata")
}

func TestS04ModelReadClearsStalePricingBaseWhenAuthoritativeValueMatchesRouting(t *testing.T) {
	resource := requireS04ModelSchema(t)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		response := map[string]interface{}{
			"model_name": "tf-canary-big-pickle",
			"litellm_params": map[string]interface{}{
				"custom_llm_provider": "openai", "model": "openai/big-pickle",
			},
			"model_info": map[string]interface{}{"base_model": "big-pickle"},
		}
		_ = json.NewEncoder(w).Encode(s04WrappedModelResponse(response, r.URL.Query().Get("modelId")))
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resource.Schema, map[string]interface{}{
		"model_name": "tf-canary-big-pickle", "custom_llm_provider": "openai",
		"base_model": "big-pickle", "pricing_base_model": "stale/pricing-key", "mode": "chat",
	})
	d.SetId("model-tf-canary-big-pickle")
	if err := resourceLiteLLMModelRead(d, NewClient(srv.URL, "test-key", false)); err != nil {
		t.Fatalf("read model: %v", err)
	}
	if got := d.Get("pricing_base_model"); got != "" {
		t.Fatalf("pricing_base_model = %#v, want cleared", got)
	}
}

func TestS04ModelReadPreservesPricingBaseWhenAPIValueIsOmitted(t *testing.T) {
	resource := requireS04ModelSchema(t)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		response := map[string]interface{}{
			"model_name": "tf-canary-big-pickle",
			"litellm_params": map[string]interface{}{
				"custom_llm_provider": "openai", "model": "openai/big-pickle",
			},
			"model_info": map[string]interface{}{},
		}
		_ = json.NewEncoder(w).Encode(s04WrappedModelResponse(response, r.URL.Query().Get("modelId")))
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resource.Schema, map[string]interface{}{
		"model_name": "tf-canary-big-pickle", "custom_llm_provider": "openai",
		"base_model": "big-pickle", "pricing_base_model": "preserved/pricing-key", "mode": "chat",
	})
	d.SetId("model-tf-canary-big-pickle")
	if err := resourceLiteLLMModelRead(d, NewClient(srv.URL, "test-key", false)); err != nil {
		t.Fatalf("read model: %v", err)
	}
	if got := d.Get("pricing_base_model"); got != "preserved/pricing-key" {
		t.Fatalf("pricing_base_model = %#v, want preserved/pricing-key", got)
	}
}

func TestS04ModelReadBackRestoresMetadataWithoutDiff(t *testing.T) {
	resource := requireS04ModelSchema(t)
	for name := range s04ModelInfo {
		if resource.Schema[name] == nil {
			t.Skip("remaining read-back assertions require the RED schema fields")
		}
	}

	response := map[string]interface{}{
		"model_name": "tf-canary-big-pickle",
		"litellm_params": map[string]interface{}{
			"custom_llm_provider": "openai", "model": "openai/big-pickle",
		},
		"model_info": s04ModelInfo,
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(s04WrappedModelResponse(response, r.URL.Query().Get("modelId")))
	}))
	defer srv.Close()

	raw := map[string]interface{}{
		"model_name": "tf-canary-big-pickle", "custom_llm_provider": "openai",
		"base_model": "big-pickle", "mode": "chat",
	}
	for name, value := range s04ModelInfo {
		raw[name] = value
	}
	d := schema.TestResourceDataRaw(t, resource.Schema, raw)
	d.SetId("model-tf-canary-big-pickle")
	if err := resourceLiteLLMModelRead(d, NewClient(srv.URL, "test-key", false)); err != nil {
		t.Fatalf("read model: %v", err)
	}
	assertS04ModelInfoState(t, d, "read-back")
	if err := resourceLiteLLMModelRead(d, NewClient(srv.URL, "test-key", false)); err != nil {
		t.Fatalf("second read model: %v", err)
	}
	assertS04ModelInfoState(t, d, "second read-back")
}

func TestS04ImportedModelReadBackIsStableAcrossRepeatedReads(t *testing.T) {
	resource := requireS04ModelSchema(t)
	if resource.Importer == nil || resource.Importer.StateContext == nil {
		t.Fatal("S04 RED: imported model cannot be read because importer is missing")
	}
	for name := range s04ModelInfo {
		if resource.Schema[name] == nil {
			t.Skip("import/read zero-diff assertion requires the RED schema fields")
		}
	}

	modelInfo := make(map[string]interface{}, len(s04ModelInfo)+1)
	for name, value := range s04ModelInfo {
		modelInfo[name] = value
	}
	modelInfo["base_model"] = "openrouter/big-pickle-pricing"
	response := map[string]interface{}{
		"model_name": "tf-canary-big-pickle",
		"litellm_params": map[string]interface{}{
			"custom_llm_provider": "openai", "model": "openai/big-pickle",
		},
		"model_info": modelInfo,
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != endpointModelInfo || r.URL.Query().Get("modelId") != "model-tf-canary-big-pickle" {
			t.Errorf("import read request = %s %s?%s", r.Method, r.URL.Path, r.URL.RawQuery)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(s04WrappedModelResponse(response, "model-tf-canary-big-pickle"))
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resource.Schema, map[string]interface{}{})
	d.SetId("model-tf-canary-big-pickle")
	states, err := resource.Importer.StateContext(context.Background(), d, nil)
	if err != nil || len(states) != 1 {
		t.Fatalf("import model: states=%d err=%v", len(states), err)
	}
	if err := resourceLiteLLMModelRead(states[0], NewClient(srv.URL, "test-key", false)); err != nil {
		t.Fatalf("read imported model: %v", err)
	}
	assertS04ModelInfoState(t, states[0], "import/read")
	if got := states[0].Get("base_model"); got != "big-pickle" {
		t.Errorf("import/read base_model = %#v, want routing base big-pickle", got)
	}
	if got := states[0].Get("pricing_base_model"); got != "openrouter/big-pickle-pricing" {
		t.Errorf("import/read pricing_base_model = %#v, want distinct pricing key", got)
	}
	if err := resourceLiteLLMModelRead(states[0], NewClient(srv.URL, "test-key", false)); err != nil {
		t.Fatalf("second read imported model: %v", err)
	}
	assertS04ModelInfoState(t, states[0], "second import/read")
	if got := states[0].Get("base_model"); got != "big-pickle" {
		t.Errorf("second import/read base_model = %#v, want stable routing base", got)
	}
	if got := states[0].Get("pricing_base_model"); got != "openrouter/big-pickle-pricing" {
		t.Errorf("second import/read pricing_base_model = %#v, want stable pricing key", got)
	}
}

func TestS04ImportedModelEmptyV2DataClearsID(t *testing.T) {
	resource := requireS04ModelSchema(t)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != endpointModelInfo || r.URL.Query().Get("modelId") != "missing/model with space" {
			t.Errorf("not-found read request = %s?%s", r.URL.Path, r.URL.RawQuery)
		}
		w.Header().Set("Content-Type", "application/json")
		if r.URL.RawQuery != "modelId=missing%2Fmodel+with+space" {
			t.Errorf("model ID not URL encoded: %s", r.URL.RawQuery)
		}
		_ = json.NewEncoder(w).Encode(map[string]interface{}{"data": []interface{}{}})
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resource.Schema, map[string]interface{}{})
	d.SetId("missing/model with space")
	states, err := resource.Importer.StateContext(context.Background(), d, nil)
	if err != nil || len(states) != 1 {
		t.Fatalf("import missing model: states=%d err=%v", len(states), err)
	}
	if err := resourceLiteLLMModelRead(states[0], NewClient(srv.URL, "test-key", false)); err != nil {
		t.Fatalf("read missing imported model: %v", err)
	}
	if states[0].Id() != "" {
		t.Fatalf("missing imported model ID = %q, want cleared", states[0].Id())
	}
}

func TestS04ModelReadRejectsMismatchedAndMultipleV2Data(t *testing.T) {
	tests := map[string][]interface{}{
		"mismatched": {
			map[string]interface{}{"model_info": map[string]interface{}{"id": "other-model"}},
		},
		"multiple": {
			map[string]interface{}{"model_info": map[string]interface{}{"id": "expected-model"}},
			map[string]interface{}{"model_info": map[string]interface{}{"id": "other-model"}},
		},
	}
	for name, data := range tests {
		t.Run(name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(map[string]interface{}{"data": data})
			}))
			defer srv.Close()

			resource := resourceLiteLLMModel()
			d := schema.TestResourceDataRaw(t, resource.Schema, map[string]interface{}{})
			d.SetId("expected-model")
			if err := resourceLiteLLMModelRead(d, NewClient(srv.URL, "test-key", false)); err == nil {
				t.Fatal("read accepted ambiguous or mismatched v2 model data")
			}
			if d.Id() != "expected-model" {
				t.Fatalf("failed read changed model ID to %q", d.Id())
			}
		})
	}
}

func TestS04ModelReadDoesNotAcceptUnwrappedZeroStruct(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"model_name": "wrong-unwrapped-shape",
			"model_info": map[string]interface{}{"id": "expected-model"},
		})
	}))
	defer srv.Close()

	resource := resourceLiteLLMModel()
	d := schema.TestResourceDataRaw(t, resource.Schema, map[string]interface{}{})
	d.SetId("expected-model")
	if err := resourceLiteLLMModelRead(d, NewClient(srv.URL, "test-key", false)); err != nil {
		t.Fatalf("unwrapped response should be treated as not found, got: %v", err)
	}
	if d.Id() != "" {
		t.Fatalf("unwrapped response produced false success with ID %q", d.Id())
	}
}
