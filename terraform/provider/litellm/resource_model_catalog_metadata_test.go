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

var catalogModelInfo = map[string]interface{}{
	"supports_vision":          true,
	"input_cost_per_character": 0.00000125,
	"default_voice":            "alloy",
	"probe_language":           "pt-BR",
	"probe_text":               "Teste de saude",
	"probe_skip":               true,
	"max_tokens":               8192,
}

func requireCatalogModelSchema(t *testing.T) *schema.Resource {
	t.Helper()
	resource := resourceLiteLLMModel()
	want := map[string]schema.ValueType{
		"supports_vision":          schema.TypeBool,
		"input_cost_per_character": schema.TypeFloat,
		"default_voice":            schema.TypeString,
		"probe_language":           schema.TypeString,
		"probe_text":               schema.TypeString,
		"probe_skip":               schema.TypeBool,
		"max_tokens":               schema.TypeInt,
	}
	for name, valueType := range want {
		field, ok := resource.Schema[name]
		if !ok {
			t.Errorf("catalog RED: model schema missing %q", name)
			continue
		}
		if field.Type != valueType {
			t.Errorf("catalog RED: model schema %q type = %v, want %v", name, field.Type, valueType)
		}
		if !field.Optional || !field.Computed || field.Required {
			t.Errorf("catalog RED: model schema %q must be Optional+Computed and not required", name)
		}
	}
	return resource
}

func requireCatalogFields(t *testing.T, resource *schema.Resource) {
	t.Helper()
	for name := range catalogModelInfo {
		if resource.Schema[name] == nil {
			t.Skipf("catalog request/read assertions require RED schema field %q", name)
		}
	}
}

func assertCatalogModelInfoState(t *testing.T, d *schema.ResourceData, label string) {
	t.Helper()
	for name, want := range catalogModelInfo {
		if got := d.Get(name); !reflect.DeepEqual(got, want) {
			t.Errorf("%s %s = %#v, want %#v", label, name, got, want)
		}
	}
}

func TestCatalogModelSchemaExposesTypedModelInfo(t *testing.T) {
	requireCatalogModelSchema(t)
}

func TestCatalogModelCreateSerializesTypedModelInfo(t *testing.T) {
	resource := requireCatalogModelSchema(t)
	requireCatalogFields(t, resource)

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
		"model_name": "catalog-model", "custom_llm_provider": "openai",
		"base_model": "catalog-upstream", "mode": "chat",
	}
	for name, value := range catalogModelInfo {
		raw[name] = value
	}
	d := schema.TestResourceDataRaw(t, resource.Schema, raw)
	if err := createOrUpdateModel(d, NewClient(srv.URL, "test-key", false), false); err != nil {
		t.Fatalf("create model: %v", err)
	}
	got, _ := request["model_info"].(map[string]interface{})
	for name, want := range catalogModelInfo {
		if !s04JSONEqual(got[name], want) {
			t.Errorf("model_info.%s = %#v, want %#v", name, got[name], want)
		}
	}
}

func TestCatalogModelUpdateSerializesExplicitFalseAndZero(t *testing.T) {
	resource := requireCatalogModelSchema(t)
	requireCatalogFields(t, resource)

	var request map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/model/catalog-model-id/update" && r.Method == http.MethodPatch {
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
		"model_name": "catalog-model", "custom_llm_provider": "openai",
		"base_model": "catalog-upstream", "mode": "chat",
		"supports_vision": false, "probe_skip": false,
		"input_cost_per_character": 0.0, "max_tokens": 0,
	})
	d.SetId("catalog-model-id")
	if err := createOrUpdateModel(d, NewClient(srv.URL, "test-key", false), true); err != nil {
		t.Fatalf("update model: %v", err)
	}
	got, _ := request["model_info"].(map[string]interface{})
	for _, name := range []string{"supports_vision", "probe_skip"} {
		value, exists := got[name]
		if !exists || value != false {
			t.Errorf("model_info.%s = %#v (exists %t), want explicit false", name, value, exists)
		}
	}
	for _, name := range []string{"input_cost_per_character", "max_tokens"} {
		value, exists := got[name]
		if !exists || value != float64(0) {
			t.Errorf("model_info.%s = %#v (exists %t), want explicit zero", name, value, exists)
		}
	}
}

func TestCatalogModelReadPreservesTypedMetadataOmittedByAPI(t *testing.T) {
	resource := requireCatalogModelSchema(t)
	requireCatalogFields(t, resource)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		response := map[string]interface{}{
			"model_name": "catalog-model",
			"litellm_params": map[string]interface{}{
				"custom_llm_provider": "openai", "model": "openai/catalog-upstream",
			},
			"model_info": map[string]interface{}{},
		}
		_ = json.NewEncoder(w).Encode(s04WrappedModelResponse(response, r.URL.Query().Get("modelId")))
	}))
	defer srv.Close()

	raw := map[string]interface{}{
		"model_name": "catalog-model", "custom_llm_provider": "openai",
		"base_model": "catalog-upstream", "mode": "chat",
	}
	for name, value := range catalogModelInfo {
		raw[name] = value
	}
	d := schema.TestResourceDataRaw(t, resource.Schema, raw)
	d.SetId("catalog-model-id")
	if err := resourceLiteLLMModelRead(d, NewClient(srv.URL, "test-key", false)); err != nil {
		t.Fatalf("read model with omitted metadata: %v", err)
	}
	assertCatalogModelInfoState(t, d, "omitted API metadata")
	for _, field := range []string{"input_modalities.#", "output_modalities.#"} {
		if d.State().Attributes[field] != "0" {
			t.Fatalf("missing empty modality state for %s", field)
		}
	}
}

func TestCatalogModelReadAdoptsAPIPopulatedMetadataWithoutConfigDrift(t *testing.T) {
	resource := requireCatalogModelSchema(t)
	requireCatalogFields(t, resource)

	apiModelInfo := map[string]interface{}{
		"supports_vision":          false,
		"input_cost_per_character": 0.0,
		"default_voice":            "alloy",
		"probe_language":           "pt-BR",
		"probe_text":               "Teste de saude",
		"probe_skip":               false,
		"max_tokens":               204800,
	}
	response := map[string]interface{}{
		"model_name": "catalog-model",
		"litellm_params": map[string]interface{}{
			"custom_llm_provider": "openai", "model": "openai/catalog-upstream",
		},
		"model_info": apiModelInfo,
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(s04WrappedModelResponse(response, "catalog-model-id"))
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resource.Schema, map[string]interface{}{
		"model_name": "catalog-model", "custom_llm_provider": "openai",
		"base_model": "catalog-upstream", "mode": "chat",
	})
	d.SetId("catalog-model-id")
	for read := 1; read <= 2; read++ {
		if err := resourceLiteLLMModelRead(d, NewClient(srv.URL, "test-key", false)); err != nil {
			t.Fatalf("read %d model with API-populated metadata: %v", read, err)
		}
		for name, want := range apiModelInfo {
			if got := d.Get(name); !reflect.DeepEqual(got, want) {
				t.Errorf("read %d API-populated %s = %#v, want %#v", read, name, got, want)
			}
		}
	}
}

func TestCatalogImportedModelRestoresTypedMetadataAcrossRepeatedReads(t *testing.T) {
	resource := requireCatalogModelSchema(t)
	requireCatalogFields(t, resource)
	if resource.Importer == nil || resource.Importer.StateContext == nil {
		t.Fatal("catalog RED: model resource has no importer")
	}

	response := map[string]interface{}{
		"model_name": "catalog-model",
		"litellm_params": map[string]interface{}{
			"custom_llm_provider": "openai", "model": "openai/catalog-upstream",
		},
		"model_info": catalogModelInfo,
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(s04WrappedModelResponse(response, "catalog-model-id"))
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resource.Schema, map[string]interface{}{})
	d.SetId("catalog-model-id")
	states, err := resource.Importer.StateContext(context.Background(), d, nil)
	if err != nil || len(states) != 1 {
		t.Fatalf("import model: states=%d err=%v", len(states), err)
	}
	for read := 1; read <= 2; read++ {
		if err := resourceLiteLLMModelRead(states[0], NewClient(srv.URL, "test-key", false)); err != nil {
			t.Fatalf("read %d imported model: %v", read, err)
		}
		assertCatalogModelInfoState(t, states[0], "import/read")
	}
}
