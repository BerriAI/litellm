package litellm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strconv"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
	"github.com/hashicorp/terraform-plugin-sdk/v2/terraform"
)

func modelInfoBody(displayName string) string {
	modelInfo := map[string]interface{}{
		"id":         "model-123",
		"db_model":   true,
		"base_model": "claude-sonnet-4-5",
		"tier":       "free",
		"mode":       "chat",
	}
	if displayName != "" {
		modelInfo["display_name"] = displayName
	}
	body, _ := json.Marshal(map[string]interface{}{
		"model_name":     "sonnet-4-5-anthropic",
		"litellm_params": map[string]interface{}{"model": "anthropic/claude-sonnet-4-5", "custom_llm_provider": "anthropic"},
		"model_info":     modelInfo,
	})
	return string(body)
}

func modelInfoDataEnvelope(displayName string) string {
	return `{"data": [` + modelInfoBody(displayName) + `]}`
}

func TestResourceLiteLLMModelCreateSendsDisplayName(t *testing.T) {
	var createPayload map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/model/new":
			if err := json.NewDecoder(r.Body).Decode(&createPayload); err != nil {
				t.Errorf("failed to decode create payload: %v", err)
			}
			w.Write([]byte(modelInfoBody("Claude Sonnet 4.5")))
		case "/model/info":
			w.Write([]byte(modelInfoBody("Claude Sonnet 4.5")))
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resourceLiteLLMModel().Schema, map[string]interface{}{
		"model_name":          "sonnet-4-5-anthropic",
		"custom_llm_provider": "anthropic",
		"base_model":          "claude-sonnet-4-5",
		"model_api_key":       "sk-ant-test",
		"mode":                "chat",
		"display_name":        "Claude Sonnet 4.5",
	})

	if err := resourceLiteLLMModelCreate(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("create failed: %v", err)
	}

	modelInfo, ok := createPayload["model_info"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected model_info object in create payload, got %v", createPayload["model_info"])
	}
	if modelInfo["display_name"] != "Claude Sonnet 4.5" {
		t.Errorf("expected model_info.display_name 'Claude Sonnet 4.5', got %v", modelInfo["display_name"])
	}
	if got := d.Get("display_name").(string); got != "Claude Sonnet 4.5" {
		t.Errorf("expected state display_name 'Claude Sonnet 4.5', got %q", got)
	}
}

func TestResourceLiteLLMModelCreateOmitsUnsetDisplayName(t *testing.T) {
	var createPayload map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/model/new":
			if err := json.NewDecoder(r.Body).Decode(&createPayload); err != nil {
				t.Errorf("failed to decode create payload: %v", err)
			}
			w.Write([]byte(modelInfoBody("")))
		case "/model/info":
			w.Write([]byte(modelInfoBody("")))
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resourceLiteLLMModel().Schema, map[string]interface{}{
		"model_name":          "sonnet-4-5-anthropic",
		"custom_llm_provider": "anthropic",
		"base_model":          "claude-sonnet-4-5",
		"model_api_key":       "sk-ant-test",
	})

	if err := resourceLiteLLMModelCreate(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("create failed: %v", err)
	}

	modelInfo := createPayload["model_info"].(map[string]interface{})
	if _, present := modelInfo["display_name"]; present {
		t.Errorf("expected display_name to be omitted from model_info when unset, got %v", modelInfo["display_name"])
	}
	if got := d.Get("display_name").(string); got != "" {
		t.Errorf("expected empty state display_name, got %q", got)
	}
}

func TestResourceLiteLLMModelReadDisplayName(t *testing.T) {
	cases := map[string]struct {
		serverBody string
		want       string
	}{
		"server value wins inside data envelope": {serverBody: modelInfoDataEnvelope("Renamed In Admin UI"), want: "Renamed In Admin UI"},
		"server value wins unwrapped":            {serverBody: modelInfoBody("Renamed In Admin UI"), want: "Renamed In Admin UI"},
		"external removal clears state":          {serverBody: modelInfoDataEnvelope(""), want: ""},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/model/info" {
					t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
				}
				w.Write([]byte(tc.serverBody))
			}))
			defer srv.Close()

			d := schema.TestResourceDataRaw(t, resourceLiteLLMModel().Schema, map[string]interface{}{
				"model_name":          "sonnet-4-5-anthropic",
				"custom_llm_provider": "anthropic",
				"base_model":          "claude-sonnet-4-5",
				"display_name":        "Claude Sonnet 4.5",
			})
			d.SetId("model-123")

			if err := resourceLiteLLMModelRead(d, NewClient(srv.URL, "test-key", true)); err != nil {
				t.Fatalf("read failed: %v", err)
			}
			if got := d.Get("display_name").(string); got != tc.want {
				t.Errorf("expected display_name %q, got %q", tc.want, got)
			}
		})
	}
}

func updateResourceData(t *testing.T, oldDisplayName, newDisplayName string) *schema.ResourceData {
	t.Helper()
	res := resourceLiteLLMModel()
	attrs := map[string]string{
		"model_name":          "sonnet-4-5-anthropic",
		"custom_llm_provider": "anthropic",
		"base_model":          "claude-sonnet-4-5",
	}
	if oldDisplayName != "" {
		attrs["display_name"] = oldDisplayName
	}
	state := &terraform.InstanceState{ID: "model-123", Attributes: attrs}
	diff, err := res.Diff(context.Background(), state, &terraform.ResourceConfig{Config: map[string]interface{}{
		"model_name":          "sonnet-4-5-anthropic",
		"custom_llm_provider": "anthropic",
		"base_model":          "claude-sonnet-4-5",
		"display_name":        newDisplayName,
	}}, nil)
	if err != nil {
		t.Fatalf("diff failed: %v", err)
	}
	d, err := schema.InternalMap(res.Schema).Data(state, diff)
	if err != nil {
		t.Fatalf("data failed: %v", err)
	}
	return d
}

func TestResourceLiteLLMModelUpdatePatchesDisplayName(t *testing.T) {
	cases := map[string]struct {
		newName string
	}{
		"changed name is patched": {newName: "Claude Sonnet 4.5 v2"},
		"cleared name is patched": {newName: ""},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			var patchPayload map[string]interface{}
			var patchPath string
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch {
				case r.Method == http.MethodPost && r.URL.Path == "/model/update":
					w.Write([]byte(modelInfoBody("Claude Sonnet 4.5")))
				case r.Method == http.MethodPatch:
					patchPath = r.URL.Path
					if err := json.NewDecoder(r.Body).Decode(&patchPayload); err != nil {
						t.Errorf("failed to decode patch payload: %v", err)
					}
					w.Write([]byte(modelInfoBody(tc.newName)))
				case r.URL.Path == "/model/info":
					w.Write([]byte(modelInfoDataEnvelope(tc.newName)))
				default:
					t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
					w.WriteHeader(http.StatusNotFound)
				}
			}))
			defer srv.Close()

			d := updateResourceData(t, "Claude Sonnet 4.5", tc.newName)
			if err := resourceLiteLLMModelUpdate(d, NewClient(srv.URL, "test-key", true)); err != nil {
				t.Fatalf("update failed: %v", err)
			}
			if patchPath != "/model/model-123/update" {
				t.Fatalf("expected PATCH /model/model-123/update, got %q", patchPath)
			}
			modelInfo := patchPayload["model_info"].(map[string]interface{})
			if modelInfo["display_name"] != tc.newName {
				t.Errorf("expected patched display_name %q, got %v", tc.newName, modelInfo["display_name"])
			}
			if got := d.Get("display_name").(string); got != tc.newName {
				t.Errorf("expected state display_name %q, got %q", tc.newName, got)
			}
		})
	}
}

func TestResourceLiteLLMModelUpdateSkipsPatchWhenDisplayNameUnchanged(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPatch {
			t.Errorf("unexpected PATCH %s", r.URL.Path)
		}
		w.Write([]byte(modelInfoDataEnvelope("Claude Sonnet 4.5")))
	}))
	defer srv.Close()

	d := updateResourceData(t, "Claude Sonnet 4.5", "Claude Sonnet 4.5")
	if err := resourceLiteLLMModelUpdate(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("update failed: %v", err)
	}
}

func createPayloadFor(t *testing.T, config map[string]interface{}) map[string]interface{} {
	t.Helper()
	var createPayload map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/model/new":
			if err := json.NewDecoder(r.Body).Decode(&createPayload); err != nil {
				t.Errorf("failed to decode create payload: %v", err)
			}
			w.Write([]byte(modelInfoBody("")))
		case "/model/info":
			w.Write([]byte(modelInfoBody("")))
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resourceLiteLLMModel().Schema, config)
	if err := resourceLiteLLMModelCreate(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("create failed: %v", err)
	}
	return createPayload
}

func TestResourceLiteLLMModelCreateSendsAdditionalModelInfo(t *testing.T) {
	payload := createPayloadFor(t, map[string]interface{}{
		"model_name":          "sonnet-4-5-anthropic",
		"custom_llm_provider": "anthropic",
		"base_model":          "claude-sonnet-4-5",
		"additional_model_info": map[string]interface{}{
			"bedrock_converse_supports_strict_tools": "false",
			"max_input_tokens":                       "1000000",
			"supported_openai_params":                `["tools", "tool_choice"]`,
		},
	})

	modelInfo := payload["model_info"].(map[string]interface{})
	if modelInfo["bedrock_converse_supports_strict_tools"] != false {
		t.Errorf("expected bedrock_converse_supports_strict_tools false, got %#v", modelInfo["bedrock_converse_supports_strict_tools"])
	}
	if modelInfo["max_input_tokens"] != float64(1000000) {
		t.Errorf("expected max_input_tokens 1000000, got %#v", modelInfo["max_input_tokens"])
	}
	if !reflect.DeepEqual(modelInfo["supported_openai_params"], []interface{}{"tools", "tool_choice"}) {
		t.Errorf("expected supported_openai_params as a list, got %#v", modelInfo["supported_openai_params"])
	}
	if modelInfo["db_model"] != true || modelInfo["base_model"] != "claude-sonnet-4-5" || modelInfo["id"] == "" {
		t.Errorf("expected the standard model_info fields next to the additional ones, got %#v", modelInfo)
	}
}

func TestResourceLiteLLMModelCreateAppliesAdditionalDropParams(t *testing.T) {
	payload := createPayloadFor(t, map[string]interface{}{
		"model_name":          "sonnet-4-5-anthropic",
		"custom_llm_provider": "anthropic",
		"base_model":          "claude-sonnet-4-5",
		"tpm":                 100,
		"additional_litellm_params": map[string]interface{}{
			"additional_drop_params": `["tpm"]`,
			"timeout":                "30",
		},
	})

	params := payload["litellm_params"].(map[string]interface{})
	if _, ok := params["tpm"]; ok {
		t.Errorf("expected tpm to be dropped, got %#v", params["tpm"])
	}
	if _, ok := params["additional_drop_params"]; ok {
		t.Errorf("expected additional_drop_params to stay out of litellm_params")
	}
	if params["timeout"] != float64(30) {
		t.Errorf("expected timeout 30, got %#v", params["timeout"])
	}
}

func updateAdditionalModelInfoData(t *testing.T, oldInfo, newInfo map[string]string) *schema.ResourceData {
	t.Helper()
	res := resourceLiteLLMModel()
	attrs := map[string]string{
		"model_name":              "sonnet-4-5-anthropic",
		"custom_llm_provider":     "anthropic",
		"base_model":              "claude-sonnet-4-5",
		"display_name":            "Claude Sonnet 4.5",
		"additional_model_info.%": strconv.Itoa(len(oldInfo)),
	}
	for key, value := range oldInfo {
		attrs["additional_model_info."+key] = value
	}
	config := map[string]interface{}{}
	for key, value := range newInfo {
		config[key] = value
	}
	state := &terraform.InstanceState{ID: "model-123", Attributes: attrs}
	diff, err := res.Diff(context.Background(), state, &terraform.ResourceConfig{Config: map[string]interface{}{
		"model_name":            "sonnet-4-5-anthropic",
		"custom_llm_provider":   "anthropic",
		"base_model":            "claude-sonnet-4-5",
		"display_name":          "Claude Sonnet 4.5",
		"additional_model_info": config,
	}}, nil)
	if err != nil {
		t.Fatalf("diff failed: %v", err)
	}
	d, err := schema.InternalMap(res.Schema).Data(state, diff)
	if err != nil {
		t.Fatalf("data failed: %v", err)
	}
	return d
}

func TestResourceLiteLLMModelUpdatePatchesChangedAdditionalModelInfo(t *testing.T) {
	var patchPayload map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/model/update":
			w.Write([]byte(modelInfoBody("Claude Sonnet 4.5")))
		case r.Method == http.MethodPatch && r.URL.Path == "/model/model-123/update":
			if err := json.NewDecoder(r.Body).Decode(&patchPayload); err != nil {
				t.Errorf("failed to decode patch payload: %v", err)
			}
			w.Write([]byte(modelInfoBody("Claude Sonnet 4.5")))
		case r.URL.Path == "/model/info":
			w.Write([]byte(modelInfoDataEnvelope("Claude Sonnet 4.5")))
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	d := updateAdditionalModelInfoData(t, map[string]string{"max_input_tokens": "200000"}, map[string]string{"max_input_tokens": "1000000"})
	if err := resourceLiteLLMModelUpdate(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("update failed: %v", err)
	}

	want := map[string]interface{}{"id": "model-123", "max_input_tokens": float64(1000000)}
	if !reflect.DeepEqual(patchPayload["model_info"], want) {
		t.Errorf("expected PATCH model_info %#v without the unchanged display_name, got %#v", want, patchPayload["model_info"])
	}
}

func TestResourceLiteLLMModelUpdateSkipsPatchWhenAdditionalModelInfoUnchanged(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPatch {
			t.Errorf("unexpected PATCH %s", r.URL.Path)
		}
		w.Write([]byte(modelInfoDataEnvelope("Claude Sonnet 4.5")))
	}))
	defer srv.Close()

	info := map[string]string{"max_input_tokens": "1000000"}
	d := updateAdditionalModelInfoData(t, info, info)
	if err := resourceLiteLLMModelUpdate(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("update failed: %v", err)
	}
}

func TestValidateAdditionalModelInfoRejectsKeysWithTheirOwnArgument(t *testing.T) {
	validate := resourceLiteLLMModel().Schema["additional_model_info"].ValidateDiagFunc
	for _, key := range []string{"id", "db_model", "base_model", "tier", "mode", "team_id", "display_name"} {
		if !validate(map[string]interface{}{key: "x"}, nil).HasError() {
			t.Errorf("expected %q to be rejected", key)
		}
	}
	if diags := validate(map[string]interface{}{"max_input_tokens": "1000000"}, nil); diags.HasError() {
		t.Errorf("expected max_input_tokens to be accepted, got %v", diags)
	}
}

func TestParseAdditionalValue(t *testing.T) {
	cases := map[string]struct {
		in   interface{}
		want interface{}
	}{
		"true becomes a bool":         {in: "true", want: true},
		"false becomes a bool":        {in: "false", want: false},
		"integer":                     {in: "16384", want: 16384},
		"float":                       {in: "0.75", want: 0.75},
		"json object":                 {in: `{"nested": {"value": 42}}`, want: map[string]interface{}{"nested": map[string]interface{}{"value": float64(42)}}},
		"json array":                  {in: `["a", "b"]`, want: []interface{}{"a", "b"}},
		"invalid json stays a string": {in: "{not json", want: "{not json"},
		"plain string":                {in: "for testing", want: "for testing"},
		"non-string passes through":   {in: 7, want: 7},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			if got := parseAdditionalValue(tc.in); !reflect.DeepEqual(got, tc.want) {
				t.Errorf("parseAdditionalValue(%#v) = %#v, want %#v", tc.in, got, tc.want)
			}
		})
	}
}
