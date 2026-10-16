package litellm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"testing"

	"github.com/hashicorp/go-cty/cty"
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

func TestResourceLiteLLMModelCreateSendsAdditionalLiteLLMParams(t *testing.T) {
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
		"model_name":          "vertex-claude-haiku",
		"custom_llm_provider": "vertex_ai",
		"base_model":          "claude-haiku-4-5",
		"vertex_project":      "test-project",
		"vertex_location":     "eu",
		"additional_litellm_params": map[string]interface{}{
			"additional_drop_params":             `["store", "vertex_location"]`,
			"tags":                               ` ["prod", "eu"]`,
			"headers":                            `{"x-team": "search"}`,
			"use_in_pass_through":                "true",
			"merge_reasoning_content_in_choices": "false",
			"max_parallel_requests":              "8",
			"input_cost_per_token":               "0.5",
			"description":                        "[draft] haiku",
		},
	})

	if err := resourceLiteLLMModelCreate(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("create failed: %v", err)
	}

	litellmParams, ok := createPayload["litellm_params"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected litellm_params object in create payload, got %v", createPayload["litellm_params"])
	}
	want := map[string]interface{}{
		"additional_drop_params":             []interface{}{"store", "vertex_location"},
		"vertex_project":                     "test-project",
		"vertex_location":                    "eu",
		"tags":                               []interface{}{"prod", "eu"},
		"headers":                            map[string]interface{}{"x-team": "search"},
		"use_in_pass_through":                true,
		"merge_reasoning_content_in_choices": false,
		"max_parallel_requests":              float64(8),
		"input_cost_per_token":               0.5,
		"description":                        "[draft] haiku",
	}
	for key, wantValue := range want {
		if got := litellmParams[key]; !reflect.DeepEqual(got, wantValue) {
			t.Errorf("litellm_params[%q] = %#v, want %#v", key, got, wantValue)
		}
	}
}

const terraformUnknownValue = "74D93920-ED26-11E3-AC10-0800200C9A66"

func TestResourceLiteLLMModelChecksAdditionalDropParams(t *testing.T) {
	cases := map[string]struct {
		dropParams string
		want       []interface{}
		wantErr    bool
	}{
		"null entry from an unset variable is skipped": {dropParams: `["store", null]`, want: []interface{}{"store"}},
		"number entry is rejected":                     {dropParams: `["store", 1]`, wantErr: true},
		"object entry is rejected":                     {dropParams: `["store", {"name": "temperature"}]`, wantErr: true},
		"object is rejected":                           {dropParams: `{"store": true}`, wantErr: true},
		"bare name is rejected":                        {dropParams: "store", wantErr: true},
		"empty value is rejected":                      {dropParams: "", wantErr: true},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
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

			res := resourceLiteLLMModel()
			config := map[string]interface{}{
				"model_name":                "vertex-claude-haiku",
				"custom_llm_provider":       "vertex_ai",
				"base_model":                "claude-haiku-4-5",
				"additional_litellm_params": map[string]interface{}{"additional_drop_params": tc.dropParams},
			}
			if _, err := res.Diff(context.Background(), nil, terraform.NewResourceConfigRaw(config), nil); (err != nil) != tc.wantErr {
				t.Errorf("plan error = %v, want error %v", err, tc.wantErr)
			}

			d := schema.TestResourceDataRaw(t, res.Schema, config)
			err := resourceLiteLLMModelCreate(d, NewClient(srv.URL, "test-key", true))
			if tc.wantErr {
				if err == nil {
					t.Fatal("expected create to fail")
				}
				if createPayload != nil {
					t.Errorf("expected no /model/new request, got %v", createPayload)
				}
				return
			}
			if err != nil {
				t.Fatalf("create failed: %v", err)
			}
			litellmParams := createPayload["litellm_params"].(map[string]interface{})
			if got := litellmParams["additional_drop_params"]; !reflect.DeepEqual(got, tc.want) {
				t.Errorf("additional_drop_params = %#v, want %#v", got, tc.want)
			}
		})
	}
}

func TestResourceLiteLLMModelPlanAcceptsDropParamsUnknownUntilApply(t *testing.T) {
	config := terraform.NewResourceConfigRaw(map[string]interface{}{
		"model_name":                "vertex-claude-haiku",
		"custom_llm_provider":       "vertex_ai",
		"additional_litellm_params": map[string]interface{}{"additional_drop_params": terraformUnknownValue},
	})
	if _, err := resourceLiteLLMModel().Diff(context.Background(), nil, config, nil); err != nil {
		t.Errorf("plan failed on a drop list not known until apply: %v", err)
	}
}

func TestResourceLiteLLMModelPlanComparesAdditionalDropParamsWithProxy(t *testing.T) {
	const (
		listKey     = "additional_litellm_params.additional_drop_params"
		parallelKey = "additional_litellm_params.max_parallel_requests"
	)
	cases := map[string]struct {
		configured     string
		configLocation string
		stored         interface{}
		storedLocation string
		wantChanged    []string
	}{
		"list never sent by an older provider": {configured: `["store"]`, configLocation: "eu", storedLocation: "eu", wantChanged: []string{listKey}},
		"listed keys an older provider deleted": {
			configured:     `["store", "vertex_location", "tpm", "max_parallel_requests"]`,
			configLocation: "eu",
			wantChanged:    []string{listKey, "vertex_location", "tpm", parallelKey},
		},
		"list changed in the Admin UI":         {configured: `["store"]`, configLocation: "eu", stored: []string{"temperature"}, storedLocation: "eu", wantChanged: []string{listKey}},
		"list set outside Terraform":           {configLocation: "eu", stored: []string{"store"}, storedLocation: "eu"},
		"same list written with spaces":        {configured: `["store", "vertex_location"]`, configLocation: "eu", stored: []string{"store", "vertex_location"}, storedLocation: "eu"},
		"null entry the proxy never received":  {configured: `["store", null]`, configLocation: "eu", stored: []string{"store"}, storedLocation: "eu"},
		"listed key left out of the config":    {configured: `["vertex_location"]`, stored: []string{"vertex_location"}, storedLocation: "us-east5"},
		"no list on the model or in the proxy": {configLocation: "eu", storedLocation: "eu"},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			litellmParams := map[string]interface{}{"model": "vertex_ai/claude-haiku-4-5", "custom_llm_provider": "vertex_ai"}
			if tc.stored != nil {
				litellmParams["additional_drop_params"] = tc.stored
			}
			if tc.storedLocation != "" {
				litellmParams["vertex_location"] = tc.storedLocation
			}
			body, _ := json.Marshal(map[string]interface{}{
				"model_name":     "vertex-claude-haiku",
				"litellm_params": litellmParams,
				"model_info":     map[string]interface{}{"id": "model-123", "db_model": true, "base_model": "claude-haiku-4-5"},
			})
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Write(body)
			}))
			defer srv.Close()

			res := resourceLiteLLMModel()
			additionalParams := map[string]interface{}{"max_parallel_requests": "8"}
			if tc.configured != "" {
				additionalParams["additional_drop_params"] = tc.configured
			}
			config := map[string]interface{}{
				"model_name":                "vertex-claude-haiku",
				"custom_llm_provider":       "vertex_ai",
				"base_model":                "claude-haiku-4-5",
				"tpm":                       1000,
				"additional_litellm_params": additionalParams,
			}
			if tc.configLocation != "" {
				config["vertex_location"] = tc.configLocation
			}
			d := schema.TestResourceDataRaw(t, res.Schema, config)
			d.SetId("model-123")
			if err := resourceLiteLLMModelRead(d, NewClient(srv.URL, "test-key", true)); err != nil {
				t.Fatalf("read failed: %v", err)
			}

			diff, err := res.Diff(context.Background(), d.State(), terraform.NewResourceConfigRaw(config), nil)
			if err != nil {
				t.Fatalf("diff failed: %v", err)
			}
			for _, key := range []string{listKey, "vertex_location", "tpm", parallelKey} {
				attr := (*terraform.ResourceAttrDiff)(nil)
				if diff != nil {
					attr = diff.Attributes[key]
				}
				got := attr != nil && attr.Old != attr.New
				if want := slices.Contains(tc.wantChanged, key); got != want {
					t.Errorf("plan shows %s change = %v, want %v", key, got, want)
				}
			}
		})
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

func TestResourceLiteLLMModelCreateSendsWriteOnlySecrets(t *testing.T) {
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

	d := resourceLiteLLMModel().Data(&terraform.InstanceState{
		Attributes: map[string]string{
			"model_name":                       "sonnet-4-5-anthropic",
			"custom_llm_provider":              "anthropic",
			"base_model":                       "claude-sonnet-4-5",
			"model_api_key_wo_version":         "1",
			"aws_secret_access_key_wo_version": "1",
			"vertex_credentials_wo_version":    "1",
		},
		RawConfig: cty.ObjectVal(map[string]cty.Value{
			"model_api_key_wo":         cty.StringVal("sk-ant-only-in-config"),
			"aws_secret_access_key_wo": cty.StringVal("aws-secret-only-in-config"),
			"vertex_credentials_wo":    cty.StringVal(`{"type":"service_account"}`),
		}),
	})

	if err := resourceLiteLLMModelCreate(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("create failed: %v", err)
	}

	params, ok := createPayload["litellm_params"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected litellm_params object in create payload, got %v", createPayload["litellm_params"])
	}
	want := map[string]string{
		"api_key":               "sk-ant-only-in-config",
		"aws_secret_access_key": "aws-secret-only-in-config",
		"vertex_credentials":    `{"type":"service_account"}`,
	}
	for key, value := range want {
		if params[key] != value {
			t.Errorf("litellm_params.%s = %v, want %q", key, params[key], value)
		}
	}
	for _, attr := range []string{"model_api_key", "aws_secret_access_key", "vertex_credentials"} {
		if got := d.Get(attr).(string); got != "" {
			t.Errorf("in-state %s should stay empty when the _wo variant is used, got %q", attr, got)
		}
	}
}

func TestResourceLiteLLMModelSchemaRejectsSecretInStateAndWriteOnly(t *testing.T) {
	base := map[string]interface{}{
		"model_name":          "m",
		"custom_llm_provider": "openai",
		"base_model":          "gpt-4.1",
	}
	cases := map[string]struct {
		extra   map[string]interface{}
		wantErr bool
	}{
		"write-only with version": {
			extra:   map[string]interface{}{"model_api_key_wo": "sk", "model_api_key_wo_version": 1},
			wantErr: false,
		},
		"write-only without version": {
			extra:   map[string]interface{}{"model_api_key_wo": "sk"},
			wantErr: true,
		},
		"both in-state and write-only": {
			extra:   map[string]interface{}{"model_api_key": "sk", "model_api_key_wo": "sk", "model_api_key_wo_version": 1},
			wantErr: true,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			raw := map[string]interface{}{}
			for k, v := range base {
				raw[k] = v
			}
			for k, v := range tc.extra {
				raw[k] = v
			}
			diags := resourceLiteLLMModel().Validate(terraform.NewResourceConfigRaw(raw))
			if diags.HasError() != tc.wantErr {
				t.Fatalf("wantErr=%v, got diags %v", tc.wantErr, diags)
			}
		})
	}
}
