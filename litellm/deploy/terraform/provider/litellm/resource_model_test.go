package litellm

import (
	"context"
	"encoding/json"
	"maps"
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
		extraConfig    map[string]interface{}
		wantChanged    []string
	}{
		"list never sent by an older provider": {configured: `["store"]`, configLocation: "eu", storedLocation: "eu", wantChanged: []string{listKey}},
		"listed keys an older provider deleted": {
			configured:     `["store", "vertex_location", "tpm", "max_parallel_requests"]`,
			configLocation: "eu",
			wantChanged:    []string{listKey, "vertex_location", "tpm", parallelKey},
		},
		"listed flag an older provider deleted": {
			configured:     `["merge_reasoning_content_in_choices"]`,
			configLocation: "eu",
			storedLocation: "eu",
			extraConfig:    map[string]interface{}{"merge_reasoning_content_in_choices": true},
			wantChanged:    []string{listKey, "merge_reasoning_content_in_choices"},
		},
		"listed key an older provider still sent": {
			configured:     `["litellm_credential_name"]`,
			configLocation: "eu",
			storedLocation: "eu",
			extraConfig:    map[string]interface{}{"litellm_credential_name": "vertex-cred"},
			wantChanged:    []string{listKey},
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
			maps.Copy(config, tc.extraConfig)
			d := schema.TestResourceDataRaw(t, res.Schema, config)
			d.SetId("model-123")
			if err := resourceLiteLLMModelRead(d, NewClient(srv.URL, "test-key", true)); err != nil {
				t.Fatalf("read failed: %v", err)
			}

			diff, err := res.Diff(context.Background(), d.State(), terraform.NewResourceConfigRaw(config), nil)
			if err != nil {
				t.Fatalf("diff failed: %v", err)
			}
			for _, key := range []string{listKey, "vertex_location", "tpm", parallelKey, "merge_reasoning_content_in_choices", "litellm_credential_name"} {
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

func TestResourceLiteLLMModelReadRoutingBaseModel(t *testing.T) {
	cases := []struct {
		name       string
		params     map[string]string
		pricing    string
		wantModel  string
		wantVendor string
	}{
		{
			name:       "import without pricing metadata",
			params:     map[string]string{"model": "azure_ai/gpt-6-astra", "custom_llm_provider": "azure_ai"},
			wantModel:  "gpt-6-astra",
			wantVendor: "azure_ai",
		},
		{
			name:       "import with independent pricing metadata and nested model name",
			params:     map[string]string{"model": "openrouter/anthropic/sonnet"},
			pricing:    "anthropic/claude-sonnet",
			wantModel:  "anthropic/sonnet",
			wantVendor: "openrouter",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			info := map[string]string{"id": "model-123"}
			if tc.pricing != "" {
				info["base_model"] = tc.pricing
			}
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodGet || r.URL.Path != "/model/info" || r.URL.Query().Get("litellm_model_id") != "model-123" {
					t.Errorf("unexpected request: %s %s", r.Method, r.URL)
				}
				err := json.NewEncoder(w).Encode(map[string]any{
					"data": []map[string]any{{"model_name": "imported-model", "litellm_params": tc.params, "model_info": info}},
				})
				if err != nil {
					t.Errorf("write model response: %v", err)
				}
			}))
			defer srv.Close()

			d := schema.TestResourceDataRaw(t, resourceLiteLLMModel().Schema, nil)
			d.SetId("model-123")
			if err := resourceLiteLLMModelRead(d, NewClient(srv.URL, "test-key", true)); err != nil {
				t.Fatalf("read imported model: %v", err)
			}
			if got := d.Get("base_model"); got != tc.wantModel {
				t.Errorf("base_model = %q, want %q", got, tc.wantModel)
			}
			if got := d.Get("custom_llm_provider"); got != tc.wantVendor {
				t.Errorf("custom_llm_provider = %q, want %q", got, tc.wantVendor)
			}
		})
	}
}

func TestResourceLiteLLMModelCreateRejectsEmptyRoutingSuffix(t *testing.T) {
	for _, baseModel := range []string{"", " "} {
		d := schema.TestResourceDataRaw(t, resourceLiteLLMModel().Schema, map[string]any{
			"model_name": "invalid-model", "custom_llm_provider": "anthropic", "base_model": baseModel,
		})
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			t.Errorf("unexpected request for base_model %q: %s %s", baseModel, r.Method, r.URL.Path)
			w.WriteHeader(http.StatusBadRequest)
		}))
		err := resourceLiteLLMModelCreate(d, NewClient(srv.URL, "test-key", true))
		srv.Close()
		if err == nil {
			t.Errorf("expected an error for base_model %q", baseModel)
		}
	}
}

func TestResourceLiteLLMModelUpdateWritesRoutingOnlyWhenItChanges(t *testing.T) {
	for _, tc := range []struct {
		name      string
		baseModel string
		wantRoute string
		wantPrice string
	}{
		{name: "rate limit change preserves external routing and pricing", baseModel: "claude-sonnet-4-5"},
		{name: "deployment change writes the new routing and default pricing", baseModel: "new-deployment", wantRoute: "anthropic/new-deployment", wantPrice: "new-deployment"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var update struct {
				Params map[string]json.RawMessage `json:"litellm_params"`
				Info   map[string]json.RawMessage `json:"model_info"`
			}
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch {
				case r.Method == http.MethodPost && r.URL.Path == "/model/update":
					if err := json.NewDecoder(r.Body).Decode(&update); err != nil {
						t.Errorf("decode model update: %v", err)
					}
				case r.Method == http.MethodGet && r.URL.Path == "/model/info":
				default:
					t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
					w.WriteHeader(http.StatusNotFound)
					return
				}
				_, _ = w.Write([]byte(modelInfoBody("")))
			}))
			defer srv.Close()

			res := resourceLiteLLMModel()
			config := map[string]any{
				"model_name": "sonnet-4-5-anthropic", "custom_llm_provider": "anthropic", "base_model": tc.baseModel, "rpm": 7,
			}
			d := schema.TestResourceDataRaw(t, res.Schema, nil)
			d.SetId("model-123")
			client := NewClient(srv.URL, "test-key", true)
			if err := resourceLiteLLMModelRead(d, client); err != nil {
				t.Fatalf("read imported model: %v", err)
			}
			state := d.State()
			diff, err := res.Diff(context.Background(), state, terraform.NewResourceConfigRaw(config), nil)
			if err != nil {
				t.Fatalf("plan model update: %v", err)
			}
			d, err = schema.InternalMap(res.Schema).Data(state, diff)
			if err != nil {
				t.Fatalf("prepare model update: %v", err)
			}
			if err := resourceLiteLLMModelUpdate(d, client); err != nil {
				t.Fatalf("update model: %v", err)
			}
			if got := string(update.Params["rpm"]); got != "7" {
				t.Errorf("litellm_params.rpm = %s, want 7", got)
			}
			for _, field := range []struct {
				key  string
				got  json.RawMessage
				want string
			}{
				{"litellm_params.model", update.Params["model"], tc.wantRoute},
				{"model_info.base_model", update.Info["base_model"], tc.wantPrice},
			} {
				if field.want == "" {
					if field.got != nil {
						t.Errorf("expected %s to be omitted, got %s", field.key, field.got)
					}
				} else if encoded, _ := json.Marshal(field.want); string(field.got) != string(encoded) {
					t.Errorf("%s = %s, want %s", field.key, field.got, encoded)
				}
			}
		})
	}
}

func TestResourceLiteLLMModelRefreshAndUpdateWithRoutingOverride(t *testing.T) {
	for _, tc := range []struct {
		name     string
		override map[string]any
		route    string
	}{
		{"keep the routing override", map[string]any{"model": "openai/deployment"}, "openai/deployment"},
		{"remove the routing override", map[string]any{}, "openai/gpt-4"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			params := map[string]any{"model": "openai/deployment", "custom_llm_provider": "openai", "rpm": 5}
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch {
				case r.Method == http.MethodPost && r.URL.Path == "/model/update":
					var update ModelRequest
					if err := json.NewDecoder(r.Body).Decode(&update); err != nil {
						t.Errorf("decode model update: %v", err)
						w.WriteHeader(http.StatusBadRequest)
						return
					}
					if got := update.LiteLLMParams["model"]; got != tc.route {
						t.Errorf("updated route = %q, want %q", got, tc.route)
					}
					maps.Copy(params, update.LiteLLMParams)
				case r.Method == http.MethodGet && r.URL.Path == "/model/info":
				default:
					t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
					w.WriteHeader(http.StatusNotFound)
					return
				}
				if err := json.NewEncoder(w).Encode(map[string]any{
					"data": []map[string]any{{"model_name": "example", "litellm_params": params, "model_info": map[string]string{"id": "model-123"}}},
				}); err != nil {
					t.Errorf("write model response: %v", err)
				}
			}))
			defer srv.Close()

			res := resourceLiteLLMModel()
			config := map[string]any{
				"model_name": "example", "custom_llm_provider": "openai", "base_model": "gpt-4", "rpm": 5,
				"additional_litellm_params": map[string]any{"model": "openai/deployment"},
			}
			d := schema.TestResourceDataRaw(t, res.Schema, config)
			d.SetId("model-123")
			client := NewClient(srv.URL, "test-key", true)
			if err := resourceLiteLLMModelRead(d, client); err != nil {
				t.Fatalf("refresh model: %v", err)
			}
			state := d.State()
			config["rpm"] = 7
			config["additional_litellm_params"] = tc.override
			diff, err := res.Diff(context.Background(), state, terraform.NewResourceConfigRaw(config), nil)
			if err != nil {
				t.Fatalf("plan model update: %v", err)
			}
			if change, ok := diff.Attributes["base_model"]; ok {
				t.Errorf("unexpected base_model change: %+v", change)
			}
			d, err = schema.InternalMap(res.Schema).Data(state, diff)
			if err != nil {
				t.Fatalf("prepare model update: %v", err)
			}
			if err := resourceLiteLLMModelUpdate(d, client); err != nil {
				t.Fatalf("update model: %v", err)
			}
			if got := d.Get("base_model"); got != "gpt-4" {
				t.Errorf("base_model after update = %q, want gpt-4", got)
			}
		})
	}
}

func TestResourceLiteLLMModelCreateOmitsEmptyTier(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/model/new" {
			var payload struct {
				ModelInfo map[string]json.RawMessage `json:"model_info"`
			}
			if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
				t.Errorf("decode model request: %v", err)
			}
			if tier, exists := payload.ModelInfo["tier"]; exists {
				t.Errorf("expected tier to be omitted, got %s", tier)
			}
		}
		_, _ = w.Write([]byte(modelInfoBody("")))
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resourceLiteLLMModel().Schema, map[string]any{
		"model_name": "sonnet-4-5-anthropic", "custom_llm_provider": "anthropic", "base_model": "claude-sonnet-4-5", "tier": "",
	})
	if err := resourceLiteLLMModelCreate(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("create without tier: %v", err)
	}
}

func TestResourceLiteLLMModelPlanKeepsStoredModeWhenUnset(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(modelInfoDataEnvelope("")))
	}))
	defer srv.Close()

	res := resourceLiteLLMModel()
	config := map[string]any{
		"model_name": "sonnet-4-5-anthropic", "custom_llm_provider": "anthropic", "base_model": "claude-sonnet-4-5",
	}
	d := schema.TestResourceDataRaw(t, res.Schema, config)
	d.SetId("model-123")
	if err := resourceLiteLLMModelRead(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("read model: %v", err)
	}
	diff, err := res.Diff(context.Background(), d.State(), terraform.NewResourceConfigRaw(config), nil)
	if err != nil {
		t.Fatalf("plan model: %v", err)
	}
	if diff != nil && diff.Attributes["mode"] != nil {
		t.Errorf("unexpected mode change: %+v", diff.Attributes["mode"])
	}
}

func TestResourceLiteLLMModelReadEmptyBodyKeepsState(t *testing.T) {
	cases := []struct {
		name       string
		serverBody string
	}{
		{name: "empty body"},
		{name: "null body", serverBody: "null"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodGet || r.URL.Path != "/model/info" {
					t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
				}
				_, _ = w.Write([]byte(tc.serverBody))
			}))
			defer srv.Close()

			d := schema.TestResourceDataRaw(t, resourceLiteLLMModel().Schema, map[string]interface{}{
				"model_name":          "sonnet-4-5-anthropic",
				"custom_llm_provider": "anthropic",
				"base_model":          "claude-sonnet-4-5",
				"display_name":        "Saved Name",
			})
			d.SetId("model-123")

			if err := resourceLiteLLMModelRead(d, NewClient(srv.URL, "test-key", true)); err == nil {
				t.Fatal("expected read error for empty response body")
			}
			if got := d.Get("display_name").(string); got != "Saved Name" {
				t.Errorf("expected display_name to remain %q, got %q", "Saved Name", got)
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

func TestResourceLiteLLMModel_ModeValidation(t *testing.T) {
	validateMode := resourceLiteLLMModel().Schema["mode"].ValidateFunc

	valid := []string{
		"completion", "embedding", "image_generation", "chat", "moderation",
		"audio_transcription", "audio_speech", "rerank",
		"responses", "realtime", "image_edit", "ocr", "search", "video_generation", "evaluation",
		"decisions", "extraction",
	}
	for _, mode := range valid {
		if _, errs := validateMode(mode, "mode"); len(errs) != 0 {
			t.Errorf("mode %q: expected no validation error, got %v", mode, errs)
		}
	}

	if _, errs := validateMode("not_a_real_mode", "mode"); len(errs) == 0 {
		t.Error(`mode "not_a_real_mode": expected a validation error, got none`)
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

func modelInfoWithCapabilities(capabilities map[string]interface{}) string {
	modelInfo := map[string]interface{}{"id": "model-123", "db_model": true, "mode": "chat"}
	maps.Copy(modelInfo, capabilities)
	body, _ := json.Marshal(map[string]interface{}{
		"model_name":     "glm",
		"litellm_params": map[string]interface{}{"model": "hosted_vllm/glm-5.3", "custom_llm_provider": "hosted_vllm"},
		"model_info":     modelInfo,
	})
	return `{"data": [` + string(body) + `]}`
}

func capabilityModelConfig(capabilities map[string]interface{}) map[string]interface{} {
	config := map[string]interface{}{"model_name": "glm", "custom_llm_provider": "hosted_vllm", "base_model": "glm-5.3"}
	maps.Copy(config, capabilities)
	return config
}

func TestResourceLiteLLMModelCreateSendsConfiguredCapabilities(t *testing.T) {
	var createPayload map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/model/new" {
			if err := json.NewDecoder(r.Body).Decode(&createPayload); err != nil {
				t.Errorf("decode create payload: %v", err)
			}
		}
		_, _ = w.Write([]byte(modelInfoWithCapabilities(nil)))
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resourceLiteLLMModel().Schema, capabilityModelConfig(map[string]interface{}{
		"max_input_tokens":   256000,
		"supports_vision":    false,
		"supports_reasoning": true,
	}))
	if err := resourceLiteLLMModelCreate(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("create failed: %v", err)
	}

	got := createPayload["model_info"].(map[string]interface{})
	want := map[string]interface{}{"max_input_tokens": float64(256000), "supports_vision": false, "supports_reasoning": true}
	for key, value := range want {
		if got[key] != value {
			t.Errorf("model_info.%s: want %v, got %v", key, value, got[key])
		}
	}
	for _, key := range []string{"max_output_tokens", "supports_function_calling"} {
		if _, present := got[key]; present {
			t.Errorf("model_info.%s should be omitted when unset, got %v", key, got[key])
		}
	}
}

func TestResourceLiteLLMModelReadCapabilities(t *testing.T) {
	cases := map[string]struct {
		config     map[string]interface{}
		server     map[string]interface{}
		wantChange []string
	}{
		"matching values plan no changes": {
			config: map[string]interface{}{"max_input_tokens": 256000, "supports_vision": false},
			server: map[string]interface{}{"max_input_tokens": 256000, "supports_vision": false},
		},
		"values changed outside Terraform show as drift": {
			config:     map[string]interface{}{"max_input_tokens": 256000, "supports_vision": false},
			server:     map[string]interface{}{"max_input_tokens": 128000, "supports_vision": true},
			wantChange: []string{"max_input_tokens", "supports_vision"},
		},
		"values missing on the proxy show as drift": {
			config:     map[string]interface{}{"max_output_tokens": 8192, "supports_function_calling": true},
			server:     map[string]interface{}{},
			wantChange: []string{"max_output_tokens", "supports_function_calling"},
		},
		"values Terraform does not manage are ignored": {
			config: map[string]interface{}{},
			server: map[string]interface{}{"max_input_tokens": 200000, "max_output_tokens": 8192, "supports_reasoning": true},
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = w.Write([]byte(modelInfoWithCapabilities(tc.server)))
			}))
			defer srv.Close()

			res := resourceLiteLLMModel()
			config := capabilityModelConfig(tc.config)
			d := schema.TestResourceDataRaw(t, res.Schema, config)
			d.SetId("model-123")
			if err := resourceLiteLLMModelRead(d, NewClient(srv.URL, "test-key", true)); err != nil {
				t.Fatalf("read failed: %v", err)
			}
			diff, err := res.Diff(context.Background(), d.State(), terraform.NewResourceConfigRaw(config), nil)
			if err != nil {
				t.Fatalf("plan failed: %v", err)
			}
			changed := []string{}
			for _, key := range modelCapabilityAttributes {
				if diff != nil && diff.Attributes[key] != nil {
					changed = append(changed, key)
				}
			}
			want := tc.wantChange
			if want == nil {
				want = []string{}
			}
			if !slices.Equal(changed, want) {
				t.Errorf("planned capability changes: want %v, got %v", want, changed)
			}
		})
	}
}

func capabilityUpdateData(t *testing.T, oldState map[string]string, newConfig map[string]interface{}) *schema.ResourceData {
	t.Helper()
	res := resourceLiteLLMModel()
	attrs := map[string]string{"model_name": "glm", "custom_llm_provider": "hosted_vllm", "base_model": "glm-5.3"}
	maps.Copy(attrs, oldState)
	state := &terraform.InstanceState{ID: "model-123", Attributes: attrs}
	diff, err := res.Diff(context.Background(), state, terraform.NewResourceConfigRaw(capabilityModelConfig(newConfig)), nil)
	if err != nil {
		t.Fatalf("diff failed: %v", err)
	}
	d, err := schema.InternalMap(res.Schema).Data(state, diff)
	if err != nil {
		t.Fatalf("data failed: %v", err)
	}
	return d
}

func TestResourceLiteLLMModelUpdatePatchesChangedCapabilities(t *testing.T) {
	var patchPayload map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPatch {
			if r.URL.Path != "/model/model-123/update" {
				t.Errorf("unexpected PATCH %s", r.URL.Path)
			}
			if err := json.NewDecoder(r.Body).Decode(&patchPayload); err != nil {
				t.Errorf("decode patch payload: %v", err)
			}
		}
		_, _ = w.Write([]byte(modelInfoWithCapabilities(map[string]interface{}{
			"max_input_tokens": 256000, "supports_vision": false,
		})))
	}))
	defer srv.Close()

	d := capabilityUpdateData(t,
		map[string]string{"max_input_tokens": "128000", "supports_vision": "true"},
		map[string]interface{}{"max_input_tokens": 256000, "supports_vision": false},
	)
	if err := resourceLiteLLMModelUpdate(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("update failed: %v", err)
	}
	if patchPayload == nil {
		t.Fatal("expected a PATCH /model/model-123/update, got none")
	}
	got := patchPayload["model_info"].(map[string]interface{})
	if got["max_input_tokens"] != float64(256000) || got["supports_vision"] != false {
		t.Errorf("patched model_info: want max_input_tokens 256000 and supports_vision false, got %v", got)
	}
	if v := d.Get("max_input_tokens").(int); v != 256000 {
		t.Errorf("state max_input_tokens: want 256000, got %d", v)
	}
}

func TestResourceLiteLLMModelUpdateSkipsPatchWhenCapabilitiesUnchanged(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPatch {
			t.Errorf("unexpected PATCH %s", r.URL.Path)
		}
		_, _ = w.Write([]byte(modelInfoWithCapabilities(map[string]interface{}{"supports_vision": false})))
	}))
	defer srv.Close()

	d := capabilityUpdateData(t,
		map[string]string{"supports_vision": "false", "tpm": "10"},
		map[string]interface{}{"supports_vision": false, "tpm": 20},
	)
	if err := resourceLiteLLMModelUpdate(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("update failed: %v", err)
	}
}

func TestResourceLiteLLMModelRejectsNonPositiveTokenLimits(t *testing.T) {
	for _, key := range []string{"max_input_tokens", "max_output_tokens"} {
		validate := resourceLiteLLMModel().Schema[key].ValidateFunc
		if _, errs := validate(0, key); len(errs) == 0 {
			t.Errorf("%s = 0 should be rejected", key)
		}
		if _, errs := validate(1, key); len(errs) != 0 {
			t.Errorf("%s = 1 should be accepted, got %v", key, errs)
		}
	}
}
