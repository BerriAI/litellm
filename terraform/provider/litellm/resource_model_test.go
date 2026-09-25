package litellm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
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

func TestResourceLiteLLMModelUpdateSendsReferencePixelRate(t *testing.T) {
	cases := map[string]struct {
		oldRate, newRate string
		wantSent         bool
		wantRate         float64
	}{
		"rate cleared to zero is sent": {oldRate: "2e-07", newRate: "0", wantSent: true, wantRate: 0},
		"unchanged rate is sent":       {oldRate: "2e-07", newRate: "2e-07", wantSent: true, wantRate: 2e-07},
		"never set rate is omitted":    {oldRate: "", newRate: "0", wantSent: false},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			var updateParams map[string]interface{}
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method == http.MethodPost && r.URL.Path == "/model/update" {
					var payload map[string]interface{}
					if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
						t.Errorf("failed to decode update payload: %v", err)
					}
					updateParams = payload["litellm_params"].(map[string]interface{})
					w.Write([]byte(modelInfoBody("")))
					return
				}
				w.Write([]byte(modelInfoDataEnvelope("")))
			}))
			defer srv.Close()

			res := resourceLiteLLMModel()
			attrs := map[string]string{
				"model_name":          "sonnet-4-5-anthropic",
				"custom_llm_provider": "anthropic",
				"base_model":          "claude-sonnet-4-5",
			}
			if tc.oldRate != "" {
				attrs["input_cost_per_reference_pixel"] = tc.oldRate
			}
			state := &terraform.InstanceState{ID: "model-123", Attributes: attrs}
			config := map[string]interface{}{
				"model_name":          "sonnet-4-5-anthropic",
				"custom_llm_provider": "anthropic",
				"base_model":          "claude-sonnet-4-5",
			}
			if tc.newRate != "0" {
				config["input_cost_per_reference_pixel"] = tc.newRate
			}
			diff, err := res.Diff(context.Background(), state, &terraform.ResourceConfig{Config: config}, nil)
			if err != nil {
				t.Fatalf("diff failed: %v", err)
			}
			d, err := schema.InternalMap(res.Schema).Data(state, diff)
			if err != nil {
				t.Fatalf("data failed: %v", err)
			}

			if err := resourceLiteLLMModelUpdate(d, NewClient(srv.URL, "test-key", true)); err != nil {
				t.Fatalf("update failed: %v", err)
			}
			rate, sent := updateParams["input_cost_per_reference_pixel"]
			if sent != tc.wantSent {
				t.Fatalf("expected input_cost_per_reference_pixel sent=%v, got %v (%v)", tc.wantSent, sent, rate)
			}
			if sent && rate != tc.wantRate {
				t.Errorf("expected input_cost_per_reference_pixel %v, got %v", tc.wantRate, rate)
			}
		})
	}
}
