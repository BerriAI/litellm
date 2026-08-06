package litellm

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func newGuardrailResourceData(t *testing.T, raw map[string]interface{}) *schema.ResourceData {
	t.Helper()
	return schema.TestResourceDataRaw(t, resourceLiteLLMGuardrail().Schema, raw)
}

func TestBuildGuardrailSpecMergesParams(t *testing.T) {
	d := newGuardrailResourceData(t, map[string]interface{}{
		"guardrail_name": "bedrock-guard",
		"guardrail":      "bedrock",
		"mode":           "pre_call",
		"default_on":     true,
		"litellm_params": `{"guardrailIdentifier": "ff6ujrregl1q", "guardrailVersion": "DRAFT"}`,
		"guardrail_info": `{"description": "content moderation"}`,
	})

	spec, err := buildGuardrailSpec(d)
	if err != nil {
		t.Fatalf("buildGuardrailSpec failed: %v", err)
	}

	if spec.GuardrailName != "bedrock-guard" {
		t.Fatalf("unexpected guardrail_name: %q", spec.GuardrailName)
	}
	if spec.LiteLLMParams["guardrail"] != "bedrock" {
		t.Fatalf("guardrail type not set in litellm_params: %v", spec.LiteLLMParams)
	}
	if spec.LiteLLMParams["mode"] != "pre_call" {
		t.Fatalf("mode not set in litellm_params: %v", spec.LiteLLMParams["mode"])
	}
	if spec.LiteLLMParams["default_on"] != true {
		t.Fatalf("default_on not merged: %v", spec.LiteLLMParams["default_on"])
	}
	if spec.LiteLLMParams["guardrailIdentifier"] != "ff6ujrregl1q" {
		t.Fatalf("extra litellm_params not merged: %v", spec.LiteLLMParams)
	}
	if spec.GuardrailInfo["description"] != "content moderation" {
		t.Fatalf("guardrail_info not parsed: %v", spec.GuardrailInfo)
	}
}

func TestBuildGuardrailSpecParsesModeArray(t *testing.T) {
	d := newGuardrailResourceData(t, map[string]interface{}{
		"guardrail_name": "multi-mode",
		"guardrail":      "presidio",
		"mode":           `["pre_call", "post_call"]`,
	})

	spec, err := buildGuardrailSpec(d)
	if err != nil {
		t.Fatalf("buildGuardrailSpec failed: %v", err)
	}

	modes, ok := spec.LiteLLMParams["mode"].([]string)
	if !ok {
		t.Fatalf("mode was not parsed into a slice: %T %v", spec.LiteLLMParams["mode"], spec.LiteLLMParams["mode"])
	}
	if len(modes) != 2 || modes[0] != "pre_call" || modes[1] != "post_call" {
		t.Fatalf("unexpected parsed modes: %v", modes)
	}
}

func TestBuildGuardrailSpecRejectsInvalidJSON(t *testing.T) {
	d := newGuardrailResourceData(t, map[string]interface{}{
		"guardrail_name": "bad",
		"guardrail":      "bedrock",
		"mode":           "pre_call",
		"litellm_params": `not-json`,
	})

	if _, err := buildGuardrailSpec(d); err == nil {
		t.Fatal("expected error for invalid litellm_params JSON, got nil")
	}
}

func TestGuardrailCreateSendsWrappedRequestAndSetsID(t *testing.T) {
	var captured GuardrailRequest

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/guardrails":
			body, _ := io.ReadAll(r.Body)
			if err := json.Unmarshal(body, &captured); err != nil {
				t.Errorf("failed to decode create request: %v", err)
			}
			json.NewEncoder(w).Encode(GuardrailResponse{
				GuardrailID:   "guard-123",
				GuardrailName: "bedrock-guard",
				LiteLLMParams: map[string]interface{}{"guardrail": "bedrock", "mode": "pre_call", "default_on": true},
			})
		case r.Method == http.MethodGet && r.URL.Path == "/guardrails/guard-123/info":
			json.NewEncoder(w).Encode(GuardrailResponse{
				GuardrailID:   "guard-123",
				GuardrailName: "bedrock-guard",
				LiteLLMParams: map[string]interface{}{"guardrail": "bedrock", "mode": "pre_call", "default_on": true},
				CreatedAt:     "2026-01-01T00:00:00Z",
			})
		default:
			t.Errorf("unexpected request %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newGuardrailResourceData(t, map[string]interface{}{
		"guardrail_name": "bedrock-guard",
		"guardrail":      "bedrock",
		"mode":           "pre_call",
		"default_on":     true,
	})

	if err := resourceLiteLLMGuardrailCreate(d, client); err != nil {
		t.Fatalf("create failed: %v", err)
	}

	if d.Id() != "guard-123" {
		t.Fatalf("resource ID not set from response: %q", d.Id())
	}
	if captured.Guardrail.GuardrailName != "bedrock-guard" {
		t.Fatalf("request was not wrapped under \"guardrail\": %+v", captured)
	}
	if captured.Guardrail.LiteLLMParams["guardrail"] != "bedrock" {
		t.Fatalf("guardrail type not sent: %v", captured.Guardrail.LiteLLMParams)
	}
	if d.Get("created_at").(string) != "2026-01-01T00:00:00Z" {
		t.Fatalf("created_at not populated from read-back: %q", d.Get("created_at"))
	}
}

func TestGuardrailReadDoesNotPersistMaskedParams(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/guardrails/guard-123/info" {
			t.Errorf("unexpected request %s %s", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(GuardrailResponse{
			GuardrailID:   "guard-123",
			GuardrailName: "bedrock-guard",
			LiteLLMParams: map[string]interface{}{
				"guardrail":  "bedrock",
				"mode":       "pre_call",
				"default_on": true,
				"api_key":    "sk-1****",
			},
			GuardrailInfo: map[string]interface{}{"description": "content moderation"},
			CreatedAt:     "2026-01-01T00:00:00Z",
			UpdatedAt:     "2026-01-02T00:00:00Z",
		})
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newGuardrailResourceData(t, map[string]interface{}{
		"guardrail_name": "bedrock-guard",
		"guardrail":      "bedrock",
		"mode":           "pre_call",
		"litellm_params": `{"api_key": "sk-secret-value"}`,
	})
	d.SetId("guard-123")

	if err := resourceLiteLLMGuardrailRead(d, client); err != nil {
		t.Fatalf("read failed: %v", err)
	}

	if got := d.Get("litellm_params").(string); got != `{"api_key": "sk-secret-value"}` {
		t.Fatalf("configured litellm_params was overwritten with masked server value: %q", got)
	}
	if d.Get("guardrail").(string) != "bedrock" {
		t.Fatalf("guardrail not reconciled from read: %q", d.Get("guardrail"))
	}
	if d.Get("default_on").(bool) != true {
		t.Fatalf("default_on not reconciled from read")
	}
	if d.Get("updated_at").(string) != "2026-01-02T00:00:00Z" {
		t.Fatalf("updated_at not populated: %q", d.Get("updated_at"))
	}
	if d.Get("guardrail_info").(string) != `{"description":"content moderation"}` {
		t.Fatalf("guardrail_info not reconciled: %q", d.Get("guardrail_info"))
	}
}

func TestGuardrailReadModeArrayRoundTrips(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(GuardrailResponse{
			GuardrailID:   "guard-9",
			GuardrailName: "multi",
			LiteLLMParams: map[string]interface{}{
				"guardrail": "presidio",
				"mode":      []interface{}{"pre_call", "post_call"},
			},
		})
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newGuardrailResourceData(t, map[string]interface{}{
		"guardrail_name": "multi",
		"guardrail":      "presidio",
		"mode":           `["pre_call", "post_call"]`,
	})
	d.SetId("guard-9")

	if err := resourceLiteLLMGuardrailRead(d, client); err != nil {
		t.Fatalf("read failed: %v", err)
	}

	if got := d.Get("mode").(string); got != `["pre_call","post_call"]` {
		t.Fatalf("mode array did not round-trip: %q", got)
	}
}

func TestGuardrailReadRemovesResourceOn404(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newGuardrailResourceData(t, map[string]interface{}{
		"guardrail_name": "gone",
		"guardrail":      "bedrock",
		"mode":           "pre_call",
	})
	d.SetId("guard-missing")

	if err := resourceLiteLLMGuardrailRead(d, client); err != nil {
		t.Fatalf("read of missing guardrail should not error: %v", err)
	}
	if d.Id() != "" {
		t.Fatalf("expected resource to be removed from state, still has ID %q", d.Id())
	}
}

func TestSuppressEquivalentJSON(t *testing.T) {
	cases := []struct {
		name           string
		oldV, newV     string
		wantSuppressed bool
	}{
		{"reordered keys", `{"a":1,"b":2}`, `{"b":2,"a":1}`, true},
		{"whitespace", `{"a":1}`, `{ "a": 1 }`, true},
		{"different values", `{"a":1}`, `{"a":2}`, false},
		{"invalid json", `{"a":1}`, `not-json`, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := suppressEquivalentJSON("", tc.oldV, tc.newV, nil); got != tc.wantSuppressed {
				t.Fatalf("suppressEquivalentJSON(%q,%q)=%v want %v", tc.oldV, tc.newV, got, tc.wantSuppressed)
			}
		})
	}
}
