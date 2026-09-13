package litellm

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func newGuardrailTestData(t *testing.T, raw map[string]interface{}) *schema.ResourceData {
	t.Helper()
	return schema.TestResourceDataRaw(t, resourceLiteLLMGuardrail().Schema, raw)
}

func guardrailInfoJSON(id, name string) string {
	body, _ := json.Marshal(map[string]interface{}{
		"guardrail_id":   id,
		"guardrail_name": name,
		"guardrail_info": map[string]interface{}{"description": "test guardrail"},
		"created_at":     "2026-01-01T00:00:00Z",
		"updated_at":     "2026-01-02T00:00:00Z",
	})
	return string(body)
}

func TestGuardrailCreate_SendsPayloadAndSetsID(t *testing.T) {
	var createPayload map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == "POST" && r.URL.Path == "/guardrails":
			if err := json.NewDecoder(r.Body).Decode(&createPayload); err != nil {
				t.Errorf("failed to decode create payload: %v", err)
			}
			w.Write([]byte(guardrailInfoJSON("gid-123", "guard1")))
		case r.Method == "GET" && r.URL.Path == "/guardrails/gid-123/info":
			w.Write([]byte(guardrailInfoJSON("gid-123", "guard1")))
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newGuardrailTestData(t, map[string]interface{}{
		"guardrail_name": "guard1",
		"guardrail":      "bedrock",
		"mode":           "pre_call",
		"default_on":     true,
		"litellm_params": `{"api_key": "sk-123", "guardrailIdentifier": "abc"}`,
		"guardrail_info": map[string]interface{}{"description": "test guardrail"},
	})

	if err := resourceLiteLLMGuardrailCreate(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if d.Id() != "gid-123" {
		t.Fatalf("expected ID 'gid-123', got %q", d.Id())
	}

	guardrail, ok := createPayload["guardrail"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected payload wrapped in 'guardrail' key, got: %v", createPayload)
	}
	if guardrail["guardrail_name"] != "guard1" {
		t.Errorf("expected guardrail_name 'guard1', got %v", guardrail["guardrail_name"])
	}
	params, ok := guardrail["litellm_params"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected litellm_params object, got: %v", guardrail["litellm_params"])
	}
	if params["guardrail"] != "bedrock" || params["mode"] != "pre_call" || params["default_on"] != true {
		t.Errorf("unexpected base litellm_params: %v", params)
	}
	if params["api_key"] != "sk-123" || params["guardrailIdentifier"] != "abc" {
		t.Errorf("expected merged extra litellm_params, got: %v", params)
	}
	info, ok := guardrail["guardrail_info"].(map[string]interface{})
	if !ok || info["description"] != "test guardrail" {
		t.Errorf("expected guardrail_info to be sent, got: %v", guardrail["guardrail_info"])
	}
}

func TestGuardrailCreate_ModeJSONArray(t *testing.T) {
	var createPayload map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method == "POST" {
			json.NewDecoder(r.Body).Decode(&createPayload)
		}
		w.Write([]byte(guardrailInfoJSON("gid-456", "guard2")))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newGuardrailTestData(t, map[string]interface{}{
		"guardrail_name": "guard2",
		"guardrail":      "lakera",
		"mode":           `["pre_call", "post_call"]`,
	})

	if err := resourceLiteLLMGuardrailCreate(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}

	params := createPayload["guardrail"].(map[string]interface{})["litellm_params"].(map[string]interface{})
	mode, ok := params["mode"].([]interface{})
	if !ok {
		t.Fatalf("expected mode to be a JSON array, got: %v", params["mode"])
	}
	if !reflect.DeepEqual(mode, []interface{}{"pre_call", "post_call"}) {
		t.Errorf("unexpected mode array: %v", mode)
	}
}

func TestGuardrailCreate_InvalidLitellmParamsJSON(t *testing.T) {
	client := NewClient("http://unused.invalid", "test-key", true)
	d := newGuardrailTestData(t, map[string]interface{}{
		"guardrail_name": "guard1",
		"guardrail":      "bedrock",
		"mode":           "pre_call",
		"litellm_params": "{not json",
	})

	if err := resourceLiteLLMGuardrailCreate(d, client); err == nil {
		t.Fatal("expected error for invalid litellm_params JSON, got nil")
	}
}

func TestGuardrailRead_MapsFieldsAndKeepsConfiguredParams(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" || r.URL.Path != "/guardrails/gid-1/info" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(guardrailInfoJSON("gid-1", "renamed-guard")))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newGuardrailTestData(t, map[string]interface{}{
		"guardrail_name": "old-name",
		"guardrail":      "bedrock",
		"mode":           "pre_call",
		"litellm_params": `{"api_key": "sk-123"}`,
	})
	d.SetId("gid-1")

	if err := resourceLiteLLMGuardrailRead(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if got := d.Get("guardrail_name").(string); got != "renamed-guard" {
		t.Errorf("expected guardrail_name 'renamed-guard', got %q", got)
	}
	if got := d.Get("created_at").(string); got != "2026-01-01T00:00:00Z" {
		t.Errorf("expected created_at to be set, got %q", got)
	}
	if got := d.Get("litellm_params").(string); got != `{"api_key": "sk-123"}` {
		t.Errorf("expected configured litellm_params to stay authoritative, got %q", got)
	}
	info := d.Get("guardrail_info").(map[string]interface{})
	if info["description"] != "test guardrail" {
		t.Errorf("expected guardrail_info from API, got: %v", info)
	}
}

func TestGuardrailRead_404ClearsID(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newGuardrailTestData(t, map[string]interface{}{
		"guardrail_name": "guard1",
		"guardrail":      "bedrock",
		"mode":           "pre_call",
	})
	d.SetId("gid-gone")

	if err := resourceLiteLLMGuardrailRead(d, client); err != nil {
		t.Fatalf("expected nil error on 404, got: %v", err)
	}
	if d.Id() != "" {
		t.Fatalf("expected ID to be cleared on 404, got %q", d.Id())
	}
}

func TestGuardrailUpdate_SendsPUTToGuardrailEndpoint(t *testing.T) {
	var updateMethod, updatePath string
	var updatePayload map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method == "PUT" {
			updateMethod, updatePath = r.Method, r.URL.Path
			json.NewDecoder(r.Body).Decode(&updatePayload)
		}
		w.Write([]byte(guardrailInfoJSON("gid-1", "new-name")))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newGuardrailTestData(t, map[string]interface{}{
		"guardrail_name": "new-name",
		"guardrail":      "bedrock",
		"mode":           "post_call",
	})
	d.SetId("gid-1")

	if err := resourceLiteLLMGuardrailUpdate(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if updateMethod != "PUT" || updatePath != "/guardrails/gid-1" {
		t.Fatalf("expected PUT /guardrails/gid-1, got %s %s", updateMethod, updatePath)
	}
	guardrail := updatePayload["guardrail"].(map[string]interface{})
	if guardrail["guardrail_name"] != "new-name" {
		t.Errorf("expected updated guardrail_name, got %v", guardrail["guardrail_name"])
	}
	if guardrail["guardrail_id"] != "gid-1" {
		t.Errorf("expected guardrail_id in update payload, got %v", guardrail["guardrail_id"])
	}
	params := guardrail["litellm_params"].(map[string]interface{})
	if params["mode"] != "post_call" {
		t.Errorf("expected updated mode 'post_call', got %v", params["mode"])
	}
}

func TestGuardrailDelete_CallsDeleteEndpoint(t *testing.T) {
	var deleteMethod, deletePath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		deleteMethod, deletePath = r.Method, r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"message": "deleted"}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newGuardrailTestData(t, map[string]interface{}{
		"guardrail_name": "guard1",
		"guardrail":      "bedrock",
		"mode":           "pre_call",
	})
	d.SetId("gid-1")

	if err := resourceLiteLLMGuardrailDelete(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if deleteMethod != "DELETE" || deletePath != "/guardrails/gid-1" {
		t.Fatalf("expected DELETE /guardrails/gid-1, got %s %s", deleteMethod, deletePath)
	}
	if d.Id() != "" {
		t.Fatalf("expected ID to be cleared after delete, got %q", d.Id())
	}
}

func TestGuardrailSuppressJSONDiff(t *testing.T) {
	if !guardrailSuppressJSONDiff("", `{"a": 1, "b": "x"}`, `{"b":"x","a":1}`, nil) {
		t.Error("expected semantically equal JSON to be suppressed")
	}
	if guardrailSuppressJSONDiff("", `{"a": 1}`, `{"a": 2}`, nil) {
		t.Error("expected different JSON not to be suppressed")
	}
	if guardrailSuppressJSONDiff("", "", `{"a": 1}`, nil) {
		t.Error("expected empty old value not to be suppressed")
	}
}
