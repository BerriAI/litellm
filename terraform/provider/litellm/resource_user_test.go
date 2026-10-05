package litellm

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func userInfoBody(userID string, info map[string]interface{}) []byte {
	body, _ := json.Marshal(map[string]interface{}{
		"user_id":   userID,
		"user_info": info,
	})
	return body
}

func TestResourceUserCreate_SendsPayloadAndSetsID(t *testing.T) {
	var createPayload map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/user/new":
			if r.Method != http.MethodPost {
				t.Errorf("expected POST /user/new, got %s", r.Method)
			}
			if err := json.NewDecoder(r.Body).Decode(&createPayload); err != nil {
				t.Fatalf("failed to decode create payload: %v", err)
			}
			w.Write([]byte(`{"user_id": "u-123", "key": "sk-generated"}`))
		case "/user/info":
			if got := r.URL.Query().Get("user_id"); got != "u-123" {
				t.Errorf("expected user_id query 'u-123', got %q", got)
			}
			w.Write(userInfoBody("u-123", map[string]interface{}{
				"user_email": "alice@example.com",
				"user_role":  "internal_user",
				"max_budget": 50.5,
				"tpm_limit":  float64(1000),
				"teams":      []interface{}{"team-1"},
			}))
		default:
			t.Errorf("unexpected request to %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resourceLiteLLMUser().Schema, map[string]interface{}{
		"user_email":       "alice@example.com",
		"user_role":        "internal_user",
		"max_budget":       50.5,
		"tpm_limit":        1000,
		"auto_create_key":  true,
		"teams":            []interface{}{"team-1"},
		"model_max_budget": `{"gpt-4o": {"max_budget": 10.0}}`,
	})

	if err := resourceLiteLLMUserCreate(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("create failed: %v", err)
	}

	if d.Id() != "u-123" {
		t.Fatalf("expected ID 'u-123', got %q", d.Id())
	}
	if got := d.Get("key").(string); got != "sk-generated" {
		t.Fatalf("expected key 'sk-generated', got %q", got)
	}
	if got := createPayload["user_email"]; got != "alice@example.com" {
		t.Errorf("expected user_email in payload, got %v", got)
	}
	if got := createPayload["user_role"]; got != "internal_user" {
		t.Errorf("expected user_role in payload, got %v", got)
	}
	if got := createPayload["max_budget"]; got != 50.5 {
		t.Errorf("expected max_budget 50.5 in payload, got %v", got)
	}
	if got := createPayload["auto_create_key"]; got != true {
		t.Errorf("expected auto_create_key true in payload, got %v", got)
	}
	mmb, ok := createPayload["model_max_budget"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected model_max_budget object in payload, got %v", createPayload["model_max_budget"])
	}
	if _, ok := mmb["gpt-4o"]; !ok {
		t.Errorf("expected gpt-4o key in model_max_budget, got %v", mmb)
	}
	if got := d.Get("user_email").(string); got != "alice@example.com" {
		t.Errorf("expected user_email in state, got %q", got)
	}
}

func TestResourceUserRead_MapsFields(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(userInfoBody("u-42", map[string]interface{}{
			"user_email":       "bob@example.com",
			"user_alias":       "bob",
			"user_role":        "proxy_admin",
			"max_budget":       100.0,
			"budget_duration":  "30d",
			"tpm_limit":        float64(5000),
			"rpm_limit":        float64(60),
			"teams":            []interface{}{"team-a", "team-b"},
			"models":           []interface{}{"gpt-4o"},
			"model_max_budget": map[string]interface{}{"gpt-4o": map[string]interface{}{"max_budget": 5.0}},
		}))
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resourceLiteLLMUser().Schema, map[string]interface{}{})
	d.SetId("u-42")

	if err := resourceLiteLLMUserRead(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("read failed: %v", err)
	}

	if got := d.Get("user_email").(string); got != "bob@example.com" {
		t.Errorf("expected user_email 'bob@example.com', got %q", got)
	}
	if got := d.Get("user_alias").(string); got != "bob" {
		t.Errorf("expected user_alias 'bob', got %q", got)
	}
	if got := d.Get("user_role").(string); got != "proxy_admin" {
		t.Errorf("expected user_role 'proxy_admin', got %q", got)
	}
	if got := d.Get("max_budget").(float64); got != 100.0 {
		t.Errorf("expected max_budget 100.0, got %v", got)
	}
	if got := d.Get("budget_duration").(string); got != "30d" {
		t.Errorf("expected budget_duration '30d', got %q", got)
	}
	if got := d.Get("tpm_limit").(int); got != 5000 {
		t.Errorf("expected tpm_limit 5000, got %d", got)
	}
	if got := d.Get("rpm_limit").(int); got != 60 {
		t.Errorf("expected rpm_limit 60, got %d", got)
	}
	teams := d.Get("teams").([]interface{})
	if len(teams) != 2 || teams[0] != "team-a" {
		t.Errorf("expected teams [team-a team-b], got %v", teams)
	}
	var mmb map[string]interface{}
	if err := json.Unmarshal([]byte(d.Get("model_max_budget").(string)), &mmb); err != nil {
		t.Fatalf("model_max_budget in state is not valid JSON: %v", err)
	}
	if _, ok := mmb["gpt-4o"]; !ok {
		t.Errorf("expected gpt-4o key in model_max_budget state, got %v", mmb)
	}
}

func TestResourceUserRead_404ClearsID(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resourceLiteLLMUser().Schema, map[string]interface{}{})
	d.SetId("gone-user")

	if err := resourceLiteLLMUserRead(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("expected nil error on 404, got: %v", err)
	}
	if d.Id() != "" {
		t.Fatalf("expected ID to be cleared on 404, got %q", d.Id())
	}
}

func TestResourceUserUpdate_SendsPayload(t *testing.T) {
	var updatePayload map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/user/update":
			if r.Method != http.MethodPost {
				t.Errorf("expected POST /user/update, got %s", r.Method)
			}
			if err := json.NewDecoder(r.Body).Decode(&updatePayload); err != nil {
				t.Fatalf("failed to decode update payload: %v", err)
			}
			w.Write([]byte(`{"user_id": "u-7"}`))
		case "/user/info":
			w.Write(userInfoBody("u-7", map[string]interface{}{"user_role": "internal_user_viewer"}))
		default:
			t.Errorf("unexpected request to %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resourceLiteLLMUser().Schema, map[string]interface{}{
		"user_role":  "internal_user_viewer",
		"max_budget": 25.0,
	})
	d.SetId("u-7")

	if err := resourceLiteLLMUserUpdate(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("update failed: %v", err)
	}

	if got := updatePayload["user_id"]; got != "u-7" {
		t.Errorf("expected user_id 'u-7' in payload, got %v", got)
	}
	if got := updatePayload["user_role"]; got != "internal_user_viewer" {
		t.Errorf("expected user_role in payload, got %v", got)
	}
	if got := updatePayload["max_budget"]; got != 25.0 {
		t.Errorf("expected max_budget 25.0 in payload, got %v", got)
	}
	if _, ok := updatePayload["auto_create_key"]; ok {
		t.Errorf("auto_create_key must not be sent on update, got %v", updatePayload["auto_create_key"])
	}
}

func TestResourceUserDelete_SendsUserIDs(t *testing.T) {
	var deletePayload map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/user/delete" || r.Method != http.MethodPost {
			t.Errorf("expected POST /user/delete, got %s %s", r.Method, r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&deletePayload); err != nil {
			t.Fatalf("failed to decode delete payload: %v", err)
		}
		w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resourceLiteLLMUser().Schema, map[string]interface{}{})
	d.SetId("u-del")

	if err := resourceLiteLLMUserDelete(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("delete failed: %v", err)
	}

	ids, ok := deletePayload["user_ids"].([]interface{})
	if !ok || len(ids) != 1 || ids[0] != "u-del" {
		t.Fatalf("expected user_ids ['u-del'], got %v", deletePayload["user_ids"])
	}
	if d.Id() != "" {
		t.Fatalf("expected ID to be cleared after delete, got %q", d.Id())
	}
}
