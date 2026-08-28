package litellm

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func TestTeamMemberAddCreateSendsMemberSettings(t *testing.T) {
	var addPayload map[string]interface{}
	var updatePayloads []map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var payload map[string]interface{}
		json.Unmarshal(body, &payload)
		switch r.URL.Path {
		case "/team/member_add":
			addPayload = payload
		case "/team/member_update":
			updatePayloads = append(updatePayloads, payload)
		default:
			t.Errorf("unexpected request path: %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, resourceLiteLLMTeamMemberAdd().Schema, map[string]interface{}{
		"team_id": "team-1",
		"member": []interface{}{
			map[string]interface{}{
				"user_id": "user-1",
				"role":    "user",
			},
		},
		"max_budget_in_team": 25.0,
		"tpm_limit":          1000,
		"rpm_limit":          10,
		"budget_duration":    "30d",
		"allowed_models":     []interface{}{"claude-opus-4-6-v1"},
	})

	if err := resourceLiteLLMTeamMemberAddCreate(d, client); err != nil {
		t.Fatalf("create failed: %v", err)
	}

	if addPayload["budget_duration"] != "30d" {
		t.Fatalf("member_add payload sent budget_duration %v, want 30d", addPayload["budget_duration"])
	}
	wantModels := []interface{}{"claude-opus-4-6-v1"}
	if !reflect.DeepEqual(addPayload["allowed_models"], wantModels) {
		t.Fatalf("member_add payload sent allowed_models %v, want %v", addPayload["allowed_models"], wantModels)
	}
	if _, ok := addPayload["tpm_limit"]; ok {
		t.Fatalf("member_add payload must not carry tpm_limit, got %v", addPayload["tpm_limit"])
	}

	if len(updatePayloads) != 1 {
		t.Fatalf("expected 1 member_update call for limits, got %d", len(updatePayloads))
	}
	update := updatePayloads[0]
	if update["tpm_limit"] != float64(1000) {
		t.Fatalf("member_update payload sent tpm_limit %v, want 1000", update["tpm_limit"])
	}
	if update["rpm_limit"] != float64(10) {
		t.Fatalf("member_update payload sent rpm_limit %v, want 10", update["rpm_limit"])
	}
	if update["user_id"] != "user-1" {
		t.Fatalf("member_update payload sent user_id %v, want user-1", update["user_id"])
	}
}

func TestTeamMemberAddCreateOmitsUnsetSettings(t *testing.T) {
	var addPayload map[string]interface{}
	updateCalls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		switch r.URL.Path {
		case "/team/member_add":
			json.Unmarshal(body, &addPayload)
		case "/team/member_update":
			updateCalls++
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, resourceLiteLLMTeamMemberAdd().Schema, map[string]interface{}{
		"team_id": "team-1",
		"member": []interface{}{
			map[string]interface{}{
				"user_id": "user-1",
				"role":    "user",
			},
		},
	})

	if err := resourceLiteLLMTeamMemberAddCreate(d, client); err != nil {
		t.Fatalf("create failed: %v", err)
	}

	for _, field := range []string{"tpm_limit", "rpm_limit", "budget_duration", "allowed_models"} {
		if _, ok := addPayload[field]; ok {
			t.Fatalf("member_add payload must not carry unset %s, got %v", field, addPayload[field])
		}
	}
	if updateCalls != 0 {
		t.Fatalf("expected no member_update calls without limits, got %d", updateCalls)
	}
}

func TestApplyUpdateSettingsCarriesAllFields(t *testing.T) {
	d := schema.TestResourceDataRaw(t, resourceLiteLLMTeamMemberAdd().Schema, map[string]interface{}{
		"team_id": "team-1",
		"member": []interface{}{
			map[string]interface{}{
				"user_id": "user-1",
				"role":    "user",
			},
		},
		"tpm_limit":       500,
		"rpm_limit":       5,
		"budget_duration": "7d",
		"allowed_models":  []interface{}{"gpt-5.2", "claude-opus-4-6-v1"},
	})

	payload := map[string]interface{}{"team_id": "team-1"}
	applyUpdateSettings(d, payload)

	if payload["tpm_limit"] != 500 || payload["rpm_limit"] != 5 {
		t.Fatalf("payload limits = %v/%v, want 500/5", payload["tpm_limit"], payload["rpm_limit"])
	}
	if payload["budget_duration"] != "7d" {
		t.Fatalf("payload budget_duration = %v, want 7d", payload["budget_duration"])
	}
	wantModels := []string{"gpt-5.2", "claude-opus-4-6-v1"}
	if !reflect.DeepEqual(payload["allowed_models"], wantModels) {
		t.Fatalf("payload allowed_models = %v, want %v", payload["allowed_models"], wantModels)
	}
}
