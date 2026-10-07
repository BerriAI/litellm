package litellm

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
	"github.com/hashicorp/terraform-plugin-sdk/v2/terraform"
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

func TestTeamMemberAddCreateSetsIDBeforeLimitsFail(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/team/member_update" {
			w.WriteHeader(http.StatusInternalServerError)
			w.Write([]byte(`{"error":"boom"}`))
			return
		}
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
		"tpm_limit": 1000,
	})

	if err := resourceLiteLLMTeamMemberAddCreate(d, client); err == nil {
		t.Fatal("create should fail when member_update fails")
	}
	if d.Id() != "team-1" {
		t.Fatalf("resource ID = %q after failed limits call, want team-1 so Terraform can taint and recreate it", d.Id())
	}
}

// newTeamMemberUpdateResourceData builds a ResourceData with one member in state
// and a real old -> new diff on the scalar settings, so d.HasChange and d.GetOk
// behave as they do during a real Update call
func newTeamMemberUpdateResourceData(t *testing.T, old, new map[string]string) *schema.ResourceData {
	t.Helper()
	attrs := map[string]string{
		"team_id":             "team-1",
		"member.#":            "1",
		"member.1.user_id":    "user-1",
		"member.1.user_email": "",
		"member.1.role":       "user",
		"allowed_models.#":    "0",
		"max_budget_in_team":  "25",
	}
	for k, v := range old {
		attrs[k] = v
	}
	diffAttrs := map[string]*terraform.ResourceAttrDiff{}
	for k, v := range new {
		diffAttrs[k] = &terraform.ResourceAttrDiff{Old: attrs[k], New: v}
	}
	for k := range old {
		if _, ok := new[k]; !ok {
			diffAttrs[k] = &terraform.ResourceAttrDiff{Old: attrs[k], New: "", NewRemoved: true}
		}
	}
	state := &terraform.InstanceState{ID: "team-1", Attributes: attrs}
	d, err := schema.InternalMap(resourceLiteLLMTeamMemberAdd().Schema).Data(state, &terraform.InstanceDiff{Attributes: diffAttrs})
	if err != nil {
		t.Fatalf("building ResourceData returned error: %v", err)
	}
	return d
}

func runTeamMemberUpdate(t *testing.T, d *schema.ResourceData) []map[string]interface{} {
	t.Helper()
	var updatePayloads []map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/team/member_update" {
			t.Errorf("unexpected request path: %s", r.URL.Path)
		}
		body, _ := io.ReadAll(r.Body)
		var payload map[string]interface{}
		json.Unmarshal(body, &payload)
		updatePayloads = append(updatePayloads, payload)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	if err := resourceLiteLLMTeamMemberAddUpdate(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("update failed: %v", err)
	}
	if len(updatePayloads) != 1 {
		t.Fatalf("expected 1 member_update call, got %d", len(updatePayloads))
	}
	return updatePayloads
}

func TestTeamMemberAddUpdateSendsChangedSettings(t *testing.T) {
	d := newTeamMemberUpdateResourceData(t,
		map[string]string{"tpm_limit": "1000", "rpm_limit": "10", "budget_duration": "30d"},
		map[string]string{"tpm_limit": "500", "rpm_limit": "5", "budget_duration": "7d", "allowed_models.#": "1", "allowed_models.0": "gpt-5.2"},
	)

	update := runTeamMemberUpdate(t, d)[0]
	if update["tpm_limit"] != float64(500) || update["rpm_limit"] != float64(5) {
		t.Fatalf("member_update payload limits = %v/%v, want 500/5", update["tpm_limit"], update["rpm_limit"])
	}
	if update["budget_duration"] != "7d" {
		t.Fatalf("member_update payload budget_duration = %v, want 7d", update["budget_duration"])
	}
	if !reflect.DeepEqual(update["allowed_models"], []interface{}{"gpt-5.2"}) {
		t.Fatalf("member_update payload allowed_models = %v, want [gpt-5.2]", update["allowed_models"])
	}
	if update["user_id"] != "user-1" {
		t.Fatalf("member_update payload user_id = %v, want user-1", update["user_id"])
	}
}

func TestTeamMemberAddUpdateClearsRemovedSettings(t *testing.T) {
	d := newTeamMemberUpdateResourceData(t,
		map[string]string{"tpm_limit": "1000", "rpm_limit": "10", "budget_duration": "30d", "allowed_models.#": "1", "allowed_models.0": "gpt-5.2"},
		map[string]string{"allowed_models.#": "0"},
	)

	update := runTeamMemberUpdate(t, d)[0]
	for _, field := range []string{"tpm_limit", "rpm_limit", "budget_duration"} {
		v, present := update[field]
		if !present {
			t.Fatalf("member_update payload omitted removed %s, so the proxy would keep the old value", field)
		}
		if v != nil {
			t.Fatalf("member_update payload %s = %v, want explicit null", field, v)
		}
	}
	if !reflect.DeepEqual(update["allowed_models"], []interface{}{}) {
		t.Fatalf("member_update payload allowed_models = %v, want empty list", update["allowed_models"])
	}
}

func TestTeamMemberAddUpdateLeavesUnchangedSettingsAlone(t *testing.T) {
	d := newTeamMemberUpdateResourceData(t,
		map[string]string{"budget_duration": "30d"},
		map[string]string{"budget_duration": "7d"},
	)

	update := runTeamMemberUpdate(t, d)[0]
	for _, field := range []string{"tpm_limit", "rpm_limit"} {
		if v, present := update[field]; present {
			t.Fatalf("member_update payload must not touch never-set %s, got %v", field, v)
		}
	}
	if _, present := update["allowed_models"]; present {
		t.Fatalf("member_update payload must not touch unchanged allowed_models, got %v", update["allowed_models"])
	}
}
