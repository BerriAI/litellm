package litellm

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
	"github.com/hashicorp/terraform-plugin-sdk/v2/terraform"
)

func TestTeamMemberUpdateSendsRole(t *testing.T) {
	var captured map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &captured)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, resourceLiteLLMTeamMember().Schema, map[string]interface{}{
		"team_id":    "team-1",
		"user_id":    "user-1",
		"user_email": "user@example.com",
		"role":       "user",
	})
	d.SetId("team-1:user-1")

	if err := resourceLiteLLMTeamMemberUpdate(d, client); err != nil {
		t.Fatalf("update failed: %v", err)
	}

	role, ok := captured["role"]
	if !ok {
		t.Fatalf("update payload missing role field: %v", captured)
	}
	if role != "user" {
		t.Fatalf("update payload sent role %v, want user", role)
	}
}

func teamMemberResourceData(t *testing.T, extra map[string]interface{}) *schema.ResourceData {
	t.Helper()
	raw := map[string]interface{}{
		"team_id":    "team-1",
		"user_id":    "user-1",
		"user_email": "user@example.com",
		"role":       "user",
	}
	for k, v := range extra {
		raw[k] = v
	}
	return schema.TestResourceDataRaw(t, resourceLiteLLMTeamMember().Schema, raw)
}

func newTeamMemberDiffResourceData(t *testing.T, old, new map[string]string) *schema.ResourceData {
	t.Helper()
	attrs := map[string]string{
		"team_id":    "team-1",
		"user_id":    "user-1",
		"user_email": "user@example.com",
		"role":       "user",
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
	state := &terraform.InstanceState{ID: "team-1:user-1", Attributes: attrs}
	d, err := schema.InternalMap(resourceLiteLLMTeamMember().Schema).Data(state, &terraform.InstanceDiff{Attributes: diffAttrs})
	if err != nil {
		t.Fatalf("building ResourceData returned error: %v", err)
	}
	return d
}

func captureTeamMemberPayload(t *testing.T, d *schema.ResourceData, apply func(*schema.ResourceData, interface{}) error) map[string]interface{} {
	t.Helper()
	var captured map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &captured)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	if err := apply(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("request failed: %v", err)
	}
	return captured
}

func TestTeamMemberCreateOmitsUnsetMaxBudget(t *testing.T) {
	payload := captureTeamMemberPayload(t, teamMemberResourceData(t, nil), resourceLiteLLMTeamMemberCreate)
	if v, ok := payload["max_budget_in_team"]; ok {
		t.Fatalf("member_add payload must not carry an unset max_budget_in_team, got %v", v)
	}
}

func TestTeamMemberCreateSendsMaxBudget(t *testing.T) {
	d := teamMemberResourceData(t, map[string]interface{}{"max_budget_in_team": 25.0})
	payload := captureTeamMemberPayload(t, d, resourceLiteLLMTeamMemberCreate)
	if payload["max_budget_in_team"] != float64(25) {
		t.Fatalf("member_add payload sent max_budget_in_team %v, want 25", payload["max_budget_in_team"])
	}
}

func TestTeamMemberUpdateOmitsUnsetMaxBudget(t *testing.T) {
	payload := captureTeamMemberPayload(t, teamMemberResourceData(t, nil), resourceLiteLLMTeamMemberUpdate)
	if v, ok := payload["max_budget_in_team"]; ok {
		t.Fatalf("member_update payload must not carry an unset max_budget_in_team, got %v", v)
	}
}

func TestTeamMemberUpdateClearsRemovedMaxBudget(t *testing.T) {
	d := newTeamMemberDiffResourceData(t, map[string]string{"max_budget_in_team": "25"}, nil)
	payload := captureTeamMemberPayload(t, d, resourceLiteLLMTeamMemberUpdate)
	v, ok := payload["max_budget_in_team"]
	if !ok {
		t.Fatal("member_update payload omitted removed max_budget_in_team, so the proxy would keep the old budget")
	}
	if v != nil {
		t.Fatalf("member_update payload max_budget_in_team = %v, want explicit null", v)
	}
}
