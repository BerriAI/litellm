package litellm

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
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

func TestTeamMemberReadRefreshesRoleAndBudget(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/team/info" {
			t.Fatalf("unexpected request %s %s", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{
			"team_id":"team-1",
			"team_info":{"members_with_roles":[{"user_id":"user-1","user_email":"u@example.com","role":"admin"}]},
			"team_memberships":[{"user_id":"user-1","litellm_budget_table":{"max_budget":12.5}}]
		}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, resourceLiteLLMTeamMember().Schema, map[string]interface{}{
		"team_id":            "team-1",
		"user_id":            "user-1",
		"user_email":         "old@example.com",
		"role":               "user",
		"max_budget_in_team": 1.0,
	})
	d.SetId("team-1:user-1")

	if err := resourceLiteLLMTeamMemberRead(d, client); err != nil {
		t.Fatalf("read failed: %v", err)
	}
	if d.Id() == "" {
		t.Fatalf("expected member to remain in state")
	}
	if got := d.Get("role").(string); got != "admin" {
		t.Fatalf("role=%q want admin", got)
	}
	if got := d.Get("user_email").(string); got != "u@example.com" {
		t.Fatalf("user_email=%q want u@example.com", got)
	}
	if got := d.Get("max_budget_in_team").(float64); got != 12.5 {
		t.Fatalf("max_budget_in_team=%v want 12.5", got)
	}
}

func TestTeamMemberReadRemovesMissingMember(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"team_id":"team-1","team_info":{"members_with_roles":[]},"team_memberships":[]}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, resourceLiteLLMTeamMember().Schema, map[string]interface{}{
		"team_id":    "team-1",
		"user_id":    "user-gone",
		"user_email": "gone@example.com",
		"role":       "user",
	})
	d.SetId("team-1:user-gone")

	if err := resourceLiteLLMTeamMemberRead(d, client); err != nil {
		t.Fatalf("read failed: %v", err)
	}
	if d.Id() != "" {
		t.Fatalf("expected member removed from state, id=%q", d.Id())
	}
}

func TestTeamMemberDeleteTreatsAlreadyGoneAsSuccess(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`{"detail":{"error":"User not found in team"}}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, resourceLiteLLMTeamMember().Schema, map[string]interface{}{
		"team_id":    "team-1",
		"user_id":    "user-1",
		"user_email": "u@example.com",
		"role":       "user",
	})
	d.SetId("team-1:user-1")

	if err := resourceLiteLLMTeamMemberDelete(d, client); err != nil {
		t.Fatalf("delete failed: %v", err)
	}
	if d.Id() != "" {
		t.Fatalf("expected empty id after gone delete, id=%q", d.Id())
	}
}
