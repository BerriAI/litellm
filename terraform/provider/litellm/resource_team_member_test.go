package litellm

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func TestTeamMemberCreateOmitsUnsetBudget(t *testing.T) {
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

	if err := resourceLiteLLMTeamMemberCreate(d, client); err != nil {
		t.Fatalf("create failed: %v", err)
	}
	if _, ok := captured["max_budget_in_team"]; ok {
		t.Fatalf("create payload included unset max_budget_in_team: %v", captured)
	}
}

func TestOptionalTeamMemberBudgetPreservesExplicitZero(t *testing.T) {
	d := schema.TestResourceDataRaw(t, resourceLiteLLMTeamMember().Schema, map[string]interface{}{
		"max_budget_in_team": 0.0,
	})
	payload := map[string]interface{}{}

	addOptionalTeamMemberBudget(d, payload)

	if budget, ok := payload["max_budget_in_team"]; !ok || budget != 0.0 {
		t.Fatalf("explicit zero budget was not preserved: %v", payload)
	}
}

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
	if _, ok := captured["max_budget_in_team"]; ok {
		t.Fatalf("update payload included unset max_budget_in_team: %v", captured)
	}
}

func TestTeamMemberAddBudgetPresence(t *testing.T) {
	for _, tc := range []struct {
		name       string
		configured bool
		budget     float64
	}{
		{"omitted", false, 0}, {"zero", true, 0}, {"positive", true, 25},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var captured map[string]interface{}
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if err := json.NewDecoder(r.Body).Decode(&captured); err != nil {
					t.Error(err)
				}
				w.Header().Set("Content-Type", "application/json")
				w.Write([]byte(`{}`))
			}))
			defer srv.Close()
			raw := map[string]interface{}{"team_id": "team-1", "member": []interface{}{map[string]interface{}{"user_id": "user-1", "role": "user"}}}
			if tc.configured {
				raw["max_budget_in_team"] = tc.budget
			}
			d := schema.TestResourceDataRaw(t, resourceLiteLLMTeamMemberAdd().Schema, raw)
			if err := resourceLiteLLMTeamMemberAddCreate(d, NewClient(srv.URL, "test-key", true)); err != nil {
				t.Fatal(err)
			}
			got, present := captured["max_budget_in_team"]
			if present != tc.configured || (present && got != tc.budget) {
				t.Fatalf("budget = %v, present = %t", got, present)
			}
		})
	}
}
