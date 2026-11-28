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

func newTeamMemberDeleteServer(t *testing.T, status int, body string) (*httptest.Server, *[]map[string]interface{}) {
	t.Helper()
	var calls []map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/team/member_delete" {
			t.Errorf("unexpected request path: %s", r.URL.Path)
		}
		raw, _ := io.ReadAll(r.Body)
		var payload map[string]interface{}
		json.Unmarshal(raw, &payload)
		calls = append(calls, payload)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return srv, &calls
}

var teamMemberDeleteCases = []struct {
	name    string
	status  int
	body    string
	wantErr bool
}{
	{"removed", http.StatusOK, `{}`, false},
	{"already removed from team", http.StatusBadRequest, `{"detail":{"error":"User not found in team"}}`, false},
	{"not found", http.StatusNotFound, `{"detail":"Not Found"}`, false},
	{"team missing", http.StatusBadRequest, `{"detail":{"error":"Team id=team-1 does not exist in db"}}`, true},
	{"forbidden", http.StatusForbidden, `{"detail":{"error":"Call not allowed"}}`, true},
	{"server error", http.StatusInternalServerError, `{"detail":{"error":"boom"}}`, true},
}

func TestTeamMemberDeleteTreatsAlreadyRemovedMemberAsDeleted(t *testing.T) {
	for _, tc := range teamMemberDeleteCases {
		t.Run(tc.name, func(t *testing.T) {
			srv, calls := newTeamMemberDeleteServer(t, tc.status, tc.body)
			d := schema.TestResourceDataRaw(t, resourceLiteLLMTeamMember().Schema, map[string]interface{}{
				"team_id":    "team-1",
				"user_id":    "user-1",
				"user_email": "user@example.com",
				"role":       "user",
			})
			d.SetId("team-1:user-1")

			err := resourceLiteLLMTeamMemberDelete(d, NewClient(srv.URL, "test-key", true))

			if len(*calls) != 1 || (*calls)[0]["user_id"] != "user-1" || (*calls)[0]["team_id"] != "team-1" {
				t.Fatalf("member_delete calls = %v, want one for team-1/user-1", *calls)
			}
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected an error for %d %s", tc.status, tc.body)
				}
				if d.Id() == "" {
					t.Fatalf("failed delete must keep the resource in state")
				}
				return
			}
			if err != nil {
				t.Fatalf("delete failed: %v", err)
			}
			if d.Id() != "" {
				t.Fatalf("expected ID cleared, got %q", d.Id())
			}
		})
	}
}

func TestTeamMemberAddDeleteTreatsAlreadyRemovedMembersAsDeleted(t *testing.T) {
	for _, tc := range teamMemberDeleteCases {
		t.Run(tc.name, func(t *testing.T) {
			srv, calls := newTeamMemberDeleteServer(t, tc.status, tc.body)
			d := schema.TestResourceDataRaw(t, resourceLiteLLMTeamMemberAdd().Schema, map[string]interface{}{
				"team_id": "team-1",
				"member": []interface{}{
					map[string]interface{}{"user_id": "user-1", "role": "user"},
					map[string]interface{}{"user_id": "user-2", "role": "user"},
				},
			})
			d.SetId("team-1")

			err := resourceLiteLLMTeamMemberAddDelete(d, NewClient(srv.URL, "test-key", true))

			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected an error for %d %s", tc.status, tc.body)
				}
				if d.Id() == "" {
					t.Fatalf("failed delete must keep the resource in state")
				}
				return
			}
			if err != nil {
				t.Fatalf("delete failed: %v", err)
			}
			if len(*calls) != 2 {
				t.Fatalf("expected a member_delete call per member, got %v", *calls)
			}
			if d.Id() != "" {
				t.Fatalf("expected ID cleared, got %q", d.Id())
			}
		})
	}
}
