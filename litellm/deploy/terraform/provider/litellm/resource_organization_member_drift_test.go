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

// fakeOrgProxy serves GET /organization/info and DELETE /organization/member_delete
// for a single organization, so Read and Delete can be checked against drift.
type fakeOrgProxy struct {
	t             *testing.T
	orgID         string
	orgExists     bool
	members       []map[string]interface{}
	deleteStatus  int
	deleteCalls   int
	deleteBodies  []map[string]interface{}
	updatedBodies []map[string]interface{}
}

func decodeBody(t *testing.T, r *http.Request) map[string]interface{} {
	var body map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		t.Errorf("decoding %s body: %v", r.URL.Path, err)
	}
	return body
}

func (p *fakeOrgProxy) serve() *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/organization/info":
			if got := r.URL.Query().Get("organization_id"); got != p.orgID {
				p.t.Errorf("organization/info called with organization_id=%q, want %q", got, p.orgID)
			}
			if !p.orgExists {
				w.WriteHeader(http.StatusNotFound)
				w.Write([]byte(`{"detail":{"error":"Organization not found"}}`))
				return
			}
			json.NewEncoder(w).Encode(map[string]interface{}{
				"organization_id": p.orgID,
				"members":         p.members,
			})
		case r.Method == http.MethodDelete && r.URL.Path == "/organization/member_delete":
			p.deleteCalls++
			p.deleteBodies = append(p.deleteBodies, decodeBody(p.t, r))
			status := p.deleteStatus
			if status == 0 {
				status = http.StatusOK
			}
			w.WriteHeader(status)
			w.Write([]byte(`{}`))
		case r.Method == http.MethodPatch && r.URL.Path == "/organization/member_update":
			p.updatedBodies = append(p.updatedBodies, decodeBody(p.t, r))
			w.Write([]byte(`{}`))
		default:
			p.t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
}

func orgMemberRow(userID, email, role string) map[string]interface{} {
	return map[string]interface{}{
		"user_id":         userID,
		"organization_id": "org-1",
		"user_role":       role,
		"user":            map[string]interface{}{"user_id": userID, "user_email": email},
	}
}

func newOrgMemberData(t *testing.T) *schema.ResourceData {
	d := schema.TestResourceDataRaw(t, resourceLiteLLMOrganizationMember().Schema, map[string]interface{}{
		"organization_id": "org-1",
		"user_id":         "user-1",
		"user_email":      "user-1@example.com",
		"role":            "org_admin",
	})
	d.SetId("org-1:user-1")
	return d
}

func TestOrganizationMemberReadRefreshesRole(t *testing.T) {
	proxy := &fakeOrgProxy{t: t, orgID: "org-1", orgExists: true, members: []map[string]interface{}{
		orgMemberRow("user-1", "user-1@example.com", "internal_user_viewer"),
	}}
	srv := proxy.serve()
	defer srv.Close()

	d := newOrgMemberData(t)
	if err := resourceLiteLLMOrganizationMemberRead(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("read failed: %v", err)
	}
	if d.Id() != "org-1:user-1" {
		t.Fatalf("member still in the organization was removed from state")
	}
	if got := d.Get("role").(string); got != "internal_user_viewer" {
		t.Fatalf("role = %q, want the proxy's internal_user_viewer", got)
	}
}

func TestOrganizationMemberReadRemovesMemberRemovedOutOfBand(t *testing.T) {
	proxy := &fakeOrgProxy{t: t, orgID: "org-1", orgExists: true, members: []map[string]interface{}{
		orgMemberRow("user-2", "user-2@example.com", "internal_user"),
	}}
	srv := proxy.serve()
	defer srv.Close()

	d := newOrgMemberData(t)
	if err := resourceLiteLLMOrganizationMemberRead(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("read failed: %v", err)
	}
	if d.Id() != "" {
		t.Fatalf("member missing from the organization should be removed from state, got ID %q", d.Id())
	}
}

func TestOrganizationMemberReadRemovesMemberOfDeletedOrganization(t *testing.T) {
	proxy := &fakeOrgProxy{t: t, orgID: "org-1", orgExists: false}
	srv := proxy.serve()
	defer srv.Close()

	d := newOrgMemberData(t)
	if err := resourceLiteLLMOrganizationMemberRead(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("read failed: %v", err)
	}
	if d.Id() != "" {
		t.Fatalf("member of a deleted organization should be removed from state, got ID %q", d.Id())
	}
}

func TestOrganizationMemberDeleteTreatsAlreadyRemovedMemberAsDeleted(t *testing.T) {
	proxy := &fakeOrgProxy{t: t, orgID: "org-1", orgExists: true, deleteStatus: http.StatusInternalServerError}
	srv := proxy.serve()
	defer srv.Close()

	d := newOrgMemberData(t)
	if err := resourceLiteLLMOrganizationMemberDelete(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("deleting a member that is already gone should succeed, got: %v", err)
	}
	if proxy.deleteCalls != 1 {
		t.Fatalf("expected 1 member_delete call, got %d", proxy.deleteCalls)
	}
	if d.Id() != "" {
		t.Fatalf("ID should be cleared after delete, got %q", d.Id())
	}
}

func TestOrganizationMemberDeleteTreatsDeletedOrganizationAsDeleted(t *testing.T) {
	proxy := &fakeOrgProxy{t: t, orgID: "org-1", orgExists: false, deleteStatus: http.StatusInternalServerError}
	srv := proxy.serve()
	defer srv.Close()

	d := newOrgMemberData(t)
	if err := resourceLiteLLMOrganizationMemberDelete(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("deleting a member of a deleted organization should succeed, got: %v", err)
	}
}

func TestOrganizationMemberDeleteKeepsErrorWhenMemberStillPresent(t *testing.T) {
	proxy := &fakeOrgProxy{t: t, orgID: "org-1", orgExists: true, deleteStatus: http.StatusInternalServerError,
		members: []map[string]interface{}{orgMemberRow("user-1", "user-1@example.com", "org_admin")}}
	srv := proxy.serve()
	defer srv.Close()

	d := newOrgMemberData(t)
	if err := resourceLiteLLMOrganizationMemberDelete(d, NewClient(srv.URL, "test-key", true)); err == nil {
		t.Fatalf("a failed delete of a member that is still present must return an error")
	}
}

func TestOrganizationMemberAddReadKeepsOnlyRemainingMembers(t *testing.T) {
	proxy := &fakeOrgProxy{t: t, orgID: "org-1", orgExists: true, members: []map[string]interface{}{
		orgMemberRow("user-1", "user-1@example.com", "internal_user_viewer"),
		// Added by email only: the proxy minted the user_id, and the stored email differs in case.
		orgMemberRow("generated-id", "bob@example.com", "internal_user"),
	}}
	srv := proxy.serve()
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resourceLiteLLMOrganizationMemberAdd().Schema, map[string]interface{}{
		"organization_id": "org-1",
		"member": []interface{}{
			map[string]interface{}{"user_id": "user-1", "role": "org_admin"},
			map[string]interface{}{"user_id": "user-2", "role": "internal_user"},
			map[string]interface{}{"user_email": "Bob@Example.com", "role": "internal_user"},
		},
	})
	d.SetId("org-1")

	if err := resourceLiteLLMOrganizationMemberAddRead(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("read failed: %v", err)
	}

	got := map[string]string{}
	for _, raw := range d.Get("member").(*schema.Set).List() {
		member := raw.(map[string]interface{})
		key := getOrgMemberKey(member)
		got[key] = member["role"].(string)
	}
	want := map[string]string{
		"id:user-1":             "internal_user_viewer",
		"email:Bob@Example.com": "internal_user",
	}
	if len(got) != len(want) {
		t.Fatalf("members in state = %v, want %v", got, want)
	}
	for key, role := range want {
		if got[key] != role {
			t.Fatalf("members in state = %v, want %v", got, want)
		}
	}
}

func TestOrganizationMemberAddReadRemovesDeletedOrganization(t *testing.T) {
	proxy := &fakeOrgProxy{t: t, orgID: "org-1", orgExists: false}
	srv := proxy.serve()
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resourceLiteLLMOrganizationMemberAdd().Schema, map[string]interface{}{
		"organization_id": "org-1",
		"member": []interface{}{
			map[string]interface{}{"user_id": "user-1", "role": "org_admin"},
		},
	})
	d.SetId("org-1")

	if err := resourceLiteLLMOrganizationMemberAddRead(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("read failed: %v", err)
	}
	if d.Id() != "" {
		t.Fatalf("members of a deleted organization should be removed from state, got ID %q", d.Id())
	}
}

func TestOrganizationMemberAddDeleteSkipsMembersAlreadyRemoved(t *testing.T) {
	proxy := &fakeOrgProxy{t: t, orgID: "org-1", orgExists: true, deleteStatus: http.StatusInternalServerError}
	srv := proxy.serve()
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resourceLiteLLMOrganizationMemberAdd().Schema, map[string]interface{}{
		"organization_id": "org-1",
		"member": []interface{}{
			map[string]interface{}{"user_id": "user-1", "role": "org_admin"},
			map[string]interface{}{"user_email": "bob@example.com", "role": "internal_user"},
		},
	})
	d.SetId("org-1")

	if err := resourceLiteLLMOrganizationMemberAddDelete(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("deleting members that are already gone should succeed, got: %v", err)
	}
	if proxy.deleteCalls != 2 {
		t.Fatalf("expected 2 member_delete calls, got %d", proxy.deleteCalls)
	}
}

func emailOnlyMember(email, role string) map[string]interface{} {
	return map[string]interface{}{"user_email": email, "role": role}
}

// refreshedEmailOnlyState applies the config, reads it back from the proxy, and
// returns the state Terraform would persist.
func refreshedEmailOnlyState(t *testing.T, client *Client, members ...interface{}) *terraform.InstanceState {
	d := schema.TestResourceDataRaw(t, resourceLiteLLMOrganizationMemberAdd().Schema, map[string]interface{}{
		"organization_id": "org-1",
		"member":          members,
	})
	d.SetId("org-1")
	if err := resourceLiteLLMOrganizationMemberAddRead(d, client); err != nil {
		t.Fatalf("read failed: %v", err)
	}
	return d.State()
}

func orgMemberAddDataFromState(t *testing.T, state *terraform.InstanceState) *schema.ResourceData {
	d, err := schema.InternalMap(resourceLiteLLMOrganizationMemberAdd().Schema).Data(state, nil)
	if err != nil {
		t.Fatalf("loading state: %v", err)
	}
	return d
}

func orgMemberAddDataWithConfig(t *testing.T, state *terraform.InstanceState, members ...interface{}) *schema.ResourceData {
	internal := schema.InternalMap(resourceLiteLLMOrganizationMemberAdd().Schema)
	config := terraform.NewResourceConfigRaw(map[string]interface{}{"organization_id": "org-1", "member": members})
	diff, err := internal.Diff(context.Background(), state, config, nil, nil, true)
	if err != nil {
		t.Fatalf("diffing config: %v", err)
	}
	d, err := internal.Data(state, diff)
	if err != nil {
		t.Fatalf("loading state with diff: %v", err)
	}
	return d
}

func TestOrganizationMemberAddReadKeepsEmailOnlyMemberAfterEmailChange(t *testing.T) {
	proxy := &fakeOrgProxy{t: t, orgID: "org-1", orgExists: true, members: []map[string]interface{}{
		orgMemberRow("generated-id", "bob@example.com", "internal_user"),
	}}
	srv := proxy.serve()
	defer srv.Close()
	client := NewClient(srv.URL, "test-key", true)

	state := refreshedEmailOnlyState(t, client, emailOnlyMember("Bob@Example.com", "internal_user"))
	proxy.members = []map[string]interface{}{orgMemberRow("generated-id", "bob-new@example.com", "internal_user_viewer")}

	d := orgMemberAddDataFromState(t, state)
	if err := resourceLiteLLMOrganizationMemberAddRead(d, client); err != nil {
		t.Fatalf("read failed: %v", err)
	}

	list := d.Get("member").(*schema.Set).List()
	if len(list) != 1 {
		t.Fatalf("member whose email changed should stay in state, got %v", list)
	}
	member := list[0].(map[string]interface{})
	if member["user_email"] != "Bob@Example.com" || member["role"] != "internal_user_viewer" {
		t.Fatalf("member = %v, want the configured email with the proxy's role", member)
	}
}

func TestOrganizationMemberAddReadDoesNotRebindResolvedIDByEmail(t *testing.T) {
	proxy := &fakeOrgProxy{t: t, orgID: "org-1", orgExists: true, members: []map[string]interface{}{
		orgMemberRow("old-id", "bob@example.com", "org_admin"),
	}}
	srv := proxy.serve()
	defer srv.Close()
	client := NewClient(srv.URL, "test-key", true)

	state := refreshedEmailOnlyState(t, client, emailOnlyMember("bob@example.com", "org_admin"))
	proxy.members = []map[string]interface{}{orgMemberRow("other-id", "BOB@example.com", "internal_user")}

	d := orgMemberAddDataFromState(t, state)
	if err := resourceLiteLLMOrganizationMemberAddRead(d, client); err != nil {
		t.Fatalf("read failed: %v", err)
	}
	if got := d.Get("member").(*schema.Set).Len(); got != 0 {
		t.Fatalf("member whose resolved user_id is gone should leave state, %d left", got)
	}
	if got := d.Get("member_user_ids").(map[string]interface{}); len(got) != 0 {
		t.Fatalf("another membership with the same email must not be tracked, got %v", got)
	}
}

func TestOrganizationMemberAddReadDropsEmailOnlyMemberRemovedOutOfBand(t *testing.T) {
	proxy := &fakeOrgProxy{t: t, orgID: "org-1", orgExists: true, members: []map[string]interface{}{
		orgMemberRow("generated-id", "bob@example.com", "internal_user"),
	}}
	srv := proxy.serve()
	defer srv.Close()
	client := NewClient(srv.URL, "test-key", true)

	state := refreshedEmailOnlyState(t, client, emailOnlyMember("bob@example.com", "internal_user"))
	proxy.members = nil

	d := orgMemberAddDataFromState(t, state)
	if err := resourceLiteLLMOrganizationMemberAddRead(d, client); err != nil {
		t.Fatalf("read failed: %v", err)
	}
	if got := d.Get("member").(*schema.Set).Len(); got != 0 {
		t.Fatalf("member removed out of band should leave state, %d left", got)
	}
	if got := d.Get("member_user_ids").(map[string]interface{}); len(got) != 0 {
		t.Fatalf("resolved user_ids of a removed member should be dropped, got %v", got)
	}
}

func TestOrganizationMemberAddDeleteRevokesByResolvedIDAfterEmailChange(t *testing.T) {
	proxy := &fakeOrgProxy{t: t, orgID: "org-1", orgExists: true, members: []map[string]interface{}{
		orgMemberRow("generated-id", "bob@example.com", "internal_user"),
	}}
	srv := proxy.serve()
	defer srv.Close()
	client := NewClient(srv.URL, "test-key", true)

	state := refreshedEmailOnlyState(t, client, emailOnlyMember("bob@example.com", "internal_user"))
	proxy.members = []map[string]interface{}{orgMemberRow("generated-id", "bob-new@example.com", "internal_user")}

	if err := resourceLiteLLMOrganizationMemberAddDelete(orgMemberAddDataFromState(t, state), client); err != nil {
		t.Fatalf("delete failed: %v", err)
	}
	if len(proxy.deleteBodies) != 1 || proxy.deleteBodies[0]["user_id"] != "generated-id" {
		t.Fatalf("delete bodies = %v, want one delete by generated-id", proxy.deleteBodies)
	}
	if _, sent := proxy.deleteBodies[0]["user_email"]; sent {
		t.Fatalf("delete should not send the stale email, got %v", proxy.deleteBodies[0])
	}
}

func TestOrganizationMemberAddDeleteKeepsErrorWhenEmailChangedMemberStillPresent(t *testing.T) {
	proxy := &fakeOrgProxy{t: t, orgID: "org-1", orgExists: true, members: []map[string]interface{}{
		orgMemberRow("generated-id", "bob@example.com", "internal_user"),
	}}
	srv := proxy.serve()
	defer srv.Close()
	client := NewClient(srv.URL, "test-key", true)

	state := refreshedEmailOnlyState(t, client, emailOnlyMember("bob@example.com", "internal_user"))
	proxy.members = []map[string]interface{}{orgMemberRow("generated-id", "bob-new@example.com", "internal_user")}
	proxy.deleteStatus = http.StatusInternalServerError

	if err := resourceLiteLLMOrganizationMemberAddDelete(orgMemberAddDataFromState(t, state), client); err == nil {
		t.Fatalf("a failed delete of a member still in the organization must return an error")
	}
}

func TestOrganizationMemberAddUpdateUsesResolvedIDAfterEmailChange(t *testing.T) {
	proxy := &fakeOrgProxy{t: t, orgID: "org-1", orgExists: true, members: []map[string]interface{}{
		orgMemberRow("generated-id", "bob@example.com", "internal_user"),
		orgMemberRow("generated-id-2", "carol@example.com", "internal_user"),
	}}
	srv := proxy.serve()
	defer srv.Close()
	client := NewClient(srv.URL, "test-key", true)

	state := refreshedEmailOnlyState(t, client,
		emailOnlyMember("bob@example.com", "internal_user"),
		emailOnlyMember("carol@example.com", "internal_user"),
	)
	proxy.members = []map[string]interface{}{
		orgMemberRow("generated-id", "bob-new@example.com", "internal_user"),
		orgMemberRow("generated-id-2", "carol-new@example.com", "internal_user"),
	}

	d := orgMemberAddDataWithConfig(t, state, emailOnlyMember("carol@example.com", "org_admin"))
	if err := resourceLiteLLMOrganizationMemberAddUpdate(d, client); err != nil {
		t.Fatalf("update failed: %v", err)
	}

	if len(proxy.deleteBodies) != 1 || proxy.deleteBodies[0]["user_id"] != "generated-id" {
		t.Fatalf("delete bodies = %v, want one delete by generated-id", proxy.deleteBodies)
	}
	if len(proxy.updatedBodies) != 1 || proxy.updatedBodies[0]["user_id"] != "generated-id-2" {
		t.Fatalf("update bodies = %v, want one update by generated-id-2", proxy.updatedBodies)
	}
}

func TestOrganizationMemberAddPlansNoChangesAfterEmailChange(t *testing.T) {
	proxy := &fakeOrgProxy{t: t, orgID: "org-1", orgExists: true, members: []map[string]interface{}{
		orgMemberRow("generated-id", "bob@example.com", "internal_user"),
	}}
	srv := proxy.serve()
	defer srv.Close()
	client := NewClient(srv.URL, "test-key", true)

	state := refreshedEmailOnlyState(t, client, emailOnlyMember("bob@example.com", "internal_user"))
	proxy.members = []map[string]interface{}{orgMemberRow("generated-id", "bob-new@example.com", "internal_user")}

	d := orgMemberAddDataFromState(t, state)
	if err := resourceLiteLLMOrganizationMemberAddRead(d, client); err != nil {
		t.Fatalf("read failed: %v", err)
	}

	config := terraform.NewResourceConfigRaw(map[string]interface{}{
		"organization_id": "org-1",
		"member":          []interface{}{emailOnlyMember("bob@example.com", "internal_user")},
	})
	internal := schema.InternalMap(resourceLiteLLMOrganizationMemberAdd().Schema)
	diff, err := internal.Diff(context.Background(), d.State(), config, nil, nil, true)
	if err != nil {
		t.Fatalf("diffing config: %v", err)
	}
	if diff != nil && !diff.Empty() {
		t.Fatalf("plan after an email change should be empty, got %v", diff)
	}
}
