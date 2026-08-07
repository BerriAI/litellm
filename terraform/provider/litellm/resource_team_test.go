package litellm

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

// TestResourceLiteLLMTeam_hasImporter guards the regression fixed alongside
// this test: docs/resources/team.md documents `terraform import
// litellm_team.<name> <team-id>`, but the resource did not set Importer, so
// every import failed with "resource litellm_team doesn't support import".
//
// Teams are routinely created out-of-band (UI, /team/new scripts) and carry
// spend history, budgets, and key associations, so import is the only way to
// adopt one into Terraform - without it the sole alternative is letting apply
// POST /team/new a duplicate.
func TestResourceLiteLLMTeam_hasImporter(t *testing.T) {
	r := ResourceLiteLLMTeam()

	if r.Importer == nil {
		t.Fatal("litellm_team must set Importer: import is documented in docs/resources/team.md")
	}

	if r.Importer.StateContext == nil {
		t.Error("litellm_team Importer must set StateContext")
	}
}

// TestResourceLiteLLMTeam_importReadsByTeamID asserts the behavior passthrough
// import depends on: that Read, given only the ID the importer passes through,
// looks the team up by that ID against /team/info and keeps it as the resource
// ID. This is what makes import adopt the real team rather than creating a
// duplicate.
//
// It deliberately does not assert on hydrated fields such as team_alias or
// max_budget. /team/info nests those under a "team_info" key while
// TeamResponse declares them at the top level, so they currently decode to
// zero values - a pre-existing bug called out in the PR description and left
// to a separate change.
func TestResourceLiteLLMTeam_importReadsByTeamID(t *testing.T) {
	const teamID = "d93eb581-d215-4332-aea9-fab87caf21de"

	// Read also calls /team/permissions_list, so key the lookups by path
	// rather than recording whichever request happened to arrive last.
	queriedTeamID := map[string]string{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		queriedTeamID[r.URL.Path] = r.URL.Query().Get("team_id")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"team_id":"` + teamID + `","team_info":{},"keys":[],"team_memberships":[]}`))
	}))
	defer srv.Close()

	d := ResourceLiteLLMTeam().TestResourceData()
	d.SetId(teamID)

	if err := resourceLiteLLMTeamRead(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("read after import: %v", err)
	}

	got, called := queriedTeamID[endpointTeamInfo]
	if !called {
		t.Fatalf("read did not call %s; called %v", endpointTeamInfo, queriedTeamID)
	}
	if got != teamID {
		t.Errorf("looked up team_id %q, want %q", got, teamID)
	}
	if d.Id() != teamID {
		t.Errorf("resource ID is %q after read, want %q - import must adopt the existing team", d.Id(), teamID)
	}
}

// TestResourceLiteLLMTeam_importDropsMissingTeam covers the other half of the
// contract: importing an ID that no longer exists must clear the ID rather
// than leave a phantom resource in state.
func TestResourceLiteLLMTeam_importDropsMissingTeam(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	d := ResourceLiteLLMTeam().TestResourceData()
	d.SetId("does-not-exist")

	if err := resourceLiteLLMTeamRead(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("read of missing team should not error, got: %v", err)
	}

	if d.Id() != "" {
		t.Errorf("resource ID is %q, want empty - a missing team must be dropped from state", d.Id())
	}
}
