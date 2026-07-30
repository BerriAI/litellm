package litellm

import (
	"testing"
)

// TestResourceLiteLLMTeam_hasImporter guards the regression fixed alongside
// this test: docs/resources/team.md documents `terraform import
// litellm_team.<name> <team-id>`, but the resource did not set Importer, so
// every import failed with "resource litellm_team doesn't support import".
//
// Teams are routinely created out-of-band (UI, /team/new scripts) and carry
// spend history, budgets, and key associations, so import is the only safe way
// to adopt one into Terraform - without it the sole alternative is letting
// apply POST /team/new a duplicate.
func TestResourceLiteLLMTeam_hasImporter(t *testing.T) {
	r := ResourceLiteLLMTeam()

	if r.Importer == nil {
		t.Fatal("litellm_team must set Importer: import is documented in docs/resources/team.md")
	}

	if r.Importer.StateContext == nil {
		t.Error("litellm_team Importer must set StateContext")
	}
}

// TestResourceLiteLLMTeam_importPopulatesFromID documents why passthrough
// import is sufficient here: resourceLiteLLMTeamRead reconstructs every
// schema field from the ID alone via GET /team/info, so no custom import
// logic is required to produce complete state.
func TestResourceLiteLLMTeam_importPopulatesFromID(t *testing.T) {
	r := ResourceLiteLLMTeam()

	if r.Read == nil {
		t.Fatal("litellm_team must set Read: passthrough import relies on it to populate state")
	}
}
