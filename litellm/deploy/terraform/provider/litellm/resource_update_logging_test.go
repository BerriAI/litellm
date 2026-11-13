package litellm

import (
	"bytes"
	"log"
	"strings"
	"testing"
)

func TestResourceUpdateLogsExcludeLiveMetadata(t *testing.T) {
	const secret = "synthetic-nested-callback-credential"
	logging := []interface{}{map[string]interface{}{
		"callback_vars": map[string]interface{}{"api_key": secret},
	}}
	for _, kind := range []string{"team", "project"} {
		t.Run(kind, func(t *testing.T) {
			var logs bytes.Buffer
			previous := log.Writer()
			log.SetOutput(&logs)
			defer log.SetOutput(previous)
			if kind == "team" {
				state := teamSafetyState()
				state.info["metadata"].(map[string]interface{})["logging"] = logging
				server := newTeamSafetyServer(t, state)
				defer server.Close()
				before := teamSafetyConfig()
				after := teamSafetyConfig()
				after["tags"] = []interface{}{"new-tag"}
				data := resourceUpdateData(t, ResourceLiteLLMTeam(), "team-1", before, after)
				if err := resourceLiteLLMTeamUpdate(data, NewClient(server.URL, "test-key", true)); err != nil {
					t.Fatal(err)
				}
				preserved := state.info["metadata"].(map[string]interface{})["logging"].([]interface{})[0].(map[string]interface{})["callback_vars"].(map[string]interface{})["api_key"]
				if preserved != secret {
					t.Fatal("update must preserve callback credentials")
				}
			} else {
				state := newProjectUpdateServer(t)
				state.project["metadata"].(map[string]interface{})["logging"] = logging
				server := state.serve(t)
				defer server.Close()
				before := map[string]interface{}{"team_id": "team-1", "tags": []interface{}{"production"}}
				after := map[string]interface{}{"team_id": "team-1", "tags": []interface{}{"new-tag"}}
				data := resourceUpdateData(t, resourceLiteLLMProject(), "proj-123", before, after)
				if err := resourceLiteLLMProjectUpdate(data, NewClient(server.URL, "test-key", true)); err != nil {
					t.Fatal(err)
				}
				snapshot, _ := state.snapshot(t)
				preserved := snapshot["metadata"].(map[string]interface{})["logging"].([]interface{})[0].(map[string]interface{})["callback_vars"].(map[string]interface{})["api_key"]
				if preserved != secret {
					t.Fatal("update must preserve callback credentials")
				}
			}
			if strings.Contains(logs.String(), secret) {
				t.Fatal("resource update logs exposed a nested callback credential")
			}
		})
	}
}
