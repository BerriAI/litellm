package litellm

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func newTeamBlockTestResourceData(t *testing.T, teamID string) *schema.ResourceData {
	t.Helper()
	return schema.TestResourceDataRaw(t, resourceLiteLLMTeamBlock().Schema, map[string]interface{}{
		"team_id": teamID,
	})
}

func TestResourceLiteLLMTeamBlockCreate(t *testing.T) {
	var blockPayload map[string]interface{}
	mux := http.NewServeMux()
	mux.HandleFunc("/team/block", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if err := json.NewDecoder(r.Body).Decode(&blockPayload); err != nil {
			t.Fatalf("failed to decode block payload: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"team_id":"team-123","blocked":true}`))
	})
	mux.HandleFunc("/team/info", func(w http.ResponseWriter, r *http.Request) {
		if got := r.URL.Query().Get("team_id"); got != "team-123" {
			t.Errorf("expected team_id query 'team-123', got %q", got)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"team_id":"team-123","team_info":{"blocked":true}}`))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newTeamBlockTestResourceData(t, "team-123")

	if err := resourceLiteLLMTeamBlockCreate(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if d.Id() != "team-123" {
		t.Fatalf("expected ID 'team-123', got %q", d.Id())
	}
	if blockPayload["team_id"] != "team-123" {
		t.Fatalf("expected block payload team_id 'team-123', got %+v", blockPayload)
	}
	if !d.Get("blocked").(bool) {
		t.Fatal("expected blocked=true in state")
	}
}

func TestResourceLiteLLMTeamBlockRead_UnblockedClearsID(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"team_id":"team-123","team_info":{"blocked":false}}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newTeamBlockTestResourceData(t, "team-123")
	d.SetId("team-123")

	if err := resourceLiteLLMTeamBlockRead(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if d.Id() != "" {
		t.Fatalf("expected ID cleared for unblocked team, got %q", d.Id())
	}
}

func TestResourceLiteLLMTeamBlockRead_404ClearsID(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newTeamBlockTestResourceData(t, "team-123")
	d.SetId("team-123")

	if err := resourceLiteLLMTeamBlockRead(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if d.Id() != "" {
		t.Fatalf("expected ID cleared on 404, got %q", d.Id())
	}
}

func TestResourceLiteLLMTeamBlockDelete(t *testing.T) {
	var gotPath string
	var unblockPayload map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		json.NewDecoder(r.Body).Decode(&unblockPayload)
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"team_id":"team-123","blocked":false}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newTeamBlockTestResourceData(t, "team-123")
	d.SetId("team-123")

	if err := resourceLiteLLMTeamBlockDelete(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if gotPath != "/team/unblock" {
		t.Fatalf("expected path /team/unblock, got %s", gotPath)
	}
	if unblockPayload["team_id"] != "team-123" {
		t.Fatalf("expected unblock payload team_id 'team-123', got %+v", unblockPayload)
	}
	if d.Id() != "" {
		t.Fatalf("expected ID cleared after delete, got %q", d.Id())
	}
}
