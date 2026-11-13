package litellm

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
)

// This server models replacement metadata, nested member budgets, and the
// budget-window side effects of sending durations to /team/update.
type teamSafetyServer struct {
	info             map[string]interface{}
	infoReads        int
	updates          int
	permissionWrites int
	window           string
	memberWindow     string
}

func newTeamSafetyServer(t *testing.T, state *teamSafetyServer) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case endpointTeamInfo:
			state.infoReads++
			json.NewEncoder(w).Encode(map[string]interface{}{"team_id": "team-1", "team_info": state.info})
		case endpointTeamPermissionsList:
			json.NewEncoder(w).Encode(map[string]interface{}{"team_id": "team-1", "team_member_permissions": state.info["team_member_permissions"]})
		case endpointTeamPermissionsUpdate:
			state.permissionWrites++
			var payload map[string]interface{}
			if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
				t.Error(err)
				w.WriteHeader(400)
				return
			}
			state.info["team_member_permissions"] = payload["team_member_permissions"]
			w.Write([]byte(`{}`))
		case endpointTeamUpdate:
			state.updates++
			var payload map[string]interface{}
			decoder := json.NewDecoder(r.Body)
			decoder.UseNumber()
			if err := decoder.Decode(&payload); err != nil {
				t.Error(err)
				w.WriteHeader(400)
				return
			}
			metadata, _ := state.info["metadata"].(map[string]interface{})
			// Metadata replaces the column, then top-level metadata-backed fields merge in.
			if value, ok := payload["metadata"]; ok {
				incoming := value.(map[string]interface{})
				incoming["team_member_budget_id"] = metadata["team_member_budget_id"]
				metadata = incoming
				state.info["metadata"] = metadata
			}
			member, _ := state.info["team_member_budget_table"].(map[string]interface{})
			for key, value := range payload {
				switch key {
				case "metadata":
				case "tags", "model_rpm_limit", "model_tpm_limit":
					metadata[key] = value
				case "budget_duration":
					state.window = "restarted"
					state.info[key] = value
				case "team_member_budget_duration":
					state.memberWindow = "restarted"
					member["budget_duration"] = value
				case "team_member_budget":
					member["max_budget"] = value
				case "team_member_rpm_limit":
					member["rpm_limit"] = value
				case "team_member_tpm_limit":
					member["tpm_limit"] = value
				case "team_member_model_max_budget":
					member["model_max_budget"] = value
				default:
					state.info[key] = value
				}
			}
			w.Write([]byte(`{}`))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
}

func teamSafetyConfig() map[string]interface{} {
	return map[string]interface{}{
		"team_alias": "eng", "models": []interface{}{"old-model"},
		"max_budget": 500.0, "soft_budget": 400.0, "rpm_limit": 100, "tpm_limit": 5000,
		"budget_duration": "1d", "team_member_budget": 50.0,
		"team_member_budget_duration": "1d", "team_member_rpm_limit": 10, "team_member_tpm_limit": 500,
		"team_member_key_duration": "1d", "team_member_permissions": []interface{}{"/key/generate"},
		"tags": []interface{}{"owned"}, "soft_budget_alerting_emails": []interface{}{"old@example.com"},
		"model_rpm_limit": map[string]interface{}{"old-model": 10}, "model_tpm_limit": map[string]interface{}{"old-model": 500},
		"team_member_model_max_budget": `{"old-model":{"max_budget":20,"budget_duration":"1d"}}`,
		"metadata":                     map[string]interface{}{"owner": "old", "external": "planned"},
	}
}

func teamSafetyState() *teamSafetyServer {
	return &teamSafetyServer{
		window: "original-window", memberWindow: "original-member-window",
		info: map[string]interface{}{
			"team_id": "team-1", "team_alias": "eng", "models": []interface{}{"old-model"},
			"max_budget": 700.0, "soft_budget": 600.0, "rpm_limit": 200, "tpm_limit": 9000,
			"budget_duration": "1d", "team_member_key_duration": "7d",
			"team_member_permissions":  []interface{}{"/key/info"},
			"team_member_budget_table": map[string]interface{}{"max_budget": 70.0, "rpm_limit": 20, "tpm_limit": 900, "budget_duration": "1d", "model_max_budget": map[string]interface{}{"live-model": map[string]interface{}{"max_budget": 30.0, "budget_duration": "7d"}}},
			"metadata": map[string]interface{}{
				"owner": "old", "external": "live", "guardrails": []interface{}{"safety"}, "policies": []interface{}{"policy-1"},
				"require_trace_id": true, "team_member_budget_id": "member-budget-1", "large_id": json.Number("9007199254740993"),
				"tags": []interface{}{"live"}, "soft_budget_alerting_emails": []interface{}{"live@example.com"},
				"model_rpm_limit": map[string]interface{}{"live-model": json.Number("20")}, "model_tpm_limit": map[string]interface{}{"live-model": json.Number("900")},
			},
		},
	}
}

func TestTeamUpdateSafetyUnrelatedEditsPreserveLiveControls(t *testing.T) {
	for _, field := range []string{"team_alias", "models"} {
		t.Run(field, func(t *testing.T) {
			state := teamSafetyState()
			want, _ := json.Marshal(state.info)
			srv := newTeamSafetyServer(t, state)
			defer srv.Close()
			before, after := teamSafetyConfig(), teamSafetyConfig()
			if field == "team_alias" {
				after[field] = "renamed"
			} else {
				after[field] = []interface{}{"new-model"}
			}
			d := resourceUpdateData(t, ResourceLiteLLMTeam(), "team-1", before, after)
			if err := resourceLiteLLMTeamUpdate(d, NewClient(srv.URL, "test-key", true)); err != nil {
				t.Fatal(err)
			}
			state.info[field] = before[field]
			got, _ := json.Marshal(state.info)
			if string(got) != string(want) {
				t.Errorf("unrelated edit changed live controls:\ngot %s\nwant %s", got, want)
			}
			if state.window != "original-window" || state.memberWindow != "original-member-window" {
				t.Errorf("budget windows restarted: %s, %s", state.window, state.memberWindow)
			}
			if state.infoReads != 1 || state.permissionWrites != 0 {
				t.Errorf("info reads=%d permission writes=%d, want 1 and 0", state.infoReads, state.permissionWrites)
			}
		})
	}
}

func TestTeamUpdateSafetyMergesChangedMetadataAgainstLiveValues(t *testing.T) {
	for _, combined := range []bool{false, true} {
		t.Run(map[bool]string{false: "owned-entry-only", true: "emails-tags-model-limits"}[combined], func(t *testing.T) {
			state := teamSafetyState()
			want := state.info["metadata"].(map[string]interface{})
			wantBytes, _ := json.Marshal(want)
			var expected map[string]interface{}
			decoder := json.NewDecoder(bytes.NewReader(wantBytes))
			decoder.UseNumber()
			if err := decoder.Decode(&expected); err != nil {
				t.Fatal(err)
			}
			expected["owner"] = "new"
			before, after := teamSafetyConfig(), teamSafetyConfig()
			after["metadata"] = map[string]interface{}{"owner": "new", "external": "planned"}
			if combined {
				after["soft_budget_alerting_emails"] = []interface{}{"new@example.com"}
				after["tags"] = []interface{}{"new-tag"}
				after["model_rpm_limit"] = map[string]interface{}{"new-model": 30}
				after["model_tpm_limit"] = map[string]interface{}{}
				expected["soft_budget_alerting_emails"] = []interface{}{"new@example.com"}
				expected["tags"] = []interface{}{"new-tag"}
				expected["model_rpm_limit"] = map[string]interface{}{"new-model": json.Number("30")}
				expected["model_tpm_limit"] = map[string]interface{}{}
			}
			srv := newTeamSafetyServer(t, state)
			defer srv.Close()
			d := resourceUpdateData(t, ResourceLiteLLMTeam(), "team-1", before, after)
			if err := resourceLiteLLMTeamUpdate(d, NewClient(srv.URL, "test-key", true)); err != nil {
				t.Fatal(err)
			}
			if got := state.info["metadata"]; !reflect.DeepEqual(got, expected) {
				t.Errorf("live metadata=%#v, want %#v", got, expected)
			}
			if state.infoReads != 2 {
				t.Errorf("info reads=%d, want live fetch plus refresh", state.infoReads)
			}
		})
	}
}

func TestTeamReadSafetyClearsNullBudgetsAndDurations(t *testing.T) {
	for _, nested := range []bool{false, true} {
		t.Run(map[bool]string{false: "absent-member-budget", true: "null-member-budget"}[nested], func(t *testing.T) {
			state := teamSafetyState()
			for _, key := range []string{"max_budget", "rpm_limit", "tpm_limit", "budget_duration", "team_member_key_duration"} {
				state.info[key] = nil
			}
			if nested {
				state.info["team_member_budget_table"] = map[string]interface{}{"max_budget": nil, "rpm_limit": nil, "tpm_limit": nil, "budget_duration": nil}
			} else {
				delete(state.info, "team_member_budget_table")
			}
			srv := newTeamSafetyServer(t, state)
			defer srv.Close()
			d := newTeamResourceData(t, teamSafetyConfig())
			d.SetId("team-1")
			if err := resourceLiteLLMTeamRead(d, NewClient(srv.URL, "test-key", true)); err != nil {
				t.Fatal(err)
			}
			for _, key := range []string{"max_budget", "rpm_limit", "tpm_limit", "team_member_budget", "team_member_rpm_limit", "team_member_tpm_limit"} {
				if value, ok := d.GetOk(key); ok {
					t.Errorf("%s retained %v after null/absent read", key, value)
				}
			}
			for _, key := range []string{"budget_duration", "team_member_budget_duration", "team_member_key_duration"} {
				if value := d.Get(key); value != "" {
					t.Errorf("%s retained %v", key, value)
				}
			}
		})
	}
}

func TestTeamReadSafetyUsesCanonicalNestedMemberBudget(t *testing.T) {
	state := teamSafetyState()
	state.info["team_member_budget_duration"] = "legacy-duration"
	srv := newTeamSafetyServer(t, state)
	defer srv.Close()
	d := newTeamResourceData(t, map[string]interface{}{"team_alias": "eng", "team_member_budget_duration": "stale"})
	d.SetId("team-1")
	if err := resourceLiteLLMTeamRead(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatal(err)
	}
	for key, want := range map[string]interface{}{"team_member_budget_duration": "1d", "team_member_budget": 70.0, "team_member_rpm_limit": 20, "team_member_tpm_limit": 900} {
		if got := d.Get(key); got != want {
			t.Errorf("%s=%v, want %v", key, got, want)
		}
	}
}

func TestTeamUpdateSafetyClearsOwnedFieldsWithoutClearingForeignMetadata(t *testing.T) {
	state := teamSafetyState()
	srv := newTeamSafetyServer(t, state)
	defer srv.Close()
	before, after := teamSafetyConfig(), teamSafetyConfig()
	after["metadata"] = map[string]interface{}{"external": "planned"}
	for _, key := range []string{"soft_budget", "soft_budget_alerting_emails", "tags", "model_rpm_limit", "model_tpm_limit", "team_member_model_max_budget"} {
		delete(after, key)
	}
	d := resourceUpdateData(t, ResourceLiteLLMTeam(), "team-1", before, after)
	if err := resourceLiteLLMTeamUpdate(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatal(err)
	}
	metadata := state.info["metadata"].(map[string]interface{})
	if _, present := metadata["owner"]; present {
		t.Error("removed owned metadata entry survived")
	}
	if _, present := metadata["soft_budget_alerting_emails"]; present {
		t.Error("removed alert emails survived")
	}
	if metadata["external"] != "live" || metadata["require_trace_id"] != true {
		t.Errorf("foreign metadata was overwritten: %#v", metadata)
	}
	if state.info["soft_budget"] != nil {
		t.Errorf("soft budget not cleared: %v", state.info["soft_budget"])
	}
	if !reflect.DeepEqual(metadata["tags"], []interface{}{}) {
		t.Errorf("tags not cleared: %v", metadata["tags"])
	}
	for _, key := range []string{"model_rpm_limit", "model_tpm_limit"} {
		if !reflect.DeepEqual(metadata[key], map[string]interface{}{}) {
			t.Errorf("%s not cleared: %v", key, metadata[key])
		}
	}
	member := state.info["team_member_budget_table"].(map[string]interface{})
	if !reflect.DeepEqual(member["model_max_budget"], map[string]interface{}{}) {
		t.Errorf("member per-model budget not cleared: %v", member["model_max_budget"])
	}
}

func TestTeamUpdateSafetyIntentionalPermissionsUseDedicatedEndpoint(t *testing.T) {
	state := teamSafetyState()
	srv := newTeamSafetyServer(t, state)
	defer srv.Close()
	before, after := teamSafetyConfig(), teamSafetyConfig()
	after["team_member_permissions"] = []interface{}{}
	d := resourceUpdateData(t, ResourceLiteLLMTeam(), "team-1", before, after)
	if err := resourceLiteLLMTeamUpdate(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatal(err)
	}
	if state.permissionWrites != 1 {
		t.Errorf("permission writes=%d, want 1", state.permissionWrites)
	}
	if !reflect.DeepEqual(state.info["team_member_permissions"], []interface{}{}) {
		t.Errorf("permissions not cleared: %v", state.info["team_member_permissions"])
	}
}
