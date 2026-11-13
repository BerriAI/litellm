package litellm

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

// Match /project/update: request metadata replaces stored metadata, tags and
// per-model rate limits are moved into it, and supplied budget fields write the
// associated budget row. Omitted fields leave the corresponding row untouched.
type projectUpdateServer struct {
	mu           sync.Mutex
	project      map[string]interface{}
	budgetWrites int
}

func newProjectUpdateServer(t *testing.T) *projectUpdateServer {
	t.Helper()
	decoder := json.NewDecoder(strings.NewReader(`{
		"project_id":"proj-123", "team_id":"team-1", "budget_id":"bud-9",
		"project_alias":"before", "description":"before description",
		"models":["gpt-4"], "blocked":true,
		"metadata":{
			"owner":"terraform", "remove_me":"old", "foreign_string":"planned",
			"eur_budget":"0", "encoded_models":"[\"gpt-4\"]", "tags":["production"],
			"model_rpm_limit":{"gpt-4":100,"server-model":200},
			"model_tpm_limit":{"gpt-4":10000,"server-model":20000},
			"foreign":{"revision":9007199254740993,"enabled":true,"items":[1,null,{"nested":"value"}]}
		},
		"litellm_budget_table":{
			"max_budget":100, "soft_budget":80, "max_parallel_requests":10,
			"rpm_limit":500, "tpm_limit":5000, "budget_duration":"30d",
			"model_max_budget":{"gpt-4":40},
			"budget_reset_at":"2026-12-01T00:00:00Z", "updated_at":"2026-11-01T00:00:00Z"
		}
	}`))
	decoder.UseNumber()
	state := &projectUpdateServer{}
	if err := decoder.Decode(&state.project); err != nil {
		t.Fatal(err)
	}
	return state
}

func (s *projectUpdateServer) serve(t *testing.T) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.mu.Lock()
		defer s.mu.Unlock()
		switch r.URL.Path {
		case endpointProjectInfo:
			if r.Method != http.MethodGet || r.URL.Query().Get("project_id") != "proj-123" {
				t.Errorf("unexpected project info request: %s %s", r.Method, r.URL)
			}
		case endpointProjectUpdate:
			if r.Method != http.MethodPost {
				t.Errorf("unexpected update method: %s", r.Method)
			}
			var update map[string]interface{}
			decoder := json.NewDecoder(r.Body)
			decoder.UseNumber()
			if err := decoder.Decode(&update); err != nil {
				t.Errorf("decode project update: %v", err)
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			if update["project_id"] != "proj-123" {
				t.Errorf("project update must identify proj-123, got %v", update["project_id"])
			}
			metadata, hasMetadata := update["metadata"].(map[string]interface{})
			for _, key := range []string{"tags", "model_rpm_limit", "model_tpm_limit"} {
				if value, ok := update[key]; ok && value != nil {
					if !hasMetadata {
						metadata = map[string]interface{}{}
						hasMetadata = true
					}
					metadata[key] = value
				}
			}
			if hasMetadata {
				s.project["metadata"] = metadata
			}
			budget := s.project["litellm_budget_table"].(map[string]interface{})
			wroteBudget := false
			for _, key := range []string{"budget_id", "max_budget", "soft_budget", "max_parallel_requests", "rpm_limit", "tpm_limit", "budget_duration", "model_max_budget"} {
				if value, ok := update[key]; ok && (value != nil || key == "max_budget") {
					budget[key] = value
					wroteBudget = true
				}
			}
			if wroteBudget {
				s.budgetWrites++
				budget["updated_at"] = "2026-11-13T00:00:00Z"
			}
			for _, key := range []string{"team_id", "project_alias", "description", "models", "blocked"} {
				if value, ok := update[key]; ok && value != nil {
					s.project[key] = value
				}
			}
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		if err := json.NewEncoder(w).Encode(s.project); err != nil {
			t.Errorf("encode project response: %v", err)
		}
	}))
}

func (s *projectUpdateServer) snapshot(t *testing.T) (map[string]interface{}, int) {
	t.Helper()
	s.mu.Lock()
	defer s.mu.Unlock()
	body, err := json.Marshal(s.project)
	if err != nil {
		t.Fatal(err)
	}
	var result map[string]interface{}
	decoder := json.NewDecoder(strings.NewReader(string(body)))
	decoder.UseNumber()
	if err := decoder.Decode(&result); err != nil {
		t.Fatal(err)
	}
	return result, s.budgetWrites
}

func projectUpdatePlan(t *testing.T, client *Client) (*schema.Resource, map[string]interface{}) {
	t.Helper()
	resource := resourceLiteLLMProject()
	d := schema.TestResourceDataRaw(t, resource.Schema, map[string]interface{}{
		"team_id":          "team-1",
		"model_rpm_limit":  map[string]interface{}{"gpt-4": 100},
		"model_tpm_limit":  map[string]interface{}{"gpt-4": 10000},
		"model_max_budget": map[string]interface{}{"gpt-4": 40.0},
	})
	d.SetId("proj-123")
	if err := resource.Read(d, client); err != nil {
		t.Fatalf("initial project read: %v", err)
	}
	before := map[string]interface{}{}
	for key, field := range resource.Schema {
		if !field.Computed {
			before[key] = d.Get(key)
		}
	}
	return resource, before
}

func copyProjectPlan(before map[string]interface{}) map[string]interface{} {
	after := map[string]interface{}{}
	for key, value := range before {
		after[key] = value
	}
	return after
}

func TestResourceLiteLLMProjectUpdatePreservesUnchangedServerState(t *testing.T) {
	for _, test := range []struct {
		name  string
		key   string
		value interface{}
	}{
		{"alias", "project_alias", "renamed"},
		{"description", "description", "new description"},
		{"models", "models", []interface{}{"gpt-4", "server-model"}},
	} {
		t.Run(test.name, func(t *testing.T) {
			state := newProjectUpdateServer(t)
			server := state.serve(t)
			defer server.Close()
			client := NewClient(server.URL, "test-key", true)
			resource, before := projectUpdatePlan(t, client)
			after := copyProjectPlan(before)
			after[test.key] = test.value
			d := resourceUpdateData(t, resource, "proj-123", before, after)
			// These controls changed outside Terraform after the plan was made.
			state.mu.Lock()
			state.project["team_id"] = "live-team"
			state.project["blocked"] = false
			state.project["metadata"].(map[string]interface{})["tags"] = []interface{}{"live-tag"}
			state.mu.Unlock()
			want, _ := state.snapshot(t)
			want[test.key] = test.value
			if err := resource.Update(d, client); err != nil {
				t.Fatalf("project update: %v", err)
			}
			got, budgetWrites := state.snapshot(t)
			if !reflect.DeepEqual(got, want) {
				t.Errorf("unrelated update changed server controls or metadata\ngot: %#v\nwant: %#v", got, want)
			}
			if budgetWrites != 0 {
				t.Errorf("unrelated update rewrote the budget row %d times", budgetWrites)
			}
		})
	}
}

func TestResourceLiteLLMProjectUpdateMergesLiveMetadataDelta(t *testing.T) {
	state := newProjectUpdateServer(t)
	server := state.serve(t)
	defer server.Close()
	client := NewClient(server.URL, "test-key", true)
	resource, before := projectUpdatePlan(t, client)
	after := copyProjectPlan(before)
	metadata := map[string]interface{}{}
	for key, value := range before["metadata"].(map[string]interface{}) {
		metadata[key] = value
	}
	metadata["owner"] = "new-owner"
	delete(metadata, "remove_me")
	after["metadata"] = metadata
	d := resourceUpdateData(t, resource, "proj-123", before, after)
	state.mu.Lock()
	live := state.project["metadata"].(map[string]interface{})
	live["foreign_string"] = "changed-after-plan"
	live["foreign"].(map[string]interface{})["revision"] = json.Number("9007199254740995")
	live["new_after_plan"] = []interface{}{true, json.Number("9007199254740997")}
	state.mu.Unlock()
	want, _ := state.snapshot(t)
	wantMetadata := want["metadata"].(map[string]interface{})
	wantMetadata["owner"] = "new-owner"
	delete(wantMetadata, "remove_me")
	if err := resource.Update(d, client); err != nil {
		t.Fatalf("metadata update: %v", err)
	}
	got, budgetWrites := state.snapshot(t)
	if !reflect.DeepEqual(got, want) {
		t.Errorf("metadata delta lost or overwrote live server data\ngot: %#v\nwant: %#v", got, want)
	}
	if budgetWrites != 0 {
		t.Errorf("metadata update rewrote the budget row %d times", budgetWrites)
	}
}

func TestResourceLiteLLMProjectUpdateMetadataControlsPreserveLiveData(t *testing.T) {
	for _, test := range []struct {
		name    string
		changes map[string]interface{}
	}{
		{"tags", map[string]interface{}{"tags": []interface{}{"new-tag"}}},
		{"rpm", map[string]interface{}{"model_rpm_limit": map[string]interface{}{"gpt-4": 0}}},
		{"tpm", map[string]interface{}{"model_tpm_limit": map[string]interface{}{}}},
		{"combined", map[string]interface{}{
			"tags": []interface{}{}, "model_rpm_limit": map[string]interface{}{"gpt-4": 250},
			"model_tpm_limit": map[string]interface{}{"gpt-4": 25000},
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			state := newProjectUpdateServer(t)
			server := state.serve(t)
			defer server.Close()
			client := NewClient(server.URL, "test-key", true)
			resource, before := projectUpdatePlan(t, client)
			after := copyProjectPlan(before)
			for key, value := range test.changes {
				after[key] = value
			}
			if test.name == "combined" {
				metadata := copyProjectPlan(before["metadata"].(map[string]interface{}))
				metadata["owner"] = "combined-owner"
				delete(metadata, "remove_me")
				after["metadata"] = metadata
			}
			d := resourceUpdateData(t, resource, "proj-123", before, after)
			state.mu.Lock()
			state.project["metadata"].(map[string]interface{})["foreign_string"] = "live-value"
			state.mu.Unlock()
			want, _ := state.snapshot(t)
			for key := range test.changes {
				body, err := json.Marshal(d.Get(key))
				if err != nil {
					t.Fatal(err)
				}
				decoder := json.NewDecoder(strings.NewReader(string(body)))
				decoder.UseNumber()
				var value interface{}
				if err := decoder.Decode(&value); err != nil {
					t.Fatal(err)
				}
				want["metadata"].(map[string]interface{})[key] = value
			}
			if test.name == "combined" {
				metadata := want["metadata"].(map[string]interface{})
				metadata["owner"] = "combined-owner"
				delete(metadata, "remove_me")
			}
			if err := resource.Update(d, client); err != nil {
				t.Fatalf("metadata control update: %v", err)
			}
			got, budgetWrites := state.snapshot(t)
			if !reflect.DeepEqual(got, want) {
				t.Errorf("metadata control update changed unrelated live state\ngot: %#v\nwant: %#v", got, want)
			}
			if budgetWrites != 0 {
				t.Errorf("metadata control update rewrote budget %d times", budgetWrites)
			}
		})
	}
}

func TestResourceLiteLLMProjectUpdateIntentionalZeroAndEmptyChanges(t *testing.T) {
	for key, value := range map[string]interface{}{
		"project_alias": "", "description": "", "blocked": false, "models": []interface{}{},
		"max_budget": 0.0, "soft_budget": 0.0, "rpm_limit": 0, "tpm_limit": 0,
		"max_parallel_requests": 0, "budget_duration": "", "model_max_budget": map[string]interface{}{},
	} {
		t.Run(key, func(t *testing.T) {
			state := newProjectUpdateServer(t)
			server := state.serve(t)
			defer server.Close()
			client := NewClient(server.URL, "test-key", true)
			resource, before := projectUpdatePlan(t, client)
			after := copyProjectPlan(before)
			after[key] = value
			d := resourceUpdateData(t, resource, "proj-123", before, after)
			want, _ := state.snapshot(t)
			body, err := json.Marshal(d.Get(key))
			if err != nil {
				t.Fatal(err)
			}
			decoder := json.NewDecoder(strings.NewReader(string(body)))
			decoder.UseNumber()
			var cleared interface{}
			if err := decoder.Decode(&cleared); err != nil {
				t.Fatal(err)
			}
			wantBudgetWrites := 0
			switch key {
			case "max_budget", "soft_budget", "rpm_limit", "tpm_limit", "max_parallel_requests", "budget_duration", "model_max_budget":
				budget := want["litellm_budget_table"].(map[string]interface{})
				budget[key] = cleared
				budget["updated_at"] = "2026-11-13T00:00:00Z"
				wantBudgetWrites = 1
			default:
				want[key] = cleared
			}
			if err := resource.Update(d, client); err != nil {
				t.Fatalf("intentional clear update: %v", err)
			}
			got, budgetWrites := state.snapshot(t)
			if !reflect.DeepEqual(got, want) {
				t.Errorf("intentional clear changed wrong server fields\ngot: %#v\nwant: %#v", got, want)
			}
			if budgetWrites != wantBudgetWrites {
				t.Errorf("expected %d intentional budget writes, got %d", wantBudgetWrites, budgetWrites)
			}
			if key == "project_alias" || key == "description" || key == "budget_duration" {
				if d.Get(key) != "" {
					t.Errorf("read restored cleared %s: %v", key, d.Get(key))
				}
			}
		})
	}
}

func TestResourceLiteLLMProjectReadClearsAbsentBudgetValues(t *testing.T) {
	for _, budget := range []string{"null", "{}", `{"max_budget":null,"soft_budget":null,"max_parallel_requests":null,"tpm_limit":null,"rpm_limit":null,"budget_duration":null}`} {
		t.Run(budget, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != endpointProjectInfo || r.Method != http.MethodGet {
					t.Errorf("unexpected request: %s %s", r.Method, r.URL)
				}
				w.Write([]byte(`{"project_id":"proj-123","team_id":"team-1","budget_id":null,"litellm_budget_table":` + budget + `}`))
			}))
			defer server.Close()
			resource := resourceLiteLLMProject()
			d := schema.TestResourceDataRaw(t, resource.Schema, map[string]interface{}{
				"team_id": "team-1", "budget_id": "old-budget", "max_budget": 100.0, "soft_budget": 80.0,
				"max_parallel_requests": 10, "rpm_limit": 500, "tpm_limit": 5000, "budget_duration": "30d",
			})
			d.SetId("proj-123")
			if err := resource.Read(d, NewClient(server.URL, "test-key", true)); err != nil {
				t.Fatalf("read absent budget: %v", err)
			}
			for key, want := range map[string]interface{}{
				"budget_id": "", "max_budget": 0.0, "soft_budget": 0.0, "max_parallel_requests": 0,
				"rpm_limit": 0, "tpm_limit": 0, "budget_duration": "",
			} {
				if got := d.Get(key); got != want {
					t.Errorf("read kept stale %s: got %v, want %v", key, got, want)
				}
			}
		})
	}
}

func TestResourceLiteLLMProjectUpdateStopsOnCurrentMetadataError(t *testing.T) {
	for _, test := range []struct {
		name   string
		status int
		body   string
	}{
		{"http", http.StatusInternalServerError, `{"error":"unavailable"}`},
		{"malformed", http.StatusOK, `{"metadata":`},
		{"non_object", http.StatusOK, `{"metadata":["invalid"]}`},
	} {
		t.Run(test.name, func(t *testing.T) {
			updates := 0
			var mu sync.Mutex
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				mu.Lock()
				defer mu.Unlock()
				if r.URL.Path == endpointProjectUpdate {
					updates++
					w.Write([]byte(`{}`))
					return
				}
				if r.URL.Path != endpointProjectInfo || r.Method != http.MethodGet {
					t.Errorf("unexpected request: %s %s", r.Method, r.URL)
				}
				w.WriteHeader(test.status)
				w.Write([]byte(test.body))
			}))
			defer server.Close()
			resource := resourceLiteLLMProject()
			d := resourceUpdateData(t, resource, "proj-123",
				map[string]interface{}{"team_id": "team-1", "metadata": map[string]interface{}{"owner": "before"}},
				map[string]interface{}{"team_id": "team-1", "metadata": map[string]interface{}{"owner": "after"}},
			)
			if err := resource.Update(d, NewClient(server.URL, "test-key", true)); err == nil {
				t.Fatal("expected current metadata error")
			}
			mu.Lock()
			defer mu.Unlock()
			if updates != 0 {
				t.Errorf("metadata read error allowed %d project writes", updates)
			}
		})
	}
}

func TestResourceLiteLLMProjectCreatePreservesDeclaredStringMetadata(t *testing.T) {
	var savedMetadata map[string]interface{}
	var mu sync.Mutex
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		switch r.URL.Path {
		case endpointProjectNew:
			var created map[string]interface{}
			if err := json.NewDecoder(r.Body).Decode(&created); err != nil {
				t.Errorf("decode project create: %v", err)
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			if created["team_id"] != "team-1" {
				t.Errorf("create lost required team: %v", created["team_id"])
			}
			savedMetadata = created["metadata"].(map[string]interface{})
			savedMetadata["tags"] = created["tags"]
		case endpointProjectInfo:
			if r.Method != http.MethodGet || r.URL.Query().Get("project_id") != "proj-123" {
				t.Errorf("unexpected project read: %s %s", r.Method, r.URL)
			}
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		json.NewEncoder(w).Encode(map[string]interface{}{
			"project_id": "proj-123", "team_id": "team-1", "metadata": savedMetadata,
		})
	}))
	defer server.Close()
	metadata := map[string]interface{}{"encoded_models": `["gpt-4"]`, "eur_budget": "0", "owner": "terraform"}
	resource := resourceLiteLLMProject()
	d := schema.TestResourceDataRaw(t, resource.Schema, map[string]interface{}{
		"team_id": "team-1", "metadata": metadata, "tags": []interface{}{"production"},
	})
	if err := resource.Create(d, NewClient(server.URL, "test-key", true)); err != nil {
		t.Fatalf("project create: %v", err)
	}
	mu.Lock()
	defer mu.Unlock()
	wantSaved := copyProjectPlan(metadata)
	wantSaved["tags"] = []interface{}{"production"}
	if !reflect.DeepEqual(savedMetadata, wantSaved) {
		t.Errorf("create changed declared metadata: got %v, want %v", savedMetadata, wantSaved)
	}
	if !reflect.DeepEqual(d.Get("metadata"), metadata) {
		t.Errorf("read changed declared string metadata: %v", d.Get("metadata"))
	}
}
