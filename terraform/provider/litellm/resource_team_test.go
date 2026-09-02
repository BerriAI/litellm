package litellm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
	"github.com/hashicorp/terraform-plugin-sdk/v2/terraform"
)

func newTeamTestServer(t *testing.T, captured *map[string]interface{}, infoBody string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case endpointTeamNew, endpointTeamUpdate:
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, captured)
			w.Write([]byte(`{}`))
		case endpointTeamInfo:
			w.Write([]byte(infoBody))
		case endpointTeamPermissionsList:
			w.Write([]byte(`{"team_id":"team-1","team_member_permissions":[],"all_available_permissions":[]}`))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
}

const teamInfoWithSoftBudget = `{
  "team_id": "team-1",
  "team_info": {
    "team_id": "team-1",
    "team_alias": "insights",
    "max_budget": 750.0,
    "soft_budget": 600.0,
    "models": ["claude-haiku-4-5"],
    "metadata": {
      "department": "customer-insights",
      "tags": ["team:customer-insights", "environment:production"],
      "soft_budget_alerting_emails": ["finops@example.com"],
      "team_member_budget_id": "budget-1"
    }
  },
  "keys": [],
  "team_memberships": []
}`

func TestTeamCreateSendsSoftBudgetTagsAndAlertEmails(t *testing.T) {
	var captured map[string]interface{}
	srv := newTeamTestServer(t, &captured, teamInfoWithSoftBudget)
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, ResourceLiteLLMTeam().Schema, map[string]interface{}{
		"team_alias":                  "insights",
		"max_budget":                  750.0,
		"soft_budget":                 600.0,
		"tags":                        []interface{}{"team:customer-insights", "environment:production"},
		"soft_budget_alerting_emails": []interface{}{"finops@example.com"},
		"metadata":                    map[string]interface{}{"department": "customer-insights"},
	})

	if err := resourceLiteLLMTeamCreate(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("create failed: %v", err)
	}

	if got := captured["soft_budget"]; got != 600.0 {
		t.Fatalf("payload soft_budget = %v, want 600", got)
	}
	wantTags := []interface{}{"team:customer-insights", "environment:production"}
	if got := captured["tags"]; !reflect.DeepEqual(got, wantTags) {
		t.Fatalf("payload tags = %v, want %v", got, wantTags)
	}
	wantMetadata := map[string]interface{}{
		"department":                  "customer-insights",
		"soft_budget_alerting_emails": []interface{}{"finops@example.com"},
	}
	if got := captured["metadata"]; !reflect.DeepEqual(got, wantMetadata) {
		t.Fatalf("payload metadata = %v, want %v", got, wantMetadata)
	}
}

func TestTeamReadMapsTeamInfoEnvelope(t *testing.T) {
	var captured map[string]interface{}
	srv := newTeamTestServer(t, &captured, teamInfoWithSoftBudget)
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, ResourceLiteLLMTeam().Schema, map[string]interface{}{})
	d.SetId("team-1")

	if err := resourceLiteLLMTeamRead(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("read failed: %v", err)
	}

	if got := d.Get("team_alias"); got != "insights" {
		t.Fatalf("team_alias = %v, want insights", got)
	}
	if got := d.Get("soft_budget"); got != 600.0 {
		t.Fatalf("soft_budget = %v, want 600", got)
	}
	if got := d.Get("max_budget"); got != 750.0 {
		t.Fatalf("max_budget = %v, want 750", got)
	}
	wantTags := []interface{}{"team:customer-insights", "environment:production"}
	if got := d.Get("tags"); !reflect.DeepEqual(got, wantTags) {
		t.Fatalf("tags = %v, want %v", got, wantTags)
	}
	wantEmails := []interface{}{"finops@example.com"}
	if got := d.Get("soft_budget_alerting_emails"); !reflect.DeepEqual(got, wantEmails) {
		t.Fatalf("soft_budget_alerting_emails = %v, want %v", got, wantEmails)
	}
	wantMetadata := map[string]interface{}{"department": "customer-insights"}
	if got := d.Get("metadata"); !reflect.DeepEqual(got, wantMetadata) {
		t.Fatalf("metadata = %v, want %v (server-managed team_member_budget_id dropped)", got, wantMetadata)
	}
}

func TestTeamUpdateClearsRemovedTagsAndSoftBudget(t *testing.T) {
	var captured map[string]interface{}
	srv := newTeamTestServer(t, &captured, `{"team_id":"team-1","team_info":{"team_id":"team-1","team_alias":"insights"},"keys":[],"team_memberships":[]}`)
	defer srv.Close()

	res := ResourceLiteLLMTeam()
	priorData := schema.TestResourceDataRaw(t, res.Schema, map[string]interface{}{
		"team_alias":                  "insights",
		"soft_budget":                 600.0,
		"tags":                        []interface{}{"team:to-be-removed"},
		"soft_budget_alerting_emails": []interface{}{"ops@example.com"},
		"metadata":                    map[string]interface{}{"department": "eng"},
	})
	priorData.SetId("team-1")
	prior := priorData.State()
	config := terraform.NewResourceConfigRaw(map[string]interface{}{
		"team_alias": "insights",
		"metadata":   map[string]interface{}{"department": "eng"},
	})
	diff, err := res.Diff(context.Background(), prior, config, nil)
	if err != nil {
		t.Fatalf("diff failed: %v", err)
	}
	d, err := schema.InternalMap(res.Schema).Data(prior, diff)
	if err != nil {
		t.Fatalf("data failed: %v", err)
	}

	if err := resourceLiteLLMTeamUpdate(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("update failed: %v", err)
	}

	if got, ok := captured["soft_budget"]; !ok || got != nil {
		t.Fatalf("payload soft_budget = %v (present=%v), want explicit null", got, ok)
	}
	if got := captured["tags"]; !reflect.DeepEqual(got, []interface{}{}) {
		t.Fatalf("payload tags = %v, want []", got)
	}
	if got := captured["metadata"]; !reflect.DeepEqual(got, map[string]interface{}{"department": "eng"}) {
		t.Fatalf("payload metadata = %v, want department only", got)
	}
}

func TestTeamReadClearsSoftBudgetWhenProxyReturnsNull(t *testing.T) {
	var captured map[string]interface{}
	srv := newTeamTestServer(t, &captured, `{"team_id":"team-1","team_info":{"team_id":"team-1","team_alias":"insights","soft_budget":null},"keys":[],"team_memberships":[]}`)
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, ResourceLiteLLMTeam().Schema, map[string]interface{}{
		"team_alias":  "insights",
		"soft_budget": 600.0,
	})
	d.SetId("team-1")

	if err := resourceLiteLLMTeamRead(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("read failed: %v", err)
	}

	if got := d.Get("soft_budget"); got != 0.0 {
		t.Fatalf("soft_budget = %v, want cleared after the proxy returned null", got)
	}
}

func newTeamResourceData(t *testing.T, raw map[string]interface{}) *schema.ResourceData {
	t.Helper()
	return schema.TestResourceDataRaw(t, ResourceLiteLLMTeam().Schema, raw)
}

func TestBuildTeamDataIncludesNewFields(t *testing.T) {
	d := newTeamResourceData(t, map[string]interface{}{
		"team_alias":                  "eng",
		"model_aliases":               map[string]interface{}{"gpt": "gpt-5.2"},
		"guardrails":                  []interface{}{"pii-mask"},
		"prompts":                     []interface{}{"prompt-1"},
		"team_member_budget":          5.0,
		"team_member_budget_duration": "30d",
		"team_member_rpm_limit":       10,
		"team_member_tpm_limit":       1000,
		"team_member_key_duration":    "7d",
		"allowed_passthrough_routes":  []interface{}{"/vertex-ai"},
	})

	data := buildTeamData(d, "team-1")

	for _, k := range []string{
		"model_aliases", "guardrails", "prompts", "team_member_budget",
		"team_member_budget_duration", "team_member_rpm_limit", "team_member_tpm_limit",
		"team_member_key_duration", "allowed_passthrough_routes",
	} {
		if _, ok := data[k]; !ok {
			t.Errorf("buildTeamData missing %s", k)
		}
	}
	if data["team_id"] != "team-1" || data["team_alias"] != "eng" {
		t.Errorf("identity fields wrong: %v", data)
	}
}

func TestTeamReadMapsNewFields(t *testing.T) {
	var captured map[string]interface{}
	srv := newTeamTestServer(t, &captured, `{
		"team_id": "team-1",
		"team_info": {
			"team_id": "team-1",
			"team_alias": "eng",
			"guardrails": ["pii-mask"],
			"team_member_budget": 5.0,
			"team_member_rpm_limit": 10
		}
	}`)
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newTeamResourceData(t, map[string]interface{}{"team_alias": "config-alias"})
	d.SetId("team-1")

	if err := resourceLiteLLMTeamRead(d, client); err != nil {
		t.Fatalf("read returned error: %v", err)
	}
	if got := d.Get("guardrails").([]interface{}); len(got) != 1 || got[0] != "pii-mask" {
		t.Errorf("guardrails = %v, want [pii-mask]", got)
	}
	if got := d.Get("team_member_budget").(float64); got != 5.0 {
		t.Errorf("team_member_budget = %v, want 5.0", got)
	}
	if got := d.Get("team_member_rpm_limit").(int); got != 10 {
		t.Errorf("team_member_rpm_limit = %v, want 10", got)
	}
}

// rpm_limit_type / tpm_limit_type are accepted by /team/new but not
// /team/update, so create must send them and update must not.
func TestTeamLimitTypesSentOnCreateOnly(t *testing.T) {
	var captured map[string]interface{}
	srv := newTeamTestServer(t, &captured, `{"team_id": "x", "team_info": {"team_alias": "eng"}}`)
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newTeamResourceData(t, map[string]interface{}{
		"team_alias":     "eng",
		"rpm_limit_type": "guaranteed_throughput",
		"tpm_limit_type": "best_effort_throughput",
	})

	if err := resourceLiteLLMTeamCreate(d, client); err != nil {
		t.Fatalf("create returned error: %v", err)
	}
	if captured["rpm_limit_type"] != "guaranteed_throughput" || captured["tpm_limit_type"] != "best_effort_throughput" {
		t.Errorf("create payload missing limit types: %v", captured)
	}

	captured = nil
	if err := resourceLiteLLMTeamUpdate(d, client); err != nil {
		t.Fatalf("update returned error: %v", err)
	}
	for _, k := range []string{"rpm_limit_type", "tpm_limit_type"} {
		if _, present := captured[k]; present {
			t.Errorf("update payload unexpectedly contains %s", k)
		}
	}
}
