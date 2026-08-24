package litellm

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func TestDataSourceUserRead_MapsFields(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/user/info" || r.Method != http.MethodGet {
			t.Errorf("expected GET /user/info, got %s %s", r.Method, r.URL.Path)
		}
		if got := r.URL.Query().Get("user_id"); got != "u-ds" {
			t.Errorf("expected user_id query 'u-ds', got %q", got)
		}
		w.Write(userInfoBody("u-ds", map[string]interface{}{
			"user_email":       "carol@example.com",
			"user_role":        "internal_user",
			"max_budget":       42.0,
			"spend":            1.5,
			"models":           []interface{}{"gpt-4o"},
			"model_max_budget": map[string]interface{}{"gpt-4o": map[string]interface{}{"max_budget": 2.0}},
		}))
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMUser().Schema, map[string]interface{}{
		"user_id": "u-ds",
	})

	if err := dataSourceLiteLLMUserRead(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("read failed: %v", err)
	}

	if d.Id() != "u-ds" {
		t.Fatalf("expected ID 'u-ds', got %q", d.Id())
	}
	if got := d.Get("user_email").(string); got != "carol@example.com" {
		t.Errorf("expected user_email 'carol@example.com', got %q", got)
	}
	if got := d.Get("spend").(float64); got != 1.5 {
		t.Errorf("expected spend 1.5, got %v", got)
	}
	models := d.Get("models").([]interface{})
	if len(models) != 1 || models[0] != "gpt-4o" {
		t.Errorf("expected models [gpt-4o], got %v", models)
	}
	var mmb map[string]interface{}
	if err := json.Unmarshal([]byte(d.Get("model_max_budget").(string)), &mmb); err != nil {
		t.Fatalf("model_max_budget in state is not valid JSON: %v", err)
	}
	if _, ok := mmb["gpt-4o"]; !ok {
		t.Errorf("expected gpt-4o key in model_max_budget state, got %v", mmb)
	}
}

func TestDataSourceUsersRead_FiltersAndMapsList(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/user/list" || r.Method != http.MethodGet {
			t.Errorf("expected GET /user/list, got %s %s", r.Method, r.URL.Path)
		}
		query := r.URL.Query()
		if got := query.Get("role"); got != "internal_user" {
			t.Errorf("expected role query 'internal_user', got %q", got)
		}
		if got := query.Get("page"); got != "2" {
			t.Errorf("expected page query '2', got %q", got)
		}
		if got := query.Get("page_size"); got != "50" {
			t.Errorf("expected page_size query '50', got %q", got)
		}
		body, _ := json.Marshal(map[string]interface{}{
			"users": []map[string]interface{}{
				{
					"user_id":    "u-1",
					"user_email": "one@example.com",
					"user_role":  "internal_user",
					"max_budget": 10.0,
					"spend":      2.0,
					"tpm_limit":  100,
					"key_count":  3,
				},
				{
					"user_id":    "u-2",
					"user_email": "two@example.com",
					"teams":      []string{"team-x"},
				},
			},
			"total":       52,
			"page":        2,
			"page_size":   50,
			"total_pages": 2,
		})
		w.Write(body)
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMUsers().Schema, map[string]interface{}{
		"role":      "internal_user",
		"page":      2,
		"page_size": 50,
	})

	if err := dataSourceLiteLLMUsersRead(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("read failed: %v", err)
	}

	users := d.Get("users").([]interface{})
	if len(users) != 2 {
		t.Fatalf("expected 2 users, got %d", len(users))
	}
	first := users[0].(map[string]interface{})
	if got := first["user_id"].(string); got != "u-1" {
		t.Errorf("expected first user_id 'u-1', got %q", got)
	}
	if got := first["spend"].(float64); got != 2.0 {
		t.Errorf("expected first spend 2.0, got %v", got)
	}
	if got := first["tpm_limit"].(int); got != 100 {
		t.Errorf("expected first tpm_limit 100, got %d", got)
	}
	if got := first["key_count"].(int); got != 3 {
		t.Errorf("expected first key_count 3, got %d", got)
	}
	second := users[1].(map[string]interface{})
	teams := second["teams"].([]interface{})
	if len(teams) != 1 || teams[0] != "team-x" {
		t.Errorf("expected second user teams [team-x], got %v", teams)
	}
	ids := d.Get("ids").([]interface{})
	if len(ids) != 2 || ids[0] != "u-1" || ids[1] != "u-2" {
		t.Errorf("expected ids [u-1 u-2], got %v", ids)
	}
	if got := d.Get("total").(int); got != 52 {
		t.Errorf("expected total 52, got %d", got)
	}
	if got := d.Get("total_pages").(int); got != 2 {
		t.Errorf("expected total_pages 2, got %d", got)
	}
}
