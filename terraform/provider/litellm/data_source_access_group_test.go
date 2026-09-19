package litellm

import (
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func TestAccessGroupDataSourceRead(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" || r.URL.Path != "/access_group/prod-models/info" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.Write(accessGroupInfoJSON("prod-models", []string{"gpt-4", "claude-3"}, 2))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMAccessGroup().Schema, map[string]interface{}{
		"access_group": "prod-models",
	})

	if err := dataSourceLiteLLMAccessGroupRead(d, client); err != nil {
		t.Fatalf("data source read failed: %v", err)
	}

	if d.Id() != "prod-models" {
		t.Fatalf("expected ID 'prod-models', got %q", d.Id())
	}
	wantModels := []interface{}{"gpt-4", "claude-3"}
	if !reflect.DeepEqual(d.Get("model_names"), wantModels) {
		t.Fatalf("expected model_names %v, got %v", wantModels, d.Get("model_names"))
	}
	if d.Get("deployment_count").(int) != 2 {
		t.Fatalf("expected deployment_count 2, got %v", d.Get("deployment_count"))
	}
}

func TestAccessGroupDataSourceReadNotFound(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMAccessGroup().Schema, map[string]interface{}{
		"access_group": "missing",
	})

	if err := dataSourceLiteLLMAccessGroupRead(d, client); err == nil {
		t.Fatal("expected error for missing access group, got nil")
	}
}

func TestAccessGroupsDataSourceRead(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" || r.URL.Path != "/access_group/list" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.Write([]byte(`{"access_groups": [` +
			`{"access_group": "group-a", "model_names": ["gpt-4"], "deployment_count": 1},` +
			`{"access_group": "group-b", "model_names": ["claude-3"], "deployment_count": 2}]}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMAccessGroups().Schema, map[string]interface{}{})

	if err := dataSourceLiteLLMAccessGroupsRead(d, client); err != nil {
		t.Fatalf("data source read failed: %v", err)
	}

	groups := d.Get("access_groups").([]interface{})
	if len(groups) != 2 {
		t.Fatalf("expected 2 access groups, got %d", len(groups))
	}
	first := groups[0].(map[string]interface{})
	if first["access_group"] != "group-a" {
		t.Fatalf("expected first access_group 'group-a', got %v", first["access_group"])
	}
	if !reflect.DeepEqual(first["model_names"], []interface{}{"gpt-4"}) {
		t.Fatalf("expected first model_names [gpt-4], got %v", first["model_names"])
	}
	if first["deployment_count"].(int) != 1 {
		t.Fatalf("expected first deployment_count 1, got %v", first["deployment_count"])
	}
	if !reflect.DeepEqual(d.Get("ids"), []interface{}{"group-a", "group-b"}) {
		t.Fatalf("expected ids [group-a group-b], got %v", d.Get("ids"))
	}
}
