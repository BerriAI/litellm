package litellm

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func accessGroupTestData(t *testing.T, raw map[string]interface{}) *schema.ResourceData {
	t.Helper()
	return schema.TestResourceDataRaw(t, resourceLiteLLMAccessGroup().Schema, raw)
}

func accessGroupInfoJSON(name string, modelNames []string, deploymentCount int) []byte {
	body, _ := json.Marshal(accessGroupInfoResponse{
		AccessGroup:     name,
		ModelNames:      modelNames,
		DeploymentCount: deploymentCount,
	})
	return body
}

func TestAccessGroupCreate(t *testing.T) {
	var createPayload map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method + " " + r.URL.Path {
		case "POST /access_group/new":
			if err := json.NewDecoder(r.Body).Decode(&createPayload); err != nil {
				t.Errorf("failed to decode create payload: %v", err)
			}
			w.Write([]byte(`{"access_group": "prod-models", "models_updated": 2}`))
		case "GET /access_group/prod-models/info":
			w.Write(accessGroupInfoJSON("prod-models", []string{"gpt-4", "claude-3"}, 2))
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := accessGroupTestData(t, map[string]interface{}{
		"access_group": "prod-models",
		"model_names":  []interface{}{"gpt-4", "claude-3"},
	})

	if err := resourceLiteLLMAccessGroupCreate(d, client); err != nil {
		t.Fatalf("create failed: %v", err)
	}

	if createPayload["access_group"] != "prod-models" {
		t.Fatalf("expected access_group 'prod-models' in payload, got %v", createPayload["access_group"])
	}
	wantModels := []interface{}{"gpt-4", "claude-3"}
	if !reflect.DeepEqual(createPayload["model_names"], wantModels) {
		t.Fatalf("expected model_names %v in payload, got %v", wantModels, createPayload["model_names"])
	}
	if d.Id() != "prod-models" {
		t.Fatalf("expected ID 'prod-models', got %q", d.Id())
	}
	if d.Get("deployment_count").(int) != 2 {
		t.Fatalf("expected deployment_count 2, got %v", d.Get("deployment_count"))
	}
}

func TestAccessGroupRead(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" || r.URL.Path != "/access_group/prod-models/info" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.Write(accessGroupInfoJSON("prod-models", []string{"gpt-4"}, 1))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := accessGroupTestData(t, map[string]interface{}{"access_group": "prod-models"})
	d.SetId("prod-models")

	if err := resourceLiteLLMAccessGroupRead(d, client); err != nil {
		t.Fatalf("read failed: %v", err)
	}

	if d.Get("access_group").(string) != "prod-models" {
		t.Fatalf("expected access_group 'prod-models', got %v", d.Get("access_group"))
	}
	wantModels := []interface{}{"gpt-4"}
	if !reflect.DeepEqual(d.Get("model_names"), wantModels) {
		t.Fatalf("expected model_names %v, got %v", wantModels, d.Get("model_names"))
	}
	if d.Get("deployment_count").(int) != 1 {
		t.Fatalf("expected deployment_count 1, got %v", d.Get("deployment_count"))
	}
}

func TestAccessGroupReadNotFound(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := accessGroupTestData(t, map[string]interface{}{"access_group": "gone"})
	d.SetId("gone")

	if err := resourceLiteLLMAccessGroupRead(d, client); err != nil {
		t.Fatalf("expected nil error on 404, got: %v", err)
	}
	if d.Id() != "" {
		t.Fatalf("expected ID to be cleared on 404, got %q", d.Id())
	}
}

func TestAccessGroupUpdate(t *testing.T) {
	var updatePayload map[string]interface{}
	var updatePath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case "PUT":
			updatePath = r.URL.Path
			if err := json.NewDecoder(r.Body).Decode(&updatePayload); err != nil {
				t.Errorf("failed to decode update payload: %v", err)
			}
			w.Write([]byte(`{"access_group": "prod-models", "models_updated": 1}`))
		case "GET":
			w.Write(accessGroupInfoJSON("prod-models", []string{"gpt-4o"}, 1))
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := accessGroupTestData(t, map[string]interface{}{
		"access_group": "prod-models",
		"model_names":  []interface{}{"gpt-4o"},
	})
	d.SetId("prod-models")

	if err := resourceLiteLLMAccessGroupUpdate(d, client); err != nil {
		t.Fatalf("update failed: %v", err)
	}

	if updatePath != "/access_group/prod-models/update" {
		t.Fatalf("expected update path '/access_group/prod-models/update', got %q", updatePath)
	}
	wantModels := []interface{}{"gpt-4o"}
	if !reflect.DeepEqual(updatePayload["model_names"], wantModels) {
		t.Fatalf("expected model_names %v in payload, got %v", wantModels, updatePayload["model_names"])
	}
	if _, ok := updatePayload["access_group"]; ok {
		t.Fatalf("update payload must not include access_group, got %v", updatePayload["access_group"])
	}
}

func TestAccessGroupDelete(t *testing.T) {
	var deleteMethod, deletePath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		deleteMethod = r.Method
		deletePath = r.URL.Path
		w.Write([]byte(`{"access_group": "prod-models", "models_updated": 2, "message": "deleted"}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := accessGroupTestData(t, map[string]interface{}{"access_group": "prod-models"})
	d.SetId("prod-models")

	if err := resourceLiteLLMAccessGroupDelete(d, client); err != nil {
		t.Fatalf("delete failed: %v", err)
	}

	if deleteMethod != "DELETE" || deletePath != "/access_group/prod-models/delete" {
		t.Fatalf("expected DELETE /access_group/prod-models/delete, got %s %s", deleteMethod, deletePath)
	}
	if d.Id() != "" {
		t.Fatalf("expected ID to be cleared after delete, got %q", d.Id())
	}
}
