package litellm

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

const testSearchToolParamsJSON = `{"search_provider": "tavily", "api_key": "sk-secret"}`

func newSearchToolTestResourceData(t *testing.T) *schema.ResourceData {
	t.Helper()
	return schema.TestResourceDataRaw(t, resourceLiteLLMSearchTool().Schema, map[string]interface{}{
		"search_tool_name": "my-search",
		"litellm_params":   testSearchToolParamsJSON,
		"search_tool_info": `{"description": "Tavily search"}`,
	})
}

func searchToolReadResponseBody() []byte {
	body, _ := json.Marshal(map[string]interface{}{
		"search_tool_id":   "st-123",
		"search_tool_name": "my-search",
		"litellm_params":   map[string]interface{}{"search_provider": "tavily", "api_key": "sk-s****"},
		"search_tool_info": map[string]interface{}{"description": "Tavily search"},
		"created_at":       "2026-01-01T00:00:00",
		"updated_at":       "2026-01-02T00:00:00",
	})
	return body
}

func TestResourceLiteLLMSearchToolCreate(t *testing.T) {
	var createPayload map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/search_tools":
			if err := json.NewDecoder(r.Body).Decode(&createPayload); err != nil {
				t.Errorf("failed to decode create payload: %v", err)
			}
			w.Write([]byte(`{"search_tool_id": "st-123", "search_tool_name": "my-search"}`))
		case r.Method == http.MethodGet && r.URL.Path == "/search_tools/st-123":
			w.Write(searchToolReadResponseBody())
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newSearchToolTestResourceData(t)

	if err := resourceLiteLLMSearchToolCreate(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if d.Id() != "st-123" {
		t.Fatalf("expected ID 'st-123', got %q", d.Id())
	}

	wrapped, ok := createPayload["search_tool"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected payload wrapped in 'search_tool', got %v", createPayload)
	}
	if wrapped["search_tool_name"] != "my-search" {
		t.Errorf("expected search_tool_name 'my-search', got %v", wrapped["search_tool_name"])
	}
	params, ok := wrapped["litellm_params"].(map[string]interface{})
	if !ok || params["search_provider"] != "tavily" || params["api_key"] != "sk-secret" {
		t.Errorf("expected litellm_params sent as JSON object, got %v", wrapped["litellm_params"])
	}
	info, ok := wrapped["search_tool_info"].(map[string]interface{})
	if !ok || info["description"] != "Tavily search" {
		t.Errorf("expected search_tool_info sent as JSON object, got %v", wrapped["search_tool_info"])
	}

	if got := d.Get("litellm_params").(string); got != testSearchToolParamsJSON {
		t.Errorf("expected litellm_params to keep configured value (masked API value not read back), got %q", got)
	}
	if d.Get("created_at").(string) != "2026-01-01T00:00:00" {
		t.Errorf("expected created_at from read-back, got %q", d.Get("created_at").(string))
	}
}

func TestResourceLiteLLMSearchToolCreateInvalidParamsJSON(t *testing.T) {
	d := schema.TestResourceDataRaw(t, resourceLiteLLMSearchTool().Schema, map[string]interface{}{
		"search_tool_name": "my-search",
		"litellm_params":   "not-json",
	})
	client := NewClient("http://unused.invalid", "test-key", true)

	if err := resourceLiteLLMSearchToolCreate(d, client); err == nil {
		t.Fatal("expected error for invalid litellm_params JSON, got nil")
	}
}

func TestResourceLiteLLMSearchToolReadMapsFields(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/search_tools/st-123" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write(searchToolReadResponseBody())
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, resourceLiteLLMSearchTool().Schema, map[string]interface{}{})
	d.SetId("st-123")

	if err := resourceLiteLLMSearchToolRead(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if d.Get("search_tool_name").(string) != "my-search" {
		t.Errorf("expected search_tool_name 'my-search', got %q", d.Get("search_tool_name").(string))
	}
	var info map[string]interface{}
	if err := json.Unmarshal([]byte(d.Get("search_tool_info").(string)), &info); err != nil {
		t.Fatalf("search_tool_info not populated as JSON: %v", err)
	}
	if info["description"] != "Tavily search" {
		t.Errorf("expected description 'Tavily search', got %v", info["description"])
	}
	if d.Get("litellm_params").(string) != "" {
		t.Errorf("expected litellm_params to never be read back, got %q", d.Get("litellm_params").(string))
	}
	if d.Get("updated_at").(string) != "2026-01-02T00:00:00" {
		t.Errorf("expected updated_at from response, got %q", d.Get("updated_at").(string))
	}
}

func TestResourceLiteLLMSearchToolRead404ClearsID(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newSearchToolTestResourceData(t)
	d.SetId("st-123")

	if err := resourceLiteLLMSearchToolRead(d, client); err != nil {
		t.Fatalf("expected nil error on 404, got: %v", err)
	}
	if d.Id() != "" {
		t.Fatalf("expected ID cleared on 404, got %q", d.Id())
	}
}

func TestResourceLiteLLMSearchToolUpdate(t *testing.T) {
	var updateMethod, updatePath string
	var updatePayload map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method == http.MethodGet {
			w.Write(searchToolReadResponseBody())
			return
		}
		updateMethod = r.Method
		updatePath = r.URL.Path
		if err := json.NewDecoder(r.Body).Decode(&updatePayload); err != nil {
			t.Errorf("failed to decode update payload: %v", err)
		}
		w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newSearchToolTestResourceData(t)
	d.SetId("st-123")

	if err := resourceLiteLLMSearchToolUpdate(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if updateMethod != http.MethodPut {
		t.Errorf("expected PUT, got %s", updateMethod)
	}
	if updatePath != "/search_tools/st-123" {
		t.Errorf("expected path '/search_tools/st-123', got %q", updatePath)
	}
	wrapped, ok := updatePayload["search_tool"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected payload wrapped in 'search_tool', got %v", updatePayload)
	}
	if wrapped["search_tool_id"] != "st-123" {
		t.Errorf("expected search_tool_id in update payload, got %v", wrapped["search_tool_id"])
	}
	if wrapped["search_tool_name"] != "my-search" {
		t.Errorf("expected search_tool_name in update payload, got %v", wrapped["search_tool_name"])
	}
}

func TestResourceLiteLLMSearchToolDelete(t *testing.T) {
	var deleteMethod, deletePath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		deleteMethod = r.Method
		deletePath = r.URL.Path
		w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newSearchToolTestResourceData(t)
	d.SetId("st-123")

	if err := resourceLiteLLMSearchToolDelete(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if deleteMethod != http.MethodDelete {
		t.Errorf("expected DELETE, got %s", deleteMethod)
	}
	if deletePath != "/search_tools/st-123" {
		t.Errorf("expected path '/search_tools/st-123', got %q", deletePath)
	}
	if d.Id() != "" {
		t.Fatalf("expected ID cleared after delete, got %q", d.Id())
	}
}
