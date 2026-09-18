package litellm

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func TestDataSourceLiteLLMSearchToolRead(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/search_tools/st-123" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write(searchToolReadResponseBody())
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMSearchTool().Schema, map[string]interface{}{
		"search_tool_id": "st-123",
	})

	if err := dataSourceLiteLLMSearchToolRead(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if d.Id() != "st-123" {
		t.Fatalf("expected ID 'st-123', got %q", d.Id())
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
}

func TestDataSourceLiteLLMSearchToolsRead(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/search_tools/list" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		body, _ := json.Marshal(map[string]interface{}{
			"search_tools": []map[string]interface{}{
				{
					"search_tool_id":   "st-1",
					"search_tool_name": "first",
					"search_tool_info": map[string]interface{}{"description": "first tool"},
					"is_from_config":   true,
				},
				{"search_tool_id": "st-2", "search_tool_name": "second"},
			},
		})
		w.Write(body)
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, dataSourceLiteLLMSearchTools().Schema, map[string]interface{}{})

	if err := dataSourceLiteLLMSearchToolsRead(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}

	ids := d.Get("ids").([]interface{})
	if len(ids) != 2 || ids[0] != "st-1" || ids[1] != "st-2" {
		t.Fatalf("expected ids [st-1 st-2], got %v", ids)
	}
	searchTools := d.Get("search_tools").([]interface{})
	if len(searchTools) != 2 {
		t.Fatalf("expected 2 search tools, got %d", len(searchTools))
	}
	first := searchTools[0].(map[string]interface{})
	if first["search_tool_name"] != "first" || first["is_from_config"] != true {
		t.Errorf("unexpected first search tool entry: %v", first)
	}
	var info map[string]interface{}
	if err := json.Unmarshal([]byte(first["search_tool_info"].(string)), &info); err != nil {
		t.Fatalf("search_tool_info not JSON-encoded in list: %v", err)
	}
	if info["description"] != "first tool" {
		t.Errorf("expected description 'first tool', got %v", info["description"])
	}
	if d.Id() == "" {
		t.Fatal("expected data source ID to be set")
	}
}
