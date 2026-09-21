package litellm

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func dataSourceLiteLLMSearchTool() *schema.Resource {
	return &schema.Resource{
		Read: dataSourceLiteLLMSearchToolRead,

		Schema: map[string]*schema.Schema{
			"search_tool_id": {
				Type:        schema.TypeString,
				Required:    true,
				Description: "Unique identifier of the search tool to retrieve.",
			},
			"search_tool_name": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"search_tool_info": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Additional metadata as a JSON object string.",
			},
			"created_at": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"updated_at": {
				Type:     schema.TypeString,
				Computed: true,
			},
		},
	}
}

func dataSourceLiteLLMSearchToolRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)
	searchToolID := d.Get("search_tool_id").(string)

	resp, err := MakeRequest(client, "GET", fmt.Sprintf(endpointSearchToolByID, searchToolID), nil)
	if err != nil {
		return fmt.Errorf("error reading search tool: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return fmt.Errorf("search tool '%s' not found", searchToolID)
	}

	if err := handleResponse(resp, "reading search tool"); err != nil {
		return err
	}

	var searchToolResp searchToolAPIResponse
	if err := json.NewDecoder(resp.Body).Decode(&searchToolResp); err != nil {
		return fmt.Errorf("error decoding search tool info response: %w", err)
	}

	// litellm_params is intentionally never exposed: it may hold provider API keys.
	d.SetId(searchToolResp.SearchToolID)
	d.Set("search_tool_name", searchToolResp.SearchToolName)
	if searchToolResp.SearchToolInfo != nil {
		infoJSON, err := json.Marshal(searchToolResp.SearchToolInfo)
		if err != nil {
			return fmt.Errorf("error encoding search_tool_info: %w", err)
		}
		d.Set("search_tool_info", string(infoJSON))
	}
	d.Set("created_at", searchToolResp.CreatedAt)
	d.Set("updated_at", searchToolResp.UpdatedAt)

	return nil
}

func dataSourceLiteLLMSearchTools() *schema.Resource {
	return &schema.Resource{
		Read: dataSourceLiteLLMSearchToolsRead,

		Schema: map[string]*schema.Schema{
			"ids": {
				Type:     schema.TypeList,
				Computed: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
			"search_tools": {
				Type:     schema.TypeList,
				Computed: true,
				Elem: &schema.Resource{
					Schema: map[string]*schema.Schema{
						"search_tool_id": {
							Type:     schema.TypeString,
							Computed: true,
						},
						"search_tool_name": {
							Type:     schema.TypeString,
							Computed: true,
						},
						"search_tool_info": {
							Type:        schema.TypeString,
							Computed:    true,
							Description: "Additional metadata as a JSON object string.",
						},
						"is_from_config": {
							Type:     schema.TypeBool,
							Computed: true,
						},
						"created_at": {
							Type:     schema.TypeString,
							Computed: true,
						},
						"updated_at": {
							Type:     schema.TypeString,
							Computed: true,
						},
					},
				},
			},
		},
	}
}

func dataSourceLiteLLMSearchToolsRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	resp, err := MakeRequest(client, "GET", endpointSearchToolsList, nil)
	if err != nil {
		return fmt.Errorf("error listing search tools: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "listing search tools"); err != nil {
		return err
	}

	var listResp struct {
		SearchTools []searchToolAPIResponse `json:"search_tools"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&listResp); err != nil {
		return fmt.Errorf("error decoding search tools list response: %w", err)
	}

	ids := make([]string, 0, len(listResp.SearchTools))
	searchTools := make([]map[string]interface{}, 0, len(listResp.SearchTools))
	for _, searchToolResp := range listResp.SearchTools {
		ids = append(ids, searchToolResp.SearchToolID)

		searchTool := map[string]interface{}{
			"search_tool_id":   searchToolResp.SearchToolID,
			"search_tool_name": searchToolResp.SearchToolName,
			"created_at":       searchToolResp.CreatedAt,
			"updated_at":       searchToolResp.UpdatedAt,
		}
		if searchToolResp.SearchToolInfo != nil {
			infoJSON, err := json.Marshal(searchToolResp.SearchToolInfo)
			if err != nil {
				return fmt.Errorf("error encoding search_tool_info: %w", err)
			}
			searchTool["search_tool_info"] = string(infoJSON)
		}
		if searchToolResp.IsFromConfig != nil {
			searchTool["is_from_config"] = *searchToolResp.IsFromConfig
		}
		searchTools = append(searchTools, searchTool)
	}

	d.SetId(strconv.FormatInt(time.Now().UnixNano(), 10))
	d.Set("ids", ids)
	d.Set("search_tools", searchTools)

	return nil
}
