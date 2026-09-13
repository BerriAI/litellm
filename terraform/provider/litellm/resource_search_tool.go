package litellm

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"reflect"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

const (
	endpointSearchTools     = "/search_tools"
	endpointSearchToolByID  = "/search_tools/%s"
	endpointSearchToolsList = "/search_tools/list"
)

type searchToolAPIResponse struct {
	SearchToolID   string                 `json:"search_tool_id"`
	SearchToolName string                 `json:"search_tool_name"`
	SearchToolInfo map[string]interface{} `json:"search_tool_info"`
	IsFromConfig   *bool                  `json:"is_from_config"`
	CreatedAt      string                 `json:"created_at"`
	UpdatedAt      string                 `json:"updated_at"`
}

func searchToolSuppressEquivalentJSON(k, oldValue, newValue string, d *schema.ResourceData) bool {
	var oldObj, newObj interface{}
	if err := json.Unmarshal([]byte(oldValue), &oldObj); err != nil {
		return false
	}
	if err := json.Unmarshal([]byte(newValue), &newObj); err != nil {
		return false
	}
	return reflect.DeepEqual(oldObj, newObj)
}

func searchToolParseJSONObject(raw, field string) (map[string]interface{}, error) {
	var obj map[string]interface{}
	if err := json.Unmarshal([]byte(raw), &obj); err != nil {
		return nil, fmt.Errorf("%s must be a JSON object: %w", field, err)
	}
	return obj, nil
}

func resourceLiteLLMSearchTool() *schema.Resource {
	return &schema.Resource{
		Create: resourceLiteLLMSearchToolCreate,
		Read:   resourceLiteLLMSearchToolRead,
		Update: resourceLiteLLMSearchToolUpdate,
		Delete: resourceLiteLLMSearchToolDelete,

		Importer: &schema.ResourceImporter{StateContext: schema.ImportStatePassthroughContext},

		Schema: map[string]*schema.Schema{
			"search_tool_name": {
				Type:        schema.TypeString,
				Required:    true,
				Description: "Name of the search tool.",
			},
			"litellm_params": {
				Type:             schema.TypeString,
				Required:         true,
				Sensitive:        true,
				DiffSuppressFunc: searchToolSuppressEquivalentJSON,
				Description: "Search tool parameters as a JSON object string (search_provider, api_key, " +
					"api_base, timeout, max_retries, ...). The API only returns masked values, so this is " +
					"never read back.",
			},
			"search_tool_info": {
				Type:             schema.TypeString,
				Optional:         true,
				DiffSuppressFunc: searchToolSuppressEquivalentJSON,
				Description:      "Additional metadata as a JSON object string (e.g. description).",
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

func buildSearchToolData(d *schema.ResourceData) (map[string]interface{}, error) {
	litellmParams, err := searchToolParseJSONObject(d.Get("litellm_params").(string), "litellm_params")
	if err != nil {
		return nil, err
	}

	searchToolData := map[string]interface{}{
		"search_tool_name": d.Get("search_tool_name").(string),
		"litellm_params":   litellmParams,
	}

	if raw, ok := d.GetOk("search_tool_info"); ok && raw.(string) != "" {
		info, err := searchToolParseJSONObject(raw.(string), "search_tool_info")
		if err != nil {
			return nil, err
		}
		searchToolData["search_tool_info"] = info
	}

	return searchToolData, nil
}

func resourceLiteLLMSearchToolCreate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	searchToolData, err := buildSearchToolData(d)
	if err != nil {
		return err
	}

	log.Printf("[DEBUG] Create search tool request for: %s", d.Get("search_tool_name").(string))

	resp, err := MakeRequest(client, "POST", endpointSearchTools, map[string]interface{}{
		"search_tool": searchToolData,
	})
	if err != nil {
		return fmt.Errorf("error creating search tool: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "creating search tool"); err != nil {
		return err
	}

	var searchToolResp searchToolAPIResponse
	if err := json.NewDecoder(resp.Body).Decode(&searchToolResp); err != nil {
		return fmt.Errorf("error decoding create search tool response: %w", err)
	}
	if searchToolResp.SearchToolID == "" {
		return fmt.Errorf("create search tool response did not contain a search_tool_id")
	}

	d.SetId(searchToolResp.SearchToolID)
	log.Printf("[INFO] Search tool created with ID: %s", searchToolResp.SearchToolID)

	return resourceLiteLLMSearchToolRead(d, m)
}

func resourceLiteLLMSearchToolRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	log.Printf("[INFO] Reading search tool with ID: %s", d.Id())

	resp, err := MakeRequest(client, "GET", fmt.Sprintf(endpointSearchToolByID, d.Id()), nil)
	if err != nil {
		return fmt.Errorf("error reading search tool: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		log.Printf("[WARN] Search tool with ID %s not found, removing from state", d.Id())
		d.SetId("")
		return nil
	}

	if err := handleResponse(resp, "reading search tool"); err != nil {
		return err
	}

	var searchToolResp searchToolAPIResponse
	if err := json.NewDecoder(resp.Body).Decode(&searchToolResp); err != nil {
		return fmt.Errorf("error decoding search tool info response: %w", err)
	}

	d.Set("search_tool_name", searchToolResp.SearchToolName)

	// litellm_params is intentionally not read back: the API masks its values and it may hold secrets.
	if searchToolResp.SearchToolInfo != nil {
		infoJSON, err := json.Marshal(searchToolResp.SearchToolInfo)
		if err != nil {
			return fmt.Errorf("error encoding search_tool_info: %w", err)
		}
		d.Set("search_tool_info", string(infoJSON))
	}
	d.Set("created_at", searchToolResp.CreatedAt)
	d.Set("updated_at", searchToolResp.UpdatedAt)

	log.Printf("[INFO] Successfully read search tool with ID: %s", d.Id())
	return nil
}

func resourceLiteLLMSearchToolUpdate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	searchToolData, err := buildSearchToolData(d)
	if err != nil {
		return err
	}
	searchToolData["search_tool_id"] = d.Id()

	log.Printf("[DEBUG] Update search tool request for ID: %s", d.Id())

	resp, err := MakeRequest(client, "PUT", fmt.Sprintf(endpointSearchToolByID, d.Id()), map[string]interface{}{
		"search_tool": searchToolData,
	})
	if err != nil {
		return fmt.Errorf("error updating search tool: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "updating search tool"); err != nil {
		return err
	}

	log.Printf("[INFO] Successfully updated search tool with ID: %s", d.Id())
	return resourceLiteLLMSearchToolRead(d, m)
}

func resourceLiteLLMSearchToolDelete(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	log.Printf("[INFO] Deleting search tool with ID: %s", d.Id())

	resp, err := MakeRequest(client, "DELETE", fmt.Sprintf(endpointSearchToolByID, d.Id()), nil)
	if err != nil {
		return fmt.Errorf("error deleting search tool: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNotFound {
		if err := handleResponse(resp, "deleting search tool"); err != nil {
			return err
		}
	}

	log.Printf("[INFO] Successfully deleted search tool with ID: %s", d.Id())
	d.SetId("")
	return nil
}
