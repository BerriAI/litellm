package litellm

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"reflect"
	"strings"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

const (
	endpointPromptCreate = "/prompts"
	endpointPromptByID   = "/prompts/%s"
	endpointPromptInfo   = "/prompts/%s/info"
	endpointPromptList   = "/prompts/list"
)

func resourceLiteLLMPrompt() *schema.Resource {
	return &schema.Resource{
		Create: resourceLiteLLMPromptCreate,
		Read:   resourceLiteLLMPromptRead,
		Update: resourceLiteLLMPromptUpdate,
		Delete: resourceLiteLLMPromptDelete,

		Importer: &schema.ResourceImporter{StateContext: schema.ImportStatePassthroughContext},

		Schema: map[string]*schema.Schema{
			"prompt_id": {
				Type:        schema.TypeString,
				Required:    true,
				ForceNew:    true,
				Description: "Unique identifier for the prompt",
			},
			"prompt_integration": {
				Type:        schema.TypeString,
				Required:    true,
				Description: "The prompt integration provider (e.g. 'langfuse', 'dotprompt')",
			},
			"api_base": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "Base URL for the prompt provider API",
			},
			"api_key": {
				Type:        schema.TypeString,
				Optional:    true,
				Sensitive:   true,
				Description: "API key for the prompt provider",
			},
			"provider_specific_query_params": {
				Type:             schema.TypeString,
				Optional:         true,
				DiffSuppressFunc: promptSuppressJSONDiff,
				Description:      "JSON string of provider-specific query parameters",
			},
			"ignore_prompt_manager_model": {
				Type:        schema.TypeBool,
				Optional:    true,
				Description: "If true, ignore the model specified in the prompt manager",
			},
			"ignore_prompt_manager_optional_params": {
				Type:        schema.TypeBool,
				Optional:    true,
				Description: "If true, ignore optional params from the prompt manager",
			},
			"dotprompt_content": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "Content for dotprompt integration",
			},
			"litellm_params": {
				Type:             schema.TypeString,
				Optional:         true,
				Sensitive:        true,
				DiffSuppressFunc: promptSuppressJSONDiff,
				Description: "JSON string with additional litellm_params merged into the request " +
					"(e.g. the integration's own prompt_id, prompt_directory, prompt_data; may contain secrets)",
			},
			"prompt_type": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "Type of prompt: 'config' or 'db'",
			},
		},
	}
}

func promptSuppressJSONDiff(k, oldValue, newValue string, d *schema.ResourceData) bool {
	var oldParsed, newParsed interface{}
	if json.Unmarshal([]byte(oldValue), &oldParsed) != nil || json.Unmarshal([]byte(newValue), &newParsed) != nil {
		return false
	}
	return reflect.DeepEqual(oldParsed, newParsed)
}

func buildPromptData(d *schema.ResourceData) (map[string]interface{}, error) {
	litellmParams := map[string]interface{}{
		"prompt_integration": d.Get("prompt_integration").(string),
	}

	for tfKey, apiKey := range map[string]string{
		"api_base":          "api_base",
		"api_key":           "api_key",
		"dotprompt_content": "dotprompt_content",
	} {
		if v := d.Get(tfKey).(string); v != "" {
			litellmParams[apiKey] = v
		}
	}

	if v := d.Get("provider_specific_query_params").(string); v != "" {
		var params map[string]interface{}
		if err := json.Unmarshal([]byte(v), &params); err != nil {
			return nil, fmt.Errorf("provider_specific_query_params is not valid JSON: %w", err)
		}
		litellmParams["provider_specific_query_params"] = params
	}

	litellmParams["ignore_prompt_manager_model"] = d.Get("ignore_prompt_manager_model").(bool)
	litellmParams["ignore_prompt_manager_optional_params"] = d.Get("ignore_prompt_manager_optional_params").(bool)

	if raw := d.Get("litellm_params").(string); raw != "" {
		var extra map[string]interface{}
		if err := json.Unmarshal([]byte(raw), &extra); err != nil {
			return nil, fmt.Errorf("litellm_params is not valid JSON: %w", err)
		}
		for k, v := range extra {
			litellmParams[k] = v
		}
	}

	promptData := map[string]interface{}{
		"prompt_id":      d.Get("prompt_id").(string),
		"litellm_params": litellmParams,
	}

	if v := d.Get("prompt_type").(string); v != "" {
		promptData["prompt_info"] = map[string]interface{}{"prompt_type": v}
	}

	return promptData, nil
}

type promptSpecAPIResponse struct {
	PromptID      string                 `json:"prompt_id"`
	LitellmParams map[string]interface{} `json:"litellm_params"`
	PromptInfo    map[string]interface{} `json:"prompt_info"`
	Version       int                    `json:"version"`
	Environment   string                 `json:"environment"`
	CreatedAt     string                 `json:"created_at"`
	UpdatedAt     string                 `json:"updated_at"`
}

func resourceLiteLLMPromptCreate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	promptData, err := buildPromptData(d)
	if err != nil {
		return err
	}

	promptID := d.Get("prompt_id").(string)
	log.Printf("[DEBUG] Create prompt request for: %s", promptID)

	resp, err := MakeRequest(client, "POST", endpointPromptCreate, promptData)
	if err != nil {
		return fmt.Errorf("error creating prompt: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "creating prompt"); err != nil {
		return err
	}

	d.SetId(promptID)
	log.Printf("[INFO] Prompt created with ID: %s", promptID)

	return resourceLiteLLMPromptRead(d, m)
}

func promptIsNotFoundResponse(resp *http.Response) bool {
	if resp.StatusCode == http.StatusNotFound {
		return true
	}
	if resp.StatusCode != http.StatusBadRequest {
		return false
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return false
	}
	resp.Body = io.NopCloser(strings.NewReader(string(body)))
	return strings.Contains(string(body), "not found")
}

func resourceLiteLLMPromptRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	log.Printf("[INFO] Reading prompt with ID: %s", d.Id())

	resp, err := MakeRequest(client, "GET", fmt.Sprintf(endpointPromptInfo, d.Id()), nil)
	if err != nil {
		return fmt.Errorf("error reading prompt: %w", err)
	}
	defer resp.Body.Close()

	if promptIsNotFoundResponse(resp) {
		log.Printf("[WARN] Prompt with ID %s not found, removing from state", d.Id())
		d.SetId("")
		return nil
	}

	if err := handleResponse(resp, "reading prompt"); err != nil {
		return err
	}

	var info struct {
		PromptSpec promptSpecAPIResponse `json:"prompt_spec"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return fmt.Errorf("error decoding prompt info response: %w", err)
	}

	d.Set("prompt_id", info.PromptSpec.PromptID)

	params := info.PromptSpec.LitellmParams
	if v, ok := params["prompt_integration"].(string); ok {
		d.Set("prompt_integration", v)
	}
	if v, ok := params["api_base"].(string); ok {
		d.Set("api_base", v)
	}
	if v, ok := params["dotprompt_content"].(string); ok {
		d.Set("dotprompt_content", v)
	}
	if v, ok := params["ignore_prompt_manager_model"].(bool); ok {
		d.Set("ignore_prompt_manager_model", v)
	}
	if v, ok := params["ignore_prompt_manager_optional_params"].(bool); ok {
		d.Set("ignore_prompt_manager_optional_params", v)
	}
	if v, ok := params["provider_specific_query_params"].(map[string]interface{}); ok {
		if encoded, err := json.Marshal(v); err == nil {
			d.Set("provider_specific_query_params", string(encoded))
		}
	}
	if v, ok := info.PromptSpec.PromptInfo["prompt_type"].(string); ok {
		d.Set("prompt_type", v)
	}
	// api_key and the litellm_params catch-all are intentionally not read back:
	// they can carry secrets, so state keeps the configured values authoritative.

	return nil
}

func resourceLiteLLMPromptUpdate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	promptData, err := buildPromptData(d)
	if err != nil {
		return err
	}

	log.Printf("[DEBUG] Update prompt request for ID: %s", d.Id())

	resp, err := MakeRequest(client, "PUT", fmt.Sprintf(endpointPromptByID, d.Id()), promptData)
	if err != nil {
		return fmt.Errorf("error updating prompt: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "updating prompt"); err != nil {
		return err
	}

	log.Printf("[INFO] Successfully updated prompt with ID: %s", d.Id())
	return resourceLiteLLMPromptRead(d, m)
}

func resourceLiteLLMPromptDelete(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	log.Printf("[INFO] Deleting prompt with ID: %s", d.Id())

	resp, err := MakeRequest(client, "DELETE", fmt.Sprintf(endpointPromptByID, d.Id()), nil)
	if err != nil {
		return fmt.Errorf("error deleting prompt: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNotFound {
		if err := handleResponse(resp, "deleting prompt"); err != nil {
			return err
		}
	}

	log.Printf("[INFO] Successfully deleted prompt with ID: %s", d.Id())
	d.SetId("")
	return nil
}
