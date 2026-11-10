package litellm

import (
	"encoding/json"
	"fmt"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func dataSourceLiteLLMPrompt() *schema.Resource {
	return &schema.Resource{
		Read: dataSourceLiteLLMPromptRead,

		Schema: map[string]*schema.Schema{
			"prompt_id": {
				Type:        schema.TypeString,
				Required:    true,
				Description: "Unique identifier of the prompt to retrieve",
			},
			"environment": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "Environment to fetch the prompt from (e.g. 'development', 'production')",
			},
			"prompt_integration": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"api_base": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"provider_specific_query_params": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"ignore_prompt_manager_model": {
				Type:     schema.TypeBool,
				Computed: true,
			},
			"ignore_prompt_manager_optional_params": {
				Type:     schema.TypeBool,
				Computed: true,
			},
			"dotprompt_content": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"prompt_type": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"version": {
				Type:     schema.TypeInt,
				Computed: true,
			},
			"environments": {
				Type:     schema.TypeList,
				Computed: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
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

func dataSourceLiteLLMPromptRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)
	promptID := d.Get("prompt_id").(string)

	endpoint := fmt.Sprintf(endpointPromptInfo, promptID)
	if env := d.Get("environment").(string); env != "" {
		endpoint = fmt.Sprintf("/prompts/%s/info?environment=%s", promptID, env)
	}

	resp, err := MakeRequest(client, "GET", endpoint, nil)
	if err != nil {
		return fmt.Errorf("failed to read prompt: %w", err)
	}
	defer resp.Body.Close()

	if promptIsNotFoundResponse(resp) {
		return fmt.Errorf("prompt '%s' not found", promptID)
	}

	if err := handleResponse(resp, "reading prompt"); err != nil {
		return err
	}

	var info struct {
		PromptSpec   promptSpecAPIResponse `json:"prompt_spec"`
		Environments []string              `json:"environments"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return fmt.Errorf("error decoding prompt info response: %w", err)
	}

	d.SetId(info.PromptSpec.PromptID)
	d.Set("prompt_id", info.PromptSpec.PromptID)
	d.Set("version", info.PromptSpec.Version)
	d.Set("environments", info.Environments)
	d.Set("created_at", info.PromptSpec.CreatedAt)
	d.Set("updated_at", info.PromptSpec.UpdatedAt)

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
	// api_key is intentionally not exposed.

	return nil
}

func dataSourceLiteLLMPrompts() *schema.Resource {
	return &schema.Resource{
		Read: dataSourceLiteLLMPromptsRead,

		Schema: map[string]*schema.Schema{
			"environment": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "Filter prompts by environment (e.g. 'development', 'production')",
			},
			"prompts": {
				Type:     schema.TypeList,
				Computed: true,
				Elem: &schema.Resource{
					Schema: map[string]*schema.Schema{
						"prompt_id": {
							Type:     schema.TypeString,
							Computed: true,
						},
						"prompt_integration": {
							Type:     schema.TypeString,
							Computed: true,
						},
						"prompt_type": {
							Type:     schema.TypeString,
							Computed: true,
						},
						"version": {
							Type:     schema.TypeInt,
							Computed: true,
						},
						"environment": {
							Type:     schema.TypeString,
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
			"ids": {
				Type:     schema.TypeList,
				Computed: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
		},
	}
}

func dataSourceLiteLLMPromptsRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	endpoint := endpointPromptList
	if env := d.Get("environment").(string); env != "" {
		endpoint = fmt.Sprintf("/prompts/list?environment=%s", env)
	}

	resp, err := MakeRequest(client, "GET", endpoint, nil)
	if err != nil {
		return fmt.Errorf("failed to list prompts: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "listing prompts"); err != nil {
		return err
	}

	var listResp struct {
		Prompts []promptSpecAPIResponse `json:"prompts"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&listResp); err != nil {
		return fmt.Errorf("error decoding prompts list response: %w", err)
	}

	prompts := make([]map[string]interface{}, 0, len(listResp.Prompts))
	ids := make([]string, 0, len(listResp.Prompts))
	for _, p := range listResp.Prompts {
		integration, _ := p.LitellmParams["prompt_integration"].(string)
		promptType, _ := p.PromptInfo["prompt_type"].(string)
		prompts = append(prompts, map[string]interface{}{
			"prompt_id":          p.PromptID,
			"prompt_integration": integration,
			"prompt_type":        promptType,
			"version":            p.Version,
			"environment":        p.Environment,
			"created_at":         p.CreatedAt,
			"updated_at":         p.UpdatedAt,
		})
		ids = append(ids, p.PromptID)
	}

	d.SetId("prompts")
	d.Set("prompts", prompts)
	d.Set("ids", ids)

	return nil
}
