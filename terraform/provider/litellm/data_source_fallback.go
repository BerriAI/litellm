package litellm

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/validation"
)

func dataSourceLiteLLMFallback() *schema.Resource {
	return &schema.Resource{
		Read: dataSourceLiteLLMFallbackRead,

		Schema: map[string]*schema.Schema{
			"model": {
				Type:        schema.TypeString,
				Required:    true,
				Description: "The model name to get fallbacks for",
			},
			"fallback_type": {
				Type:         schema.TypeString,
				Optional:     true,
				Default:      "general",
				ValidateFunc: validation.StringInSlice([]string{"general", "context_window", "content_policy"}, false),
				Description:  "Type of fallback: 'general' (default), 'context_window', or 'content_policy'",
			},
			"fallback_models": {
				Type:        schema.TypeList,
				Computed:    true,
				Elem:        &schema.Schema{Type: schema.TypeString},
				Description: "List of fallback model names in order of priority",
			},
		},
	}
}

func dataSourceLiteLLMFallbackRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)
	model := d.Get("model").(string)
	fallbackType := GetStringValue(d.Get("fallback_type").(string), "general")

	endpoint := fmt.Sprintf("/fallback/%s?fallback_type=%s", url.PathEscape(model), url.QueryEscape(fallbackType))
	resp, err := MakeRequest(client, "GET", endpoint, nil)
	if err != nil {
		return fmt.Errorf("failed to read fallback: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return fmt.Errorf("no %s fallbacks configured for model '%s'", fallbackType, model)
	}

	if err := handleResponse(resp, "reading fallback"); err != nil {
		return err
	}

	var fallbackResp FallbackGetResponse
	if err := json.NewDecoder(resp.Body).Decode(&fallbackResp); err != nil {
		return fmt.Errorf("error decoding fallback response: %w", err)
	}

	d.SetId(model)
	d.Set("model", GetStringValue(fallbackResp.Model, model))
	d.Set("fallback_models", fallbackResp.FallbackModels)
	d.Set("fallback_type", GetStringValue(fallbackResp.FallbackType, fallbackType))

	return nil
}
