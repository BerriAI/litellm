package litellm

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/url"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/validation"
)

const endpointFallbackCreate = "/fallback"

type FallbackGetResponse struct {
	Model          string   `json:"model"`
	FallbackModels []string `json:"fallback_models"`
	FallbackType   string   `json:"fallback_type"`
}

func resourceLiteLLMFallback() *schema.Resource {
	return &schema.Resource{
		Create: resourceLiteLLMFallbackCreate,
		Read:   resourceLiteLLMFallbackRead,
		Update: resourceLiteLLMFallbackUpdate,
		Delete: resourceLiteLLMFallbackDelete,

		Importer: &schema.ResourceImporter{
			StateContext: schema.ImportStatePassthroughContext,
		},

		Schema: map[string]*schema.Schema{
			"model": {
				Type:        schema.TypeString,
				Required:    true,
				ForceNew:    true,
				Description: "The model name to configure fallbacks for",
			},
			"fallback_models": {
				Type:        schema.TypeList,
				Required:    true,
				MinItems:    1,
				Elem:        &schema.Schema{Type: schema.TypeString},
				Description: "List of fallback model names in order of priority",
			},
			"fallback_type": {
				Type:         schema.TypeString,
				Optional:     true,
				ForceNew:     true,
				Default:      "general",
				ValidateFunc: validation.StringInSlice([]string{"general", "context_window", "content_policy"}, false),
				Description:  "Type of fallback: 'general' (default), 'context_window', or 'content_policy'",
			},
		},
	}
}

func fallbackTypeFromState(d *schema.ResourceData) string {
	return GetStringValue(d.Get("fallback_type").(string), "general")
}

func buildFallbackData(d *schema.ResourceData) map[string]interface{} {
	return map[string]interface{}{
		"model":           d.Get("model").(string),
		"fallback_models": d.Get("fallback_models"),
		"fallback_type":   fallbackTypeFromState(d),
	}
}

func upsertLiteLLMFallback(d *schema.ResourceData, m interface{}, action string) error {
	client := m.(*Client)

	fallbackData := buildFallbackData(d)
	log.Printf("[DEBUG] %s fallback request payload: %+v", action, fallbackData)

	resp, err := MakeRequest(client, "POST", endpointFallbackCreate, fallbackData)
	if err != nil {
		return fmt.Errorf("error %s fallback: %w", action, err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, action+" fallback"); err != nil {
		return err
	}

	d.SetId(d.Get("model").(string))
	return resourceLiteLLMFallbackRead(d, m)
}

func resourceLiteLLMFallbackCreate(d *schema.ResourceData, m interface{}) error {
	return upsertLiteLLMFallback(d, m, "creating")
}

func resourceLiteLLMFallbackRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	log.Printf("[INFO] Reading fallback for model: %s", d.Id())

	endpoint := fmt.Sprintf("/fallback/%s?fallback_type=%s",
		url.PathEscape(d.Id()), url.QueryEscape(fallbackTypeFromState(d)))
	resp, err := MakeRequest(client, "GET", endpoint, nil)
	if err != nil {
		return fmt.Errorf("error reading fallback: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		log.Printf("[WARN] Fallback for model %s not found, removing from state", d.Id())
		d.SetId("")
		return nil
	}

	if err := handleResponse(resp, "reading fallback"); err != nil {
		return err
	}

	var fallbackResp FallbackGetResponse
	if err := json.NewDecoder(resp.Body).Decode(&fallbackResp); err != nil {
		return fmt.Errorf("error decoding fallback response: %w", err)
	}

	d.Set("model", GetStringValue(fallbackResp.Model, d.Id()))
	d.Set("fallback_models", fallbackResp.FallbackModels)
	d.Set("fallback_type", GetStringValue(fallbackResp.FallbackType, fallbackTypeFromState(d)))

	return nil
}

func resourceLiteLLMFallbackUpdate(d *schema.ResourceData, m interface{}) error {
	return upsertLiteLLMFallback(d, m, "updating")
}

func resourceLiteLLMFallbackDelete(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	log.Printf("[INFO] Deleting fallback for model: %s", d.Id())

	endpoint := fmt.Sprintf("/fallback/%s?fallback_type=%s",
		url.PathEscape(d.Id()), url.QueryEscape(fallbackTypeFromState(d)))
	resp, err := MakeRequest(client, "DELETE", endpoint, nil)
	if err != nil {
		return fmt.Errorf("error deleting fallback: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNotFound {
		if err := handleResponse(resp, "deleting fallback"); err != nil {
			return err
		}
	}

	d.SetId("")
	return nil
}
