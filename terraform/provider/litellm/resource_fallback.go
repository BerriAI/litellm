package litellm

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"slices"
	"time"

	"github.com/hashicorp/terraform-plugin-sdk/v2/diag"
	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/validation"
)

const (
	endpointFallbackCreate     = "/fallback"
	fallbackReadBackTimeout    = 2 * time.Minute
	fallbackReadBackFirstDelay = time.Second
	fallbackReadBackMaxDelay   = 10 * time.Second
)

type waitFunc func(ctx context.Context, delay time.Duration) error

func waitOrCancel(ctx context.Context, delay time.Duration) error {
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

type FallbackGetResponse struct {
	Model          string   `json:"model"`
	FallbackModels []string `json:"fallback_models"`
	FallbackType   string   `json:"fallback_type"`
}

func resourceLiteLLMFallback() *schema.Resource {
	return &schema.Resource{
		CreateContext: resourceLiteLLMFallbackCreate,
		Read:          resourceLiteLLMFallbackRead,
		UpdateContext: resourceLiteLLMFallbackUpdate,
		Delete:        resourceLiteLLMFallbackDelete,

		Timeouts: &schema.ResourceTimeout{
			Create: schema.DefaultTimeout(fallbackReadBackTimeout),
			Update: schema.DefaultTimeout(fallbackReadBackTimeout),
		},

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

func upsertLiteLLMFallback(ctx context.Context, d *schema.ResourceData, m interface{}, action string, wait waitFunc) error {
	client := m.(*Client)

	fallbackData := buildFallbackData(d)
	log.Printf("[DEBUG] %s fallback request payload: %+v", action, fallbackData)

	resp, err := MakeRequestWithContext(ctx, client, "POST", endpointFallbackCreate, fallbackData)
	if err != nil {
		return fmt.Errorf("error %s fallback: %w", action, err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, action+" fallback"); err != nil {
		return err
	}

	d.SetId(d.Get("model").(string))
	return readBackWrittenFallback(ctx, d, client, wait)
}

// A replica that has not reloaded the write yet answers 404 or the previous list, so poll until it serves the write.
func readBackWrittenFallback(ctx context.Context, d *schema.ResourceData, client *Client, wait waitFunc) error {
	written := stringListFromState(d.Get("fallback_models"))
	delay := fallbackReadBackFirstDelay
	for attempt := 1; ; attempt++ {
		served, err := fetchFallback(ctx, client, d.Id(), fallbackTypeFromState(d))
		if err != nil {
			return err
		}
		if served != nil && slices.Equal(served.FallbackModels, written) {
			setFallbackState(d, served)
			return nil
		}
		log.Printf("[INFO] Fallback for model %s not served as written yet (read %d), retrying in %v", d.Id(), attempt, delay)
		if err := wait(ctx, delay); err != nil {
			return fmt.Errorf("fallback for model %s was saved, but the proxy still did not serve %v after %d reads: %w; re-running apply should resolve this", d.Id(), written, attempt, err)
		}
		delay = min(delay*2, fallbackReadBackMaxDelay)
	}
}

func stringListFromState(raw interface{}) []string {
	items := raw.([]interface{})
	values := make([]string, 0, len(items))
	for _, item := range items {
		values = append(values, item.(string))
	}
	return values
}

func resourceLiteLLMFallbackCreate(ctx context.Context, d *schema.ResourceData, m interface{}) diag.Diagnostics {
	return diag.FromErr(upsertLiteLLMFallback(ctx, d, m, "creating", waitOrCancel))
}

func fetchFallback(ctx context.Context, client *Client, model, fallbackType string) (*FallbackGetResponse, error) {
	endpoint := fmt.Sprintf("/fallback/%s?fallback_type=%s", url.PathEscape(model), url.QueryEscape(fallbackType))
	resp, err := MakeRequestWithContext(ctx, client, "GET", endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("error reading fallback: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return nil, nil
	}

	if err := handleResponse(resp, "reading fallback"); err != nil {
		return nil, err
	}

	var fallbackResp FallbackGetResponse
	if err := json.NewDecoder(resp.Body).Decode(&fallbackResp); err != nil {
		return nil, fmt.Errorf("error decoding fallback response: %w", err)
	}
	return &fallbackResp, nil
}

func setFallbackState(d *schema.ResourceData, fallbackResp *FallbackGetResponse) {
	d.Set("model", GetStringValue(fallbackResp.Model, d.Id()))
	d.Set("fallback_models", fallbackResp.FallbackModels)
	d.Set("fallback_type", GetStringValue(fallbackResp.FallbackType, fallbackTypeFromState(d)))
}

func resourceLiteLLMFallbackRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	log.Printf("[INFO] Reading fallback for model: %s", d.Id())

	fallbackResp, err := fetchFallback(context.Background(), client, d.Id(), fallbackTypeFromState(d))
	if err != nil {
		return err
	}
	if fallbackResp == nil {
		log.Printf("[WARN] Fallback for model %s not found, removing from state", d.Id())
		d.SetId("")
		return nil
	}

	setFallbackState(d, fallbackResp)
	return nil
}

func resourceLiteLLMFallbackUpdate(ctx context.Context, d *schema.ResourceData, m interface{}) diag.Diagnostics {
	return diag.FromErr(upsertLiteLLMFallback(ctx, d, m, "updating", waitOrCancel))
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
