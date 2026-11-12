package litellm

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"reflect"
	"time"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/validation"
)

const endpointFallbackCreate = "/fallback"

// fallbackReadBackDelays are the waits between read-backs after a fallback write. GET /fallback/{model}
// answers from the in-memory router of whichever replica serves it, and replicas reload fallbacks from
// the database only every proxy_config_reload_interval_seconds (30 s by default), so on a multi-replica
// proxy the first read-back often reaches a replica that has not loaded the write yet. The waits add up
// to 45 s, longer than one default reload.
var fallbackReadBackDelays = []time.Duration{
	1 * time.Second, 2 * time.Second, 4 * time.Second, 8 * time.Second, 10 * time.Second, 10 * time.Second, 10 * time.Second,
}

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
	return readFallbackAfterWrite(d, m)
}

// readFallbackAfterWrite reads a fallback back after POST /fallback, retrying while the proxy answers 404
// or still returns the previous fallback_models, since the read can reach a replica that has not reloaded
// the write yet (see fallbackReadBackDelays). A refresh keeps treating 404 as removed, through
// resourceLiteLLMFallbackRead, so drift is still detected.
func readFallbackAfterWrite(d *schema.ResourceData, m interface{}) error {
	id := d.Id()
	want := expandStringList(d.Get("fallback_models").([]interface{}))

	for attempt := 0; ; attempt++ {
		if err := resourceLiteLLMFallbackRead(d, m); err != nil {
			return err
		}
		got := expandStringList(d.Get("fallback_models").([]interface{}))
		if d.Id() != "" && reflect.DeepEqual(got, want) {
			return nil
		}
		if attempt == len(fallbackReadBackDelays) {
			break
		}
		delay := fallbackReadBackDelays[attempt]
		if d.Id() == "" {
			log.Printf("[INFO] Fallback for model %s not visible yet (replica not reloaded), retrying in %v", id, delay)
		} else {
			log.Printf("[INFO] Fallback for model %s still shows %v instead of %v, retrying in %v", id, got, want, delay)
		}
		// the read cleared the ID and may have stored the old models; put back what was written
		d.SetId(id)
		if err := d.Set("fallback_models", want); err != nil {
			return err
		}
		time.Sleep(delay)
	}

	d.SetId(id)
	return fmt.Errorf("fallback for model %s was written but not read back with fallback_models %v after %d attempts; the proxy may still be reloading it, so re-running apply should resolve this", id, want, len(fallbackReadBackDelays)+1)
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
