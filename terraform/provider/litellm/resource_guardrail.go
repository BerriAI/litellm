package litellm

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"reflect"
	"strings"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

const (
	endpointGuardrailCreate = "/guardrails"
	endpointGuardrailByID   = "/guardrails/%s"
	endpointGuardrailInfo   = "/guardrails/%s/info"
)

func resourceLiteLLMGuardrail() *schema.Resource {
	return &schema.Resource{
		Create: resourceLiteLLMGuardrailCreate,
		Read:   resourceLiteLLMGuardrailRead,
		Update: resourceLiteLLMGuardrailUpdate,
		Delete: resourceLiteLLMGuardrailDelete,

		Importer: &schema.ResourceImporter{StateContext: schema.ImportStatePassthroughContext},

		Schema: map[string]*schema.Schema{
			"guardrail_name": {
				Type:        schema.TypeString,
				Required:    true,
				Description: "Human-readable name for the guardrail",
			},
			"guardrail": {
				Type:        schema.TypeString,
				Required:    true,
				Description: "The guardrail integration type (e.g. 'bedrock', 'lakera', 'presidio', 'hide_secrets')",
			},
			"mode": {
				Type:     schema.TypeString,
				Required: true,
				Description: "When to apply the guardrail: a single value ('pre_call', 'post_call', 'during_call', " +
					"'logging_only') or a JSON array of values (e.g. '[\"pre_call\", \"post_call\"]')",
			},
			"default_on": {
				Type:        schema.TypeBool,
				Optional:    true,
				Description: "Whether the guardrail is enabled by default for all requests",
			},
			"litellm_params": {
				Type:             schema.TypeString,
				Optional:         true,
				Sensitive:        true,
				DiffSuppressFunc: guardrailSuppressJSONDiff,
				Description:      "JSON string with additional provider-specific litellm_params (may contain API keys)",
			},
			"guardrail_info": {
				Type:        schema.TypeMap,
				Optional:    true,
				Elem:        &schema.Schema{Type: schema.TypeString},
				Description: "Additional metadata for the guardrail",
			},
			"created_at": {
				Type:     schema.TypeString,
				Computed: true,
			},
		},
	}
}

func guardrailSuppressJSONDiff(k, oldValue, newValue string, d *schema.ResourceData) bool {
	var oldParsed, newParsed interface{}
	if json.Unmarshal([]byte(oldValue), &oldParsed) != nil || json.Unmarshal([]byte(newValue), &newParsed) != nil {
		return false
	}
	return reflect.DeepEqual(oldParsed, newParsed)
}

func guardrailParseMode(mode string) interface{} {
	if strings.HasPrefix(strings.TrimSpace(mode), "[") {
		var modes []string
		if err := json.Unmarshal([]byte(mode), &modes); err == nil {
			return modes
		}
	}
	return mode
}

func buildGuardrailData(d *schema.ResourceData, guardrailID string) (map[string]interface{}, error) {
	litellmParams := map[string]interface{}{
		"guardrail":  d.Get("guardrail").(string),
		"mode":       guardrailParseMode(d.Get("mode").(string)),
		"default_on": d.Get("default_on").(bool),
	}

	if raw := d.Get("litellm_params").(string); raw != "" {
		var extra map[string]interface{}
		if err := json.Unmarshal([]byte(raw), &extra); err != nil {
			return nil, fmt.Errorf("litellm_params is not valid JSON: %w", err)
		}
		for k, v := range extra {
			litellmParams[k] = v
		}
	}

	guardrail := map[string]interface{}{
		"guardrail_name": d.Get("guardrail_name").(string),
		"litellm_params": litellmParams,
	}

	if guardrailID != "" {
		guardrail["guardrail_id"] = guardrailID
	}

	if v, ok := d.GetOk("guardrail_info"); ok {
		guardrail["guardrail_info"] = v
	}

	return map[string]interface{}{"guardrail": guardrail}, nil
}

type guardrailInfoAPIResponse struct {
	GuardrailID   string                 `json:"guardrail_id"`
	GuardrailName string                 `json:"guardrail_name"`
	GuardrailInfo map[string]interface{} `json:"guardrail_info"`
	CreatedAt     string                 `json:"created_at"`
	UpdatedAt     string                 `json:"updated_at"`
}

func resourceLiteLLMGuardrailCreate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	guardrailData, err := buildGuardrailData(d, "")
	if err != nil {
		return err
	}

	log.Printf("[DEBUG] Create guardrail request for: %s", d.Get("guardrail_name").(string))

	resp, err := MakeRequest(client, "POST", endpointGuardrailCreate, guardrailData)
	if err != nil {
		return fmt.Errorf("error creating guardrail: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "creating guardrail"); err != nil {
		return err
	}

	var created guardrailInfoAPIResponse
	if err := json.NewDecoder(resp.Body).Decode(&created); err != nil {
		return fmt.Errorf("error decoding create guardrail response: %w", err)
	}
	if created.GuardrailID == "" {
		return fmt.Errorf("create guardrail response did not contain a guardrail_id")
	}

	d.SetId(created.GuardrailID)
	log.Printf("[INFO] Guardrail created with ID: %s", created.GuardrailID)

	return resourceLiteLLMGuardrailRead(d, m)
}

func resourceLiteLLMGuardrailRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	log.Printf("[INFO] Reading guardrail with ID: %s", d.Id())

	resp, err := MakeRequest(client, "GET", fmt.Sprintf(endpointGuardrailInfo, d.Id()), nil)
	if err != nil {
		return fmt.Errorf("error reading guardrail: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		log.Printf("[WARN] Guardrail with ID %s not found, removing from state", d.Id())
		d.SetId("")
		return nil
	}

	if err := handleResponse(resp, "reading guardrail"); err != nil {
		return err
	}

	var info guardrailInfoAPIResponse
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return fmt.Errorf("error decoding guardrail info response: %w", err)
	}

	d.Set("guardrail_name", info.GuardrailName)
	d.Set("created_at", info.CreatedAt)
	if len(info.GuardrailInfo) > 0 {
		d.Set("guardrail_info", guardrailInfoToStringMap(info.GuardrailInfo))
	}
	// guardrail, mode, default_on and litellm_params are intentionally not read
	// back: the API masks litellm_params values, so state keeps the configured
	// values authoritative.

	return nil
}

func resourceLiteLLMGuardrailUpdate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	guardrailData, err := buildGuardrailData(d, d.Id())
	if err != nil {
		return err
	}

	log.Printf("[DEBUG] Update guardrail request for ID: %s", d.Id())

	resp, err := MakeRequest(client, "PUT", fmt.Sprintf(endpointGuardrailByID, d.Id()), guardrailData)
	if err != nil {
		return fmt.Errorf("error updating guardrail: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "updating guardrail"); err != nil {
		return err
	}

	log.Printf("[INFO] Successfully updated guardrail with ID: %s", d.Id())
	return resourceLiteLLMGuardrailRead(d, m)
}

func resourceLiteLLMGuardrailDelete(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	log.Printf("[INFO] Deleting guardrail with ID: %s", d.Id())

	resp, err := MakeRequest(client, "DELETE", fmt.Sprintf(endpointGuardrailByID, d.Id()), nil)
	if err != nil {
		return fmt.Errorf("error deleting guardrail: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNotFound {
		if err := handleResponse(resp, "deleting guardrail"); err != nil {
			return err
		}
	}

	log.Printf("[INFO] Successfully deleted guardrail with ID: %s", d.Id())
	d.SetId("")
	return nil
}

func guardrailInfoToStringMap(info map[string]interface{}) map[string]string {
	result := make(map[string]string, len(info))
	for k, v := range info {
		result[k] = fmt.Sprintf("%v", v)
	}
	return result
}
