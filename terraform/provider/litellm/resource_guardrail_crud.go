package litellm

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func buildGuardrailSpec(d *schema.ResourceData) (GuardrailSpec, error) {
	litellmParams := map[string]interface{}{
		"guardrail":  d.Get("guardrail").(string),
		"mode":       parseGuardrailMode(d.Get("mode").(string)),
		"default_on": d.Get("default_on").(bool),
	}

	if raw, ok := d.GetOk("litellm_params"); ok {
		extra, err := decodeJSONObject(raw.(string))
		if err != nil {
			return GuardrailSpec{}, fmt.Errorf("litellm_params must be a JSON object: %w", err)
		}
		for k, v := range extra {
			litellmParams[k] = v
		}
	}

	spec := GuardrailSpec{
		GuardrailName: d.Get("guardrail_name").(string),
		LiteLLMParams: litellmParams,
	}

	if raw, ok := d.GetOk("guardrail_info"); ok {
		info, err := decodeJSONObject(raw.(string))
		if err != nil {
			return GuardrailSpec{}, fmt.Errorf("guardrail_info must be a JSON object: %w", err)
		}
		spec.GuardrailInfo = info
	}

	return spec, nil
}

// parseGuardrailMode returns a []string when the value is a JSON array, otherwise the raw string.
func parseGuardrailMode(mode string) interface{} {
	trimmed := strings.TrimSpace(mode)
	if strings.HasPrefix(trimmed, "[") {
		var modes []string
		if err := json.Unmarshal([]byte(trimmed), &modes); err == nil {
			return modes
		}
	}
	return mode
}

func decodeJSONObject(raw string) (map[string]interface{}, error) {
	if strings.TrimSpace(raw) == "" {
		return nil, nil
	}
	var out map[string]interface{}
	if err := json.Unmarshal([]byte(raw), &out); err != nil {
		return nil, err
	}
	return out, nil
}

func resourceLiteLLMGuardrailCreate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	spec, err := buildGuardrailSpec(d)
	if err != nil {
		return err
	}

	resp, err := MakeRequest(client, "POST", "/guardrails", GuardrailRequest{Guardrail: spec})
	if err != nil {
		return fmt.Errorf("failed to create guardrail: %w", err)
	}
	defer resp.Body.Close()

	var guardrailResp GuardrailResponse
	if err := handleGuardrailAPIResponse(resp, &guardrailResp, client); err != nil {
		return fmt.Errorf("failed to create guardrail: %w", err)
	}

	if guardrailResp.GuardrailID == "" {
		return fmt.Errorf("guardrail created but the API did not return a guardrail_id")
	}

	d.SetId(guardrailResp.GuardrailID)

	return resourceLiteLLMGuardrailRead(d, m)
}

func resourceLiteLLMGuardrailRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	resp, err := MakeRequest(client, "GET", fmt.Sprintf("/guardrails/%s/info", d.Id()), nil)
	if err != nil {
		return fmt.Errorf("failed to read guardrail: %w", err)
	}
	defer resp.Body.Close()

	var guardrailResp GuardrailResponse
	if err := handleGuardrailAPIResponse(resp, &guardrailResp, client); err != nil {
		if err.Error() == "guardrail_not_found" {
			d.SetId("")
			return nil
		}
		return fmt.Errorf("failed to read guardrail: %w", err)
	}

	d.Set("guardrail_id", guardrailResp.GuardrailID)
	d.Set("guardrail_name", guardrailResp.GuardrailName)
	d.Set("created_at", guardrailResp.CreatedAt)
	d.Set("updated_at", guardrailResp.UpdatedAt)

	// Reconcile the non-sensitive params that live inside litellm_params. The
	// rest of litellm_params is masked by the proxy on read, so persisting it
	// would produce perpetual diffs; the configured value is left untouched.
	if guardrail, ok := guardrailResp.LiteLLMParams["guardrail"].(string); ok {
		d.Set("guardrail", guardrail)
	}
	if defaultOn, ok := guardrailResp.LiteLLMParams["default_on"].(bool); ok {
		d.Set("default_on", defaultOn)
	}
	if err := setGuardrailMode(d, guardrailResp.LiteLLMParams["mode"]); err != nil {
		return err
	}

	if len(guardrailResp.GuardrailInfo) > 0 {
		encoded, err := json.Marshal(guardrailResp.GuardrailInfo)
		if err != nil {
			return fmt.Errorf("failed to encode guardrail_info: %w", err)
		}
		d.Set("guardrail_info", string(encoded))
	}

	return nil
}

func setGuardrailMode(d *schema.ResourceData, mode interface{}) error {
	switch value := mode.(type) {
	case string:
		d.Set("mode", value)
	case []interface{}:
		encoded, err := json.Marshal(value)
		if err != nil {
			return fmt.Errorf("failed to encode guardrail mode: %w", err)
		}
		d.Set("mode", string(encoded))
	}
	return nil
}

func resourceLiteLLMGuardrailUpdate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	spec, err := buildGuardrailSpec(d)
	if err != nil {
		return err
	}
	spec.GuardrailID = d.Id()

	resp, err := MakeRequest(client, "PUT", fmt.Sprintf("/guardrails/%s", d.Id()), GuardrailRequest{Guardrail: spec})
	if err != nil {
		return fmt.Errorf("failed to update guardrail: %w", err)
	}
	defer resp.Body.Close()

	if err := handleGuardrailAPIResponse(resp, nil, client); err != nil {
		return fmt.Errorf("failed to update guardrail: %w", err)
	}

	return resourceLiteLLMGuardrailRead(d, m)
}

func resourceLiteLLMGuardrailDelete(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	resp, err := MakeRequest(client, "DELETE", fmt.Sprintf("/guardrails/%s", d.Id()), nil)
	if err != nil {
		return fmt.Errorf("failed to delete guardrail: %w", err)
	}
	defer resp.Body.Close()

	if err := handleGuardrailAPIResponse(resp, nil, client); err != nil {
		if err.Error() == "guardrail_not_found" {
			d.SetId("")
			return nil
		}
		return fmt.Errorf("failed to delete guardrail: %w", err)
	}

	d.SetId("")
	return nil
}

// suppressEquivalentJSON suppresses diffs between two JSON strings that are
// semantically equal but differ in key ordering or whitespace.
func suppressEquivalentJSON(_, oldValue, newValue string, _ *schema.ResourceData) bool {
	if oldValue == newValue {
		return true
	}
	var oldObj, newObj interface{}
	if err := json.Unmarshal([]byte(oldValue), &oldObj); err != nil {
		return false
	}
	if err := json.Unmarshal([]byte(newValue), &newObj); err != nil {
		return false
	}
	oldNorm, err := json.Marshal(oldObj)
	if err != nil {
		return false
	}
	newNorm, err := json.Marshal(newObj)
	if err != nil {
		return false
	}
	return string(oldNorm) == string(newNorm)
}
