package litellm

import (
	"encoding/json"
	"fmt"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func dataSourceLiteLLMGuardrail() *schema.Resource {
	return &schema.Resource{
		Read: dataSourceLiteLLMGuardrailRead,

		Schema: map[string]*schema.Schema{
			"guardrail_id": {
				Type:        schema.TypeString,
				Required:    true,
				Description: "Unique identifier of the guardrail to retrieve",
			},
			"guardrail_name": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Human-readable name for the guardrail",
			},
			"guardrail": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "The guardrail integration type",
			},
			"mode": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "When the guardrail runs. A JSON array string when the guardrail runs in multiple modes",
			},
			"default_on": {
				Type:        schema.TypeBool,
				Computed:    true,
				Description: "Whether the guardrail runs on every request by default",
			},
			"litellm_params": {
				Type:        schema.TypeString,
				Computed:    true,
				Sensitive:   true,
				Description: "JSON object of the guardrail's litellm_params, with sensitive values masked by the proxy",
			},
			"guardrail_info": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "JSON object of free-form metadata stored alongside the guardrail",
			},
			"created_at": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Timestamp when the guardrail was created",
			},
			"updated_at": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Timestamp when the guardrail was last updated",
			},
		},
	}
}

func dataSourceLiteLLMGuardrailRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)
	guardrailID := d.Get("guardrail_id").(string)

	resp, err := MakeRequest(client, "GET", fmt.Sprintf("/guardrails/%s/info", guardrailID), nil)
	if err != nil {
		return fmt.Errorf("failed to read guardrail: %w", err)
	}
	defer resp.Body.Close()

	var guardrailResp GuardrailResponse
	if err := handleGuardrailAPIResponse(resp, &guardrailResp, client); err != nil {
		if err.Error() == "guardrail_not_found" {
			return fmt.Errorf("guardrail '%s' not found", guardrailID)
		}
		return fmt.Errorf("failed to read guardrail: %w", err)
	}

	d.SetId(guardrailResp.GuardrailID)
	d.Set("guardrail_id", guardrailResp.GuardrailID)
	d.Set("guardrail_name", guardrailResp.GuardrailName)
	d.Set("created_at", guardrailResp.CreatedAt)
	d.Set("updated_at", guardrailResp.UpdatedAt)

	if guardrail, ok := guardrailResp.LiteLLMParams["guardrail"].(string); ok {
		d.Set("guardrail", guardrail)
	}
	if defaultOn, ok := guardrailResp.LiteLLMParams["default_on"].(bool); ok {
		d.Set("default_on", defaultOn)
	}
	if err := setGuardrailMode(d, guardrailResp.LiteLLMParams["mode"]); err != nil {
		return err
	}

	if len(guardrailResp.LiteLLMParams) > 0 {
		encoded, err := json.Marshal(guardrailResp.LiteLLMParams)
		if err != nil {
			return fmt.Errorf("failed to encode litellm_params: %w", err)
		}
		d.Set("litellm_params", string(encoded))
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
