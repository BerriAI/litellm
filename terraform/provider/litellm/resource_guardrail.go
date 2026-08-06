package litellm

import (
	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func resourceLiteLLMGuardrail() *schema.Resource {
	return &schema.Resource{
		Create: resourceLiteLLMGuardrailCreate,
		Read:   resourceLiteLLMGuardrailRead,
		Update: resourceLiteLLMGuardrailUpdate,
		Delete: resourceLiteLLMGuardrailDelete,
		Importer: &schema.ResourceImporter{
			StateContext: schema.ImportStatePassthroughContext,
		},

		Schema: map[string]*schema.Schema{
			"guardrail_name": {
				Type:        schema.TypeString,
				Required:    true,
				Description: "Human-readable name for the guardrail",
			},
			"guardrail": {
				Type:        schema.TypeString,
				Required:    true,
				Description: "The guardrail integration type (e.g. \"bedrock\", \"presidio\", \"lakera_v2\", \"aporia\")",
			},
			"mode": {
				Type:        schema.TypeString,
				Required:    true,
				Description: "When to run the guardrail: \"pre_call\", \"post_call\", \"during_call\", or \"logging_only\". A JSON array string (e.g. \"[\\\"pre_call\\\", \\\"post_call\\\"]\") runs the guardrail in multiple modes",
			},
			"default_on": {
				Type:        schema.TypeBool,
				Optional:    true,
				Default:     false,
				Description: "Whether the guardrail runs on every request by default",
			},
			"litellm_params": {
				Type:             schema.TypeString,
				Optional:         true,
				Sensitive:        true,
				DiffSuppressFunc: suppressEquivalentJSON,
				Description:      "JSON object of additional provider-specific parameters merged into litellm_params (e.g. guardrailIdentifier, api_key, api_base). Stored unencrypted in state, so prefer referencing secrets from the environment. Values are masked when read back, so this field is not refreshed from the API to avoid spurious diffs",
			},
			"guardrail_info": {
				Type:             schema.TypeString,
				Optional:         true,
				DiffSuppressFunc: suppressEquivalentJSON,
				Description:      "JSON object of free-form metadata stored alongside the guardrail",
			},
			"guardrail_id": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Unique identifier for the guardrail",
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
