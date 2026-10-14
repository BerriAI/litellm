package litellm

import (
	"github.com/hashicorp/go-cty/cty"
	"github.com/hashicorp/terraform-plugin-sdk/v2/diag"
	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/validation"
)

var modelInfoKeysWithArguments = map[string]struct{}{
	"id":           {},
	"db_model":     {},
	"base_model":   {},
	"tier":         {},
	"mode":         {},
	"team_id":      {},
	"display_name": {},
}

func validateAdditionalModelInfo(value interface{}, _ cty.Path) diag.Diagnostics {
	for key := range value.(map[string]interface{}) {
		if _, owned := modelInfoKeysWithArguments[key]; owned {
			return diag.Errorf("additional_model_info cannot set %q, it is set by the provider or by its own argument", key)
		}
	}
	return nil
}

func resourceLiteLLMModel() *schema.Resource {
	return &schema.Resource{
		Create: resourceLiteLLMModelCreate,
		Read:   resourceLiteLLMModelRead,
		Update: resourceLiteLLMModelUpdate,
		Delete: resourceLiteLLMModelDelete,
		Importer: &schema.ResourceImporter{
			StateContext: schema.ImportStatePassthroughContext,
		},

		Schema: map[string]*schema.Schema{
			"model_name": {
				Type:     schema.TypeString,
				Required: true,
			},
			"custom_llm_provider": {
				Type:     schema.TypeString,
				Required: true,
			},
			"tpm": {
				Type:     schema.TypeInt,
				Optional: true,
			},
			"rpm": {
				Type:     schema.TypeInt,
				Optional: true,
			},
			"reasoning_effort": {
				Type:     schema.TypeString,
				Optional: true,
				ValidateFunc: validation.StringInSlice([]string{
					"low",
					"medium",
					"high",
				}, false),
			},
			"thinking_enabled": {
				Type:     schema.TypeBool,
				Optional: true,
				Default:  false,
			},
			"thinking_budget_tokens": {
				Type:     schema.TypeInt,
				Optional: true,
				Default:  1024,
				DiffSuppressFunc: func(k, old, new string, d *schema.ResourceData) bool {
					// Only include thinking_budget_tokens in the diff if thinking_enabled is true
					return !d.Get("thinking_enabled").(bool)
				},
			},
			"merge_reasoning_content_in_choices": {
				Type:     schema.TypeBool,
				Optional: true,
			},
			"model_api_key": {
				Type:      schema.TypeString,
				Optional:  true,
				Sensitive: true,
			},
			"model_api_base": {
				Type:     schema.TypeString,
				Optional: true,
			},
			"api_version": {
				Type:     schema.TypeString,
				Optional: true,
			},
			"base_model": {
				Type:     schema.TypeString,
				Required: true,
			},
			"pricing_base_model": {
				// Optional pricing key fed to model_info.base_model, DECOUPLED
				// from routing. When set, litellm_params.model still routes via
				// base_model, but cost is looked up against this key (e.g.
				// "us/gpt-4.1-2025-04-14" for Azure Data Zone pricing).
				Type:     schema.TypeString,
				Optional: true,
			},
			"tier": {
				Type:     schema.TypeString,
				Optional: true,
				Default:  "free",
			},
			"team_id": {
				Type:     schema.TypeString,
				Optional: true,
			},
			"display_name": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "Human-readable name returned as display_name by /v1/models, shown in client model pickers instead of model_name",
			},
			"mode": {
				Type:     schema.TypeString,
				Optional: true,
				ValidateFunc: validation.StringInSlice([]string{
					"completion",
					"embedding",
					"image_generation",
					"chat",
					"moderation",
					"audio_transcription",
					"audio_speech",
					"rerank",
				}, false),
			},
			"input_cost_per_million_tokens": {
				Type:     schema.TypeFloat,
				Optional: true,
			},
			"output_cost_per_million_tokens": {
				Type:     schema.TypeFloat,
				Optional: true,
			},
			"input_cost_per_pixel": {
				Type:     schema.TypeFloat,
				Optional: true,
			},
			"output_cost_per_pixel": {
				Type:     schema.TypeFloat,
				Optional: true,
			},
			"input_cost_per_second": {
				Type:     schema.TypeFloat,
				Optional: true,
			},
			"output_cost_per_second": {
				Type:     schema.TypeFloat,
				Optional: true,
			},
			"aws_access_key_id": {
				Type:      schema.TypeString,
				Optional:  true,
				Sensitive: true,
			},
			"aws_secret_access_key": {
				Type:      schema.TypeString,
				Optional:  true,
				Sensitive: true,
			},
			"aws_region_name": {
				Type:     schema.TypeString,
				Optional: true,
			},
			"aws_session_name": {
				Type:      schema.TypeString,
				Optional:  true,
				Sensitive: true,
			},
			"aws_role_name": {
				Type:      schema.TypeString,
				Optional:  true,
				Sensitive: true,
			},
			"vertex_project": {
				Type:      schema.TypeString,
				Optional:  true,
				Sensitive: true,
			},
			"vertex_location": {
				Type:      schema.TypeString,
				Optional:  true,
				Sensitive: true,
			},
			"vertex_credentials": {
				Type:      schema.TypeString,
				Optional:  true,
				Sensitive: true,
			},
			"litellm_credential_name": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "Name of the LiteLLM credential to use",
			},
			"additional_litellm_params": {
				Type:     schema.TypeMap,
				Optional: true,
				Elem: &schema.Schema{
					Type: schema.TypeString,
				},
				Description: "Additional parameters to pass to litellm_params beyond the standard ones",
			},
			"additional_model_info": {
				Type:     schema.TypeMap,
				Optional: true,
				Elem: &schema.Schema{
					Type: schema.TypeString,
				},
				ValidateDiagFunc: validateAdditionalModelInfo,
				Description:      "Additional fields to set in model_info beyond the standard ones, such as max_input_tokens or bedrock_converse_supports_strict_tools",
			},
		},
	}
}
