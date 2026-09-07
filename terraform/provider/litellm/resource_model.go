package litellm

import (
	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/validation"
)

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
				Computed: true,
			},
			"max_input_tokens": {
				Type:         schema.TypeInt,
				Optional:     true,
				ValidateFunc: validation.IntAtLeast(0),
			},
			"max_output_tokens": {
				Type:         schema.TypeInt,
				Optional:     true,
				ValidateFunc: validation.IntAtLeast(0),
			},
			"input_modalities": {
				Type:        schema.TypeList,
				Optional:    true,
				Computed:    true,
				Description: "Input modalities; empty or omitted preserves remote metadata because SDKv2 collapses empty optional lists",
				Elem: &schema.Schema{
					Type: schema.TypeString,
				},
			},
			"output_modalities": {
				Type:        schema.TypeList,
				Optional:    true,
				Computed:    true,
				Description: "Output modalities; empty or omitted preserves remote metadata because SDKv2 collapses empty optional lists",
				Elem: &schema.Schema{
					Type: schema.TypeString,
				},
			},
			"supports_reasoning": {
				Type:     schema.TypeBool,
				Optional: true,
			},
			"supports_function_calling": {
				Type:     schema.TypeBool,
				Optional: true,
			},
			"supports_vision": {
				Type:     schema.TypeBool,
				Optional: true,
				Computed: true,
			},
			"input_cost_per_character": {
				Type:         schema.TypeFloat,
				Optional:     true,
				Computed:     true,
				ValidateFunc: validation.FloatAtLeast(0),
			},
			"cache_read_input_cost_per_million_tokens": {
				Type:         schema.TypeFloat,
				Optional:     true,
				Computed:     true,
				ValidateFunc: validation.FloatAtLeast(0),
			},
			"default_voice": {
				Type:     schema.TypeString,
				Optional: true,
				Computed: true,
			},
			"probe_language": {
				Type:     schema.TypeString,
				Optional: true,
				Computed: true,
			},
			"probe_text": {
				Type:     schema.TypeString,
				Optional: true,
				Computed: true,
			},
			"probe_skip": {
				Type:     schema.TypeBool,
				Optional: true,
				Computed: true,
			},
			"max_tokens": {
				Type:         schema.TypeInt,
				Optional:     true,
				Computed:     true,
				ValidateFunc: validation.IntAtLeast(0),
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
		},
	}
}
