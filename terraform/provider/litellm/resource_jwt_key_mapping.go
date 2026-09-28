package litellm

import (
	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func resourceLiteLLMJWTKeyMapping() *schema.Resource {
	return &schema.Resource{
		Create: resourceLiteLLMJWTKeyMappingCreate,
		Read:   resourceLiteLLMJWTKeyMappingRead,
		Update: resourceLiteLLMJWTKeyMappingUpdate,
		Delete: resourceLiteLLMJWTKeyMappingDelete,

		Importer: &schema.ResourceImporter{
			StateContext: schema.ImportStatePassthroughContext,
		},

		Schema: map[string]*schema.Schema{
			"jwt_claim_name": {
				Type:        schema.TypeString,
				Required:    true,
				ForceNew:    true,
				Description: "Name of the JWT claim to match on, for example client_id, azp or sub. Must match virtual_key_claim_field in the proxy JWT config",
			},
			"jwt_claim_value": {
				Type:        schema.TypeString,
				Required:    true,
				ForceNew:    true,
				Description: "Value of the claim identifying the JWT client. Unique together with jwt_claim_name",
			},
			"key": {
				Type:         schema.TypeString,
				Optional:     true,
				Sensitive:    true,
				ExactlyOneOf: []string{"key", "token_id"},
				Description:  "The virtual key this claim value maps to, as plaintext. The proxy stores only a hash of it and never returns it, so drift on this attribute cannot be detected and Terraform tracks the configured value. litellm_key marks its generated key write-only, so this cannot reference a litellm_key resource; use token_id for that, or supply the plaintext from a variable or a secret manager",
			},
			"token_id": {
				Type:         schema.TypeString,
				Optional:     true,
				ExactlyOneOf: []string{"key", "token_id"},
				Description:  "The SHA-256 hash of the virtual key this claim value maps to, which is what the proxy stores. litellm_key exposes it as token_id, so unlike key it can be referenced directly from a litellm_key resource. Not a secret, so it is not marked sensitive",
			},
			"description": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "Description of the mapping",
			},
			"is_active": {
				Type:        schema.TypeBool,
				Optional:    true,
				Default:     true,
				Description: "Whether the mapping is active. Inactive mappings are ignored during JWT auth",
			},
			"created_at": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Timestamp when the mapping was created",
			},
			"updated_at": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Timestamp when the mapping was last updated",
			},
			"created_by": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "User who created the mapping",
			},
			"updated_by": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "User who last updated the mapping",
			},
		},
	}
}
