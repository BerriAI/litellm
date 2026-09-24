package litellm

import (
	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func resourceLiteLLMCredential() *schema.Resource {
	return &schema.Resource{
		Create: resourceLiteLLMCredentialCreate,
		Read:   resourceLiteLLMCredentialRead,
		Update: resourceLiteLLMCredentialUpdate,
		Delete: resourceLiteLLMCredentialDelete,
		Importer: &schema.ResourceImporter{
			StateContext: schema.ImportStatePassthroughContext,
		},

		Schema: map[string]*schema.Schema{
			"credential_name": {
				Type:        schema.TypeString,
				Required:    true,
				ForceNew:    true,
				Description: "Name of the credential",
			},
			"model_id": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "Model ID associated with this credential",
			},
			"credential_info": {
				Type:        schema.TypeMap,
				Optional:    true,
				Elem:        &schema.Schema{Type: schema.TypeString},
				Description: "Additional information about the credential",
			},
			"credential_values": {
				Type:        schema.TypeMap,
				Required:    true,
				Sensitive:   true,
				Elem:        &schema.Schema{Type: schema.TypeString},
				Description: "Sensitive credential values (API keys, tokens, etc.)",
			},
			"adopt_existing": {
				Type:     schema.TypeBool,
				Optional: true,
				Default:  false,
				Description: "Take over a credential of this name that already exists on the proxy instead of failing. " +
					"Off by default: create reports the conflict and points at `terraform import`, so an apply never " +
					"silently overwrites a credential it does not manage. Turning this on overwrites the existing " +
					"credential's values with the ones in this configuration.",
			},
		},
	}
}
