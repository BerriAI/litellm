package litellm

import (
	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/validation"
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
				Type:         schema.TypeMap,
				Optional:     true,
				Sensitive:    true,
				Elem:         &schema.Schema{Type: schema.TypeString},
				ExactlyOneOf: []string{"credential_values", "credential_values_wo"},
				Description: "Sensitive credential values (API keys, tokens, etc.). Stored in plaintext in Terraform state; " +
					"use credential_values_wo to keep them out of state.",
			},
			"credential_values_wo": {
				Type:         schema.TypeString,
				Optional:     true,
				WriteOnly:    true,
				Sensitive:    true,
				RequiredWith: []string{"credential_values_wo_version"},
				ValidateFunc: validation.StringIsJSON,
				Description: "Write-only alternative to credential_values: a JSON object of credential values " +
					"(jsonencode({ api_key = var.key })). Sent to the proxy on every apply and never written to " +
					"plan or state. Requires Terraform 1.11 or later. Bump credential_values_wo_version to push a " +
					"new value.",
			},
			"credential_values_wo_version": {
				Type:         schema.TypeInt,
				Optional:     true,
				RequiredWith: []string{"credential_values_wo"},
				Description: "Version of credential_values_wo. Write-only values produce no diff, so change this " +
					"number whenever credential_values_wo changes to trigger an update.",
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
