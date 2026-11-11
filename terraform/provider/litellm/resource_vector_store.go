package litellm

import (
	"fmt"
	"slices"

	"github.com/hashicorp/go-cty/cty"
	"github.com/hashicorp/terraform-plugin-sdk/v2/diag"
	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

var secretParamKeys = []string{
	"api_key",
	"aws_access_key_id",
	"aws_secret_access_key",
	"aws_session_token",
	"vertex_credentials",
	"client_secret",
	"password",
	"token",
}

// warnOnSecretKeys flags secret-looking keys in an in-state map at plan time,
// since the map is written to state in plaintext.
func warnOnSecretKeys(v interface{}, path cty.Path) diag.Diagnostics {
	m, ok := v.(map[string]interface{})
	if !ok {
		return nil
	}
	var diags diag.Diagnostics
	for key := range m {
		if slices.Contains(secretParamKeys, key) {
			diags = append(diags, diag.Diagnostic{
				Severity:      diag.Warning,
				Summary:       fmt.Sprintf("%q is stored in plaintext in Terraform state", key),
				Detail:        "Move it to a litellm_credential using credential_values_wo and reference it through litellm_credential_name.",
				AttributePath: path,
			})
		}
	}
	return diags
}

func resourceLiteLLMVectorStore() *schema.Resource {
	return &schema.Resource{
		Create: resourceLiteLLMVectorStoreCreate,
		Read:   resourceLiteLLMVectorStoreRead,
		Update: resourceLiteLLMVectorStoreUpdate,
		Delete: resourceLiteLLMVectorStoreDelete,
		Importer: &schema.ResourceImporter{
			StateContext: schema.ImportStatePassthroughContext,
		},

		Schema: map[string]*schema.Schema{
			"vector_store_id": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Unique identifier for the vector store",
			},
			"vector_store_name": {
				Type:        schema.TypeString,
				Required:    true,
				Description: "Name of the vector store",
			},
			"custom_llm_provider": {
				Type:        schema.TypeString,
				Required:    true,
				Description: "Custom LLM provider for the vector store",
			},
			"vector_store_description": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "Description of the vector store",
			},
			"vector_store_metadata": {
				Type:        schema.TypeMap,
				Optional:    true,
				Elem:        &schema.Schema{Type: schema.TypeString},
				Description: "Metadata associated with the vector store",
			},
			"litellm_credential_name": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "Name of the LiteLLM credential to use",
			},
			"litellm_params": {
				Type:             schema.TypeMap,
				Optional:         true,
				Sensitive:        true,
				Elem:             &schema.Schema{Type: schema.TypeString},
				ValidateDiagFunc: warnOnSecretKeys,
				Description: "Additional LiteLLM parameters. Stored in plaintext in Terraform state, so keep secrets " +
					"in a litellm_credential (credential_values_wo) referenced through litellm_credential_name.",
			},
			"created_at": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Timestamp when the vector store was created",
			},
			"updated_at": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Timestamp when the vector store was last updated",
			},
		},
	}
}
