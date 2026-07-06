package litellm

import (
	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func resourceLiteLLMVectorStore() *schema.Resource {
	return &schema.Resource{
		Create: resourceLiteLLMVectorStoreCreate,
		Read:   resourceLiteLLMVectorStoreRead,
		Update: resourceLiteLLMVectorStoreUpdate,
		Delete: resourceLiteLLMVectorStoreDelete,

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
				Type:        schema.TypeMap,
				Optional:    true,
				Elem:        &schema.Schema{Type: schema.TypeString},
				Description: "Additional LiteLLM parameters",
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
