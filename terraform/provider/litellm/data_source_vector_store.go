package litellm

import (
	"fmt"
	"net/http"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func dataSourceLiteLLMVectorStore() *schema.Resource {
	return &schema.Resource{
		Read: dataSourceLiteLLMVectorStoreRead,

		Schema: map[string]*schema.Schema{
			"vector_store_id": {
				Type:        schema.TypeString,
				Required:    true,
				Description: "Unique identifier for the vector store to retrieve",
			},
			"vector_store_name": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Name of the vector store",
			},
			"custom_llm_provider": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Custom LLM provider for the vector store",
			},
			"vector_store_description": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Description of the vector store",
			},
			"vector_store_metadata": {
				Type:        schema.TypeMap,
				Computed:    true,
				Elem:        &schema.Schema{Type: schema.TypeString},
				Description: "Metadata associated with the vector store",
			},
			"litellm_credential_name": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Name of the LiteLLM credential used",
			},
			"litellm_params": {
				Type:        schema.TypeMap,
				Computed:    true,
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

func dataSourceLiteLLMVectorStoreRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)
	vectorStoreID := d.Get("vector_store_id").(string)

	// Use the info endpoint to get vector store details
	infoRequest := VectorStoreInfoRequest{
		VectorStoreID: vectorStoreID,
	}

	resp, err := MakeRequest(client, "POST", "/vector_store/info", infoRequest)
	if err != nil {
		return fmt.Errorf("failed to read vector store: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return fmt.Errorf("vector store '%s' not found", vectorStoreID)
	}

	var vectorStoreResp VectorStoreResponse
	err = handleVectorStoreAPIResponse(resp, &vectorStoreResp, client)
	if err != nil {
		if err.Error() == "vector_store_not_found" {
			return fmt.Errorf("vector store '%s' not found", vectorStoreID)
		}
		return fmt.Errorf("failed to read vector store: %w", err)
	}

	// Set the data source ID to the vector store ID
	d.SetId(vectorStoreResp.VectorStoreID)
	d.Set("vector_store_id", vectorStoreResp.VectorStoreID)
	d.Set("vector_store_name", vectorStoreResp.VectorStoreName)
	d.Set("custom_llm_provider", vectorStoreResp.CustomLLMProvider)
	d.Set("vector_store_description", vectorStoreResp.VectorStoreDescription)
	d.Set("vector_store_metadata", vectorStoreResp.VectorStoreMetadata)
	d.Set("litellm_credential_name", vectorStoreResp.LiteLLMCredentialName)
	d.Set("litellm_params", vectorStoreResp.LiteLLMParams)
	d.Set("created_at", vectorStoreResp.CreatedAt)
	d.Set("updated_at", vectorStoreResp.UpdatedAt)

	return nil
}
