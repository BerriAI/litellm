package litellm

import (
	"fmt"
	"net/http"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func resourceLiteLLMVectorStoreCreate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	vectorStoreName := d.Get("vector_store_name").(string)
	customLLMProvider := d.Get("custom_llm_provider").(string)
	vectorStoreDescription := d.Get("vector_store_description").(string)
	vectorStoreMetadata := d.Get("vector_store_metadata").(map[string]interface{})
	litellmCredentialName := d.Get("litellm_credential_name").(string)
	litellmParams := d.Get("litellm_params").(map[string]interface{})

	// Convert metadata to map[string]interface{} for JSON
	metadataMap := make(map[string]interface{})
	for k, v := range vectorStoreMetadata {
		metadataMap[k] = v
	}

	// Convert litellm_params to map[string]interface{} for JSON
	paramsMap := make(map[string]interface{})
	for k, v := range litellmParams {
		paramsMap[k] = v
	}

	vectorStoreRequest := VectorStoreRequest{
		CustomLLMProvider:      customLLMProvider,
		VectorStoreName:        vectorStoreName,
		VectorStoreDescription: vectorStoreDescription,
		VectorStoreMetadata:    metadataMap,
		LiteLLMCredentialName:  litellmCredentialName,
		LiteLLMParams:          paramsMap,
	}

	resp, err := MakeRequest(client, "POST", "/vector_store/new", vectorStoreRequest)
	if err != nil {
		return fmt.Errorf("failed to create vector store: %w", err)
	}
	defer resp.Body.Close()

	err = handleVectorStoreAPIResponse(resp, nil, client)
	if err != nil {
		return fmt.Errorf("failed to create vector store: %w", err)
	}

	// Set the resource ID to the vector store name for now
	// We'll update this after reading the response to get the actual ID
	d.SetId(vectorStoreName)

	return resourceLiteLLMVectorStoreRead(d, m)
}

func resourceLiteLLMVectorStoreRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)
	vectorStoreID := d.Id()

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
		d.SetId("")
		return nil
	}

	var vectorStoreResp VectorStoreResponse
	err = handleVectorStoreAPIResponse(resp, &vectorStoreResp, client)
	if err != nil {
		if err.Error() == "vector_store_not_found" {
			d.SetId("")
			return nil
		}
		return fmt.Errorf("failed to read vector store: %w", err)
	}

	// Update the resource ID to the actual vector store ID from the response
	if vectorStoreResp.VectorStoreID != "" {
		d.SetId(vectorStoreResp.VectorStoreID)
	}

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

func resourceLiteLLMVectorStoreUpdate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)
	vectorStoreID := d.Id()

	vectorStoreName := d.Get("vector_store_name").(string)
	customLLMProvider := d.Get("custom_llm_provider").(string)
	vectorStoreDescription := d.Get("vector_store_description").(string)
	vectorStoreMetadata := d.Get("vector_store_metadata").(map[string]interface{})

	// Convert metadata to map[string]interface{} for JSON
	metadataMap := make(map[string]interface{})
	for k, v := range vectorStoreMetadata {
		metadataMap[k] = v
	}

	vectorStoreRequest := VectorStoreRequest{
		VectorStoreID:          vectorStoreID,
		CustomLLMProvider:      customLLMProvider,
		VectorStoreName:        vectorStoreName,
		VectorStoreDescription: vectorStoreDescription,
		VectorStoreMetadata:    metadataMap,
	}

	resp, err := MakeRequest(client, "POST", "/vector_store/update", vectorStoreRequest)
	if err != nil {
		return fmt.Errorf("failed to update vector store: %w", err)
	}
	defer resp.Body.Close()

	err = handleVectorStoreAPIResponse(resp, nil, client)
	if err != nil {
		return fmt.Errorf("failed to update vector store: %w", err)
	}

	return resourceLiteLLMVectorStoreRead(d, m)
}

func resourceLiteLLMVectorStoreDelete(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)
	vectorStoreID := d.Id()

	deleteRequest := VectorStoreDeleteRequest{
		VectorStoreID: vectorStoreID,
	}

	resp, err := MakeRequest(client, "POST", "/vector_store/delete", deleteRequest)
	if err != nil {
		return fmt.Errorf("failed to delete vector store: %w", err)
	}
	defer resp.Body.Close()

	err = handleVectorStoreAPIResponse(resp, nil, client)
	if err != nil {
		if err.Error() == "vector_store_not_found" {
			d.SetId("")
			return nil
		}
		return fmt.Errorf("failed to delete vector store: %w", err)
	}

	d.SetId("")
	return nil
}
