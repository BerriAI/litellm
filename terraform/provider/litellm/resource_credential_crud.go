package litellm

import (
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

// retryCredentialRead attempts to read a credential with exponential backoff.
// If the read path clears the ID (e.g., transient 404 right after create),
// we treat it as retryable instead of accepting an empty state.
func retryCredentialRead(d *schema.ResourceData, m interface{}, maxRetries int) error {
	var err error
	delay := 1 * time.Second
	maxDelay := 10 * time.Second
	origID := d.Id()

	for i := 0; i < maxRetries; i++ {
		log.Printf("[INFO] Attempting to read credential (attempt %d/%d)", i+1, maxRetries)

		err = resourceLiteLLMCredentialRead(d, m)
		// If read succeeded but wiped the ID, treat as not found so we retry.
		if err == nil && d.Id() == "" {
			d.SetId(origID)
			err = fmt.Errorf("credential_not_found")
		}

		if err == nil {
			log.Printf("[INFO] Successfully read credential after %d attempts", i+1)
			return nil
		}

		if !strings.Contains(err.Error(), "credential_not_found") {
			return err
		}

		if i < maxRetries-1 {
			log.Printf("[INFO] Credential not found yet, retrying in %v...", delay)
			time.Sleep(delay)

			delay *= 2
			if delay > maxDelay {
				delay = maxDelay
			}
		}
	}

	log.Printf("[WARN] Failed to read credential after %d attempts: %v", maxRetries, err)
	return err
}

func resourceLiteLLMCredentialCreate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	credentialName := d.Get("credential_name").(string)
	modelID := d.Get("model_id").(string)
	credentialInfo := d.Get("credential_info").(map[string]interface{})
	credentialValues := d.Get("credential_values").(map[string]interface{})

	// Convert credential_info to map[string]interface{} for JSON
	credInfoMap := make(map[string]interface{})
	for k, v := range credentialInfo {
		credInfoMap[k] = v
	}

	// Convert credential_values to map[string]interface{} for JSON
	credValuesMap := make(map[string]interface{})
	for k, v := range credentialValues {
		credValuesMap[k] = v
	}

	credentialRequest := CredentialRequest{
		CredentialName:   credentialName,
		ModelID:          modelID,
		CredentialInfo:   credInfoMap,
		CredentialValues: credValuesMap,
	}

	resp, err := MakeRequest(client, "POST", "/credentials", credentialRequest)
	if err != nil {
		return fmt.Errorf("failed to create credential: %w", err)
	}
	defer resp.Body.Close()

	err = handleCredentialAPIResponse(resp, nil, client)
	if err != nil {
		return fmt.Errorf("failed to create credential: %w", err)
	}

	// Set the resource ID to the credential name
	d.SetId(credentialName)

	log.Printf("[INFO] Credential created with name %s. Starting retry mechanism to read the credential...", credentialName)
	return retryCredentialRead(d, m, 5)
}

func resourceLiteLLMCredentialRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)
	credentialName := d.Id()

	// Try to get credential by name first
	modelID := d.Get("model_id").(string)
	endpoint := fmt.Sprintf("/credentials/by_name/%s", credentialName)
	if modelID != "" {
		endpoint += fmt.Sprintf("?model_id=%s", modelID)
	}

	resp, err := MakeRequest(client, "GET", endpoint, nil)
	if err != nil {
		return fmt.Errorf("failed to read credential: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		d.SetId("")
		return nil
	}

	var credentialResp CredentialResponse
	err = handleCredentialAPIResponse(resp, &credentialResp, client)
	if err != nil {
		if err.Error() == "credential_not_found" {
			d.SetId("")
			return nil
		}
		return fmt.Errorf("failed to read credential: %w", err)
	}

	d.Set("credential_name", credentialResp.CredentialName)
	d.Set("credential_info", credentialResp.CredentialInfo)
	// Note: We don't set credential_values from the response for security reasons
	// The API might not return sensitive values, and we want to preserve what's in state

	return nil
}

func resourceLiteLLMCredentialUpdate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)
	credentialName := d.Id()

	credentialInfo := d.Get("credential_info").(map[string]interface{})
	credentialValues := d.Get("credential_values").(map[string]interface{})

	// Convert credential_info to map[string]interface{} for JSON
	credInfoMap := make(map[string]interface{})
	for k, v := range credentialInfo {
		credInfoMap[k] = v
	}

	// Convert credential_values to map[string]interface{} for JSON
	credValuesMap := make(map[string]interface{})
	for k, v := range credentialValues {
		credValuesMap[k] = v
	}

	credentialRequest := CredentialRequest{
		CredentialName:   credentialName,
		CredentialInfo:   credInfoMap,
		CredentialValues: credValuesMap,
	}

	endpoint := fmt.Sprintf("/credentials/%s", credentialName)
	resp, err := MakeRequest(client, "PATCH", endpoint, credentialRequest)
	if err != nil {
		return fmt.Errorf("failed to update credential: %w", err)
	}
	defer resp.Body.Close()

	err = handleCredentialAPIResponse(resp, nil, client)
	if err != nil {
		return fmt.Errorf("failed to update credential: %w", err)
	}

	log.Printf("[INFO] Credential updated with name %s. Starting retry mechanism to read the credential...", credentialName)
	return retryCredentialRead(d, m, 5)
}

func resourceLiteLLMCredentialDelete(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)
	credentialName := d.Id()

	endpoint := fmt.Sprintf("/credentials/%s", credentialName)
	resp, err := MakeRequest(client, "DELETE", endpoint, nil)
	if err != nil {
		return fmt.Errorf("failed to delete credential: %w", err)
	}
	defer resp.Body.Close()

	err = handleCredentialAPIResponse(resp, nil, client)
	if err != nil {
		if err.Error() == "credential_not_found" {
			d.SetId("")
			return nil
		}
		return fmt.Errorf("failed to delete credential: %w", err)
	}

	d.SetId("")
	return nil
}
