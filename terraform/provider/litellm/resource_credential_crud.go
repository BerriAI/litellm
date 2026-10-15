package litellm

import (
	"errors"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

const (
	endpointCredential               = "/credentials/%s"
	endpointCredentialByName         = "/credentials/by_name/%s"
	endpointCredentialByNameForModel = "/credentials/by_name/%s?model_id=%s"
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

func credentialRequestFromResource(d *schema.ResourceData, credentialName string) CredentialRequest {
	credInfoMap := make(map[string]interface{})
	for k, v := range d.Get("credential_info").(map[string]interface{}) {
		credInfoMap[k] = v
	}
	credValuesMap := make(map[string]interface{})
	for k, v := range d.Get("credential_values").(map[string]interface{}) {
		credValuesMap[k] = v
	}
	return CredentialRequest{
		CredentialName:   credentialName,
		ModelID:          d.Get("model_id").(string),
		CredentialInfo:   credInfoMap,
		CredentialValues: credValuesMap,
	}
}

func resourceLiteLLMCredentialCreate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)
	credentialName := d.Get("credential_name").(string)

	resp, err := MakeRequest(client, "POST", "/credentials", credentialRequestFromResource(d, credentialName))
	if err != nil {
		return fmt.Errorf("failed to create credential: %w", err)
	}
	defer resp.Body.Close()

	err = handleCredentialAPIResponse(resp, nil, client)
	if err != nil {
		if errors.Is(err, errCredentialConflict) {
			return handleCredentialNameConflict(d, m, credentialName)
		}
		return fmt.Errorf("failed to create credential: %w", err)
	}

	d.SetId(credentialName)

	log.Printf("[INFO] Credential created with name %s. Starting retry mechanism to read the credential...", credentialName)
	return retryCredentialRead(d, m, 5)
}

func handleCredentialNameConflict(d *schema.ResourceData, m interface{}, credentialName string) error {
	if !d.Get("adopt_existing").(bool) {
		return fmt.Errorf(
			"credential %q already exists on the proxy but is not in Terraform state. "+
				"Import it to manage it here:\n\n"+
				"  terraform import litellm_credential.<this resource's name in your config> %s\n\n"+
				"The next apply then updates it to match this configuration. To take it over during "+
				"create instead, set adopt_existing = true on this resource, which overwrites the "+
				"existing credential's values with the ones configured here",
			credentialName, shellSingleQuote(credentialName),
		)
	}

	log.Printf("[WARN] Credential %q already exists; adopt_existing is set, so taking it over and updating it to match configuration.", credentialName)
	d.SetId(credentialName)
	if err := patchCredential(m.(*Client), d, credentialName); err != nil {
		d.SetId("")
		return fmt.Errorf("failed to adopt existing credential %q: %w", credentialName, err)
	}
	return retryCredentialRead(d, m, 5)
}

func shellSingleQuote(s string) string {
	return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'"
}

func resourceLiteLLMCredentialRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)
	credentialName := d.Id()

	endpoint := fmt.Sprintf(endpointCredentialByName, url.PathEscape(credentialName))
	if modelID := d.Get("model_id").(string); modelID != "" {
		endpoint = fmt.Sprintf(endpointCredentialByNameForModel, url.PathEscape(credentialName), url.QueryEscape(modelID))
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

func patchCredential(client *Client, d *schema.ResourceData, credentialName string) error {
	resp, err := MakeRequest(client, "PATCH", fmt.Sprintf(endpointCredential, url.PathEscape(credentialName)), credentialRequestFromResource(d, credentialName))
	if err != nil {
		return fmt.Errorf("failed to update credential: %w", err)
	}
	defer resp.Body.Close()

	if err := handleCredentialAPIResponse(resp, nil, client); err != nil {
		return fmt.Errorf("failed to update credential: %w", err)
	}
	return nil
}

func resourceLiteLLMCredentialUpdate(d *schema.ResourceData, m interface{}) error {
	if !d.HasChangesExcept("adopt_existing") {
		return nil
	}

	credentialName := d.Id()
	if err := patchCredential(m.(*Client), d, credentialName); err != nil {
		return err
	}

	log.Printf("[INFO] Credential updated with name %s. Starting retry mechanism to read the credential...", credentialName)
	return retryCredentialRead(d, m, 5)
}

func resourceLiteLLMCredentialDelete(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)
	credentialName := d.Id()

	resp, err := MakeRequest(client, "DELETE", fmt.Sprintf(endpointCredential, url.PathEscape(credentialName)), nil)
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
