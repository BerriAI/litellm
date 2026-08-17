package litellm

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

const jwtKeyMappingNotFound = "jwt_key_mapping_not_found"

func resourceLiteLLMJWTKeyMappingCreate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	createRequest := JWTKeyMappingRequest{
		JWTClaimName:  d.Get("jwt_claim_name").(string),
		JWTClaimValue: d.Get("jwt_claim_value").(string),
		Key:           d.Get("key").(string),
		Description:   d.Get("description").(string),
	}

	resp, err := MakeRequest(client, "POST", "/jwt/key/mapping/new", createRequest)
	if err != nil {
		return fmt.Errorf("failed to create JWT key mapping: %w", err)
	}
	defer resp.Body.Close()

	var mapping JWTKeyMappingResponse
	if err := handleJWTKeyMappingAPIResponse(resp, &mapping, client); err != nil {
		return fmt.Errorf("failed to create JWT key mapping: %w", err)
	}

	if mapping.ID == "" {
		return fmt.Errorf("failed to create JWT key mapping: the proxy returned no mapping id")
	}

	d.SetId(mapping.ID)

	if !d.Get("is_active").(bool) {
		if err := updateJWTKeyMapping(d, client); err != nil {
			return fmt.Errorf("JWT key mapping %s was created but could not be deactivated: %w", mapping.ID, err)
		}
	}

	return resourceLiteLLMJWTKeyMappingRead(d, m)
}

func resourceLiteLLMJWTKeyMappingRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	resp, err := MakeRequest(client, "GET", fmt.Sprintf("/jwt/key/mapping/info?id=%s", url.QueryEscape(d.Id())), nil)
	if err != nil {
		return fmt.Errorf("failed to read JWT key mapping: %w", err)
	}
	defer resp.Body.Close()

	var mapping JWTKeyMappingResponse
	if err := handleJWTKeyMappingAPIResponse(resp, &mapping, client); err != nil {
		if err.Error() == jwtKeyMappingNotFound {
			d.SetId("")
			return nil
		}
		return fmt.Errorf("failed to read JWT key mapping: %w", err)
	}

	d.SetId(mapping.ID)
	d.Set("jwt_claim_name", mapping.JWTClaimName)
	d.Set("jwt_claim_value", mapping.JWTClaimValue)
	d.Set("description", mapping.Description)
	d.Set("is_active", mapping.IsActive)
	d.Set("created_at", mapping.CreatedAt)
	d.Set("updated_at", mapping.UpdatedAt)
	d.Set("created_by", mapping.CreatedBy)
	d.Set("updated_by", mapping.UpdatedBy)

	return nil
}

func resourceLiteLLMJWTKeyMappingUpdate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	oldKey, _ := d.GetChange("key")

	if err := updateJWTKeyMapping(d, client); err != nil {
		// The update is a single atomic API call: on failure nothing changed
		// server-side. Revert key explicitly since the proxy never returns it,
		// so Read can't resync it the way it resyncs description/is_active below.
		d.Set("key", oldKey)
		if readErr := resourceLiteLLMJWTKeyMappingRead(d, m); readErr != nil {
			return fmt.Errorf("failed to update JWT key mapping: %w (and failed to refresh state afterward: %v)", err, readErr)
		}
		return fmt.Errorf("failed to update JWT key mapping: %w", err)
	}

	return resourceLiteLLMJWTKeyMappingRead(d, m)
}

func resourceLiteLLMJWTKeyMappingDelete(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	resp, err := MakeRequest(client, "POST", "/jwt/key/mapping/delete", JWTKeyMappingDeleteRequest{ID: d.Id()})
	if err != nil {
		return fmt.Errorf("failed to delete JWT key mapping: %w", err)
	}
	defer resp.Body.Close()

	if err := handleJWTKeyMappingAPIResponse(resp, nil, client); err != nil {
		if err.Error() != jwtKeyMappingNotFound {
			return fmt.Errorf("failed to delete JWT key mapping: %w", err)
		}
	}

	d.SetId("")
	return nil
}

func updateJWTKeyMapping(d *schema.ResourceData, client *Client) error {
	updateRequest := JWTKeyMappingUpdateRequest{
		ID:          d.Id(),
		Key:         d.Get("key").(string),
		Description: d.Get("description").(string),
		IsActive:    d.Get("is_active").(bool),
	}

	resp, err := MakeRequest(client, "POST", "/jwt/key/mapping/update", updateRequest)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	return handleJWTKeyMappingAPIResponse(resp, nil, client)
}

func handleJWTKeyMappingAPIResponse(resp *http.Response, result interface{}, client *Client) error {
	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read response body: %v", err)
	}

	if resp.StatusCode == http.StatusNotFound {
		return fmt.Errorf(jwtKeyMappingNotFound)
	}

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		return fmt.Errorf("API request failed: Status: %s, Response: %s",
			resp.Status, client.redactSensitiveData(string(bodyBytes)))
	}

	if result == nil {
		return nil
	}

	if err := json.Unmarshal(bodyBytes, result); err != nil {
		return fmt.Errorf("failed to parse response: %v", err)
	}

	return nil
}
