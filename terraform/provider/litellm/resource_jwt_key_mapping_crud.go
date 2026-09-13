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

	// The create endpoint has no is_active field and always activates the
	// mapping, so a JWT client matching this claim can authenticate during
	// the gap before the deactivation call below runs. If deactivation
	// itself fails, delete the mapping rather than leaving it active and
	// unmanaged indefinitely.
	if !d.Get("is_active").(bool) {
		if err := updateJWTKeyMapping(d, client); err != nil {
			if deleteErr := deleteJWTKeyMapping(mapping.ID, client); deleteErr != nil {
				return fmt.Errorf(
					"JWT key mapping %s was created active and could not be deactivated (%v); it also could not be deleted and remains active on the proxy, remove it manually via POST /jwt/key/mapping/delete: %v",
					mapping.ID, err, deleteErr,
				)
			}
			d.SetId("")
			return fmt.Errorf("JWT key mapping was created active but could not be deactivated, so it was deleted instead: %w", err)
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
	oldDescription, _ := d.GetChange("description")
	oldIsActive, _ := d.GetChange("is_active")

	if err := updateJWTKeyMapping(d, client); err != nil {
		// The update is a single atomic API call: on failure nothing changed
		// server-side. Revert every field the update could have changed before
		// attempting to resync, so a failed refresh can't leave the rejected
		// values persisted into state.
		d.Set("key", oldKey)
		d.Set("description", oldDescription)
		d.Set("is_active", oldIsActive)
		if readErr := resourceLiteLLMJWTKeyMappingRead(d, m); readErr != nil {
			return fmt.Errorf("failed to update JWT key mapping: %w (and failed to refresh state afterward: %v)", err, readErr)
		}
		return fmt.Errorf("failed to update JWT key mapping: %w", err)
	}

	return resourceLiteLLMJWTKeyMappingRead(d, m)
}

func resourceLiteLLMJWTKeyMappingDelete(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	if err := deleteJWTKeyMapping(d.Id(), client); err != nil {
		return fmt.Errorf("failed to delete JWT key mapping: %w", err)
	}

	d.SetId("")
	return nil
}

func deleteJWTKeyMapping(id string, client *Client) error {
	resp, err := MakeRequest(client, "POST", "/jwt/key/mapping/delete", JWTKeyMappingDeleteRequest{ID: id})
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if err := handleJWTKeyMappingAPIResponse(resp, nil, client); err != nil {
		if err.Error() != jwtKeyMappingNotFound {
			return err
		}
	}

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
