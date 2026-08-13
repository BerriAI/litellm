package litellm

import (
	"fmt"
	"net/http"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func dataSourceLiteLLMCredential() *schema.Resource {
	return &schema.Resource{
		Read: dataSourceLiteLLMCredentialRead,

		Schema: map[string]*schema.Schema{
			"credential_name": {
				Type:        schema.TypeString,
				Required:    true,
				Description: "Name of the credential to retrieve",
			},
			"model_id": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "Model ID associated with this credential",
			},
			"credential_info": {
				Type:        schema.TypeMap,
				Computed:    true,
				Elem:        &schema.Schema{Type: schema.TypeString},
				Description: "Additional information about the credential",
			},
			// Note: credential_values are not exposed in data sources for security reasons
		},
	}
}

func dataSourceLiteLLMCredentialRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)
	credentialName := d.Get("credential_name").(string)
	modelID := d.Get("model_id").(string)

	// Use the same endpoint as the resource read operation
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
		return fmt.Errorf("credential '%s' not found", credentialName)
	}

	var credentialResp CredentialResponse
	err = handleCredentialAPIResponse(resp, &credentialResp, client)
	if err != nil {
		if err.Error() == "credential_not_found" {
			return fmt.Errorf("credential '%s' not found", credentialName)
		}
		return fmt.Errorf("failed to read credential: %w", err)
	}

	// Set the data source ID to the credential name
	d.SetId(credentialResp.CredentialName)
	d.Set("credential_name", credentialResp.CredentialName)
	d.Set("credential_info", credentialResp.CredentialInfo)
	// Note: We don't expose credential_values in data sources for security reasons

	return nil
}
