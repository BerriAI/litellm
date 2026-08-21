package litellm

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"

	"github.com/google/uuid"
	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

const (
	endpointOrganizationNew    = "/organization/new"
	endpointOrganizationInfo   = "/organization/info"
	endpointOrganizationUpdate = "/organization/update"
	endpointOrganizationDelete = "/organization/delete"
)

func resourceLiteLLMOrganization() *schema.Resource {
	return &schema.Resource{
		Create: resourceLiteLLMOrganizationCreate,
		Read:   resourceLiteLLMOrganizationRead,
		Update: resourceLiteLLMOrganizationUpdate,
		Delete: resourceLiteLLMOrganizationDelete,

		Schema: map[string]*schema.Schema{
			"organization_alias": {
				Type:     schema.TypeString,
				Required: true,
			},
			"metadata": {
				Type:     schema.TypeMap,
				Optional: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
			"models": {
				Type:     schema.TypeList,
				Optional: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
			"max_budget": {
				Type:     schema.TypeFloat,
				Optional: true,
			},
			"budget_duration": {
				Type:     schema.TypeString,
				Optional: true,
			},
			"tpm_limit": {
				Type:     schema.TypeInt,
				Optional: true,
			},
			"rpm_limit": {
				Type:     schema.TypeInt,
				Optional: true,
			},
			"blocked": {
				Type:     schema.TypeBool,
				Optional: true,
			},
		},
	}
}

func resourceLiteLLMOrganizationCreate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	orgID := uuid.New().String()
	orgData := buildOrganizationData(d, orgID)

	log.Printf("[DEBUG] Create organization request payload: %+v", orgData)

	resp, err := MakeRequest(client, "POST", endpointOrganizationNew, orgData)
	if err != nil {
		return fmt.Errorf("error creating organization: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "creating organization"); err != nil {
		return err
	}

	d.SetId(orgID)
	log.Printf("[INFO] Organization created with ID: %s", orgID)

	return resourceLiteLLMOrganizationRead(d, m)
}

func resourceLiteLLMOrganizationRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	log.Printf("[INFO] Reading organization with ID: %s", d.Id())

	resp, err := MakeRequest(client, "POST", endpointOrganizationInfo, map[string]interface{}{
		"organizations": []string{d.Id()},
	})
	if err != nil {
		return fmt.Errorf("error reading organization: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		log.Printf("[WARN] Organization with ID %s not found, removing from state", d.Id())
		d.SetId("")
		return nil
	}

	var orgResps []OrganizationResponse
	if err := json.NewDecoder(resp.Body).Decode(&orgResps); err != nil {
		return fmt.Errorf("error decoding organization info response: %w", err)
	}

	if len(orgResps) == 0 {
		log.Printf("[WARN] Organization with ID %s not found in response, removing from state", d.Id())
		d.SetId("")
		return nil
	}

	orgResp := orgResps[0]

	d.Set("organization_alias", GetStringValue(orgResp.OrganizationAlias, d.Get("organization_alias").(string)))

	if orgResp.Metadata != nil {
		d.Set("metadata", orgResp.Metadata)
	} else {
		d.Set("metadata", d.Get("metadata"))
	}

	if orgResp.Models != nil {
		d.Set("models", orgResp.Models)
	} else {
		d.Set("models", d.Get("models"))
	}

	if orgResp.MaxBudget != nil {
		d.Set("max_budget", *orgResp.MaxBudget)
	}
	d.Set("budget_duration", GetStringValue(orgResp.BudgetDuration, d.Get("budget_duration").(string)))
	if orgResp.TPMLimit != nil {
		d.Set("tpm_limit", *orgResp.TPMLimit)
	}
	if orgResp.RPMLimit != nil {
		d.Set("rpm_limit", *orgResp.RPMLimit)
	}
	d.Set("blocked", GetBoolValue(orgResp.Blocked, d.Get("blocked").(bool)))

	log.Printf("[INFO] Successfully read organization with ID: %s", d.Id())
	return nil
}

func resourceLiteLLMOrganizationUpdate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	orgData := buildOrganizationData(d, d.Id())
	log.Printf("[DEBUG] Update organization request payload: %+v", orgData)

	resp, err := MakeRequest(client, "PATCH", endpointOrganizationUpdate, orgData)
	if err != nil {
		return fmt.Errorf("error updating organization: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "updating organization"); err != nil {
		return err
	}

	log.Printf("[INFO] Successfully updated organization with ID: %s", d.Id())
	return resourceLiteLLMOrganizationRead(d, m)
}

func resourceLiteLLMOrganizationDelete(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	log.Printf("[INFO] Deleting organization with ID: %s", d.Id())

	deleteData := map[string]interface{}{
		"organization_ids": []string{d.Id()},
	}

	resp, err := MakeRequest(client, "DELETE", endpointOrganizationDelete, deleteData)

	if err != nil {
		return fmt.Errorf("error deleting organization: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "deleting organization"); err != nil {
		return err
	}

	log.Printf("[INFO] Successfully deleted organization with ID: %s", d.Id())
	d.SetId("")
	return nil
}

func buildOrganizationData(d *schema.ResourceData, orgID string) map[string]interface{} {
	orgData := map[string]interface{}{
		"organization_id":    orgID,
		"organization_alias": d.Get("organization_alias").(string),
	}

	for _, key := range []string{"metadata", "models", "max_budget", "budget_duration", "tpm_limit", "rpm_limit", "blocked"} {
		if v, ok := d.GetOk(key); ok {
			orgData[key] = v
		}
	}

	return orgData
}
