package litellm

import (
	"encoding/json"
	"fmt"
	"log"
	"net/url"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

const endpointOrganizationList = "/organization/list"

type organizationBudget struct {
	MaxBudget           *float64 `json:"max_budget"`
	SoftBudget          *float64 `json:"soft_budget"`
	TPMLimit            *int     `json:"tpm_limit"`
	RPMLimit            *int     `json:"rpm_limit"`
	MaxParallelRequests *int     `json:"max_parallel_requests"`
	BudgetDuration      string   `json:"budget_duration"`
}

type organizationDetail struct {
	OrganizationID    string                 `json:"organization_id"`
	OrganizationAlias string                 `json:"organization_alias"`
	BudgetID          string                 `json:"budget_id"`
	Models            []string               `json:"models"`
	Spend             float64                `json:"spend"`
	Metadata          map[string]interface{} `json:"metadata"`
	CreatedAt         string                 `json:"created_at"`
	UpdatedAt         string                 `json:"updated_at"`
	Budget            *organizationBudget    `json:"litellm_budget_table"`
}

func dataSourceLiteLLMOrganization() *schema.Resource {
	return &schema.Resource{
		Read: dataSourceLiteLLMOrganizationRead,

		Schema: map[string]*schema.Schema{
			"organization_id": {
				Type:        schema.TypeString,
				Required:    true,
				Description: "Unique identifier of the organization to retrieve",
			},
			"organization_alias": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"budget_id": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"models": {
				Type:     schema.TypeList,
				Computed: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
			"spend": {
				Type:     schema.TypeFloat,
				Computed: true,
			},
			"metadata": {
				Type:     schema.TypeMap,
				Computed: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
			"max_budget": {
				Type:     schema.TypeFloat,
				Computed: true,
			},
			"soft_budget": {
				Type:     schema.TypeFloat,
				Computed: true,
			},
			"tpm_limit": {
				Type:     schema.TypeInt,
				Computed: true,
			},
			"rpm_limit": {
				Type:     schema.TypeInt,
				Computed: true,
			},
			"max_parallel_requests": {
				Type:     schema.TypeInt,
				Computed: true,
			},
			"budget_duration": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"created_at": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"updated_at": {
				Type:     schema.TypeString,
				Computed: true,
			},
		},
	}
}

func dataSourceLiteLLMOrganizationRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)
	orgID := d.Get("organization_id").(string)

	endpoint := fmt.Sprintf("%s?organization_id=%s", endpointOrganizationInfo, url.QueryEscape(orgID))
	resp, err := MakeRequest(client, "GET", endpoint, nil)
	if err != nil {
		return fmt.Errorf("failed to read organization: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "reading organization info"); err != nil {
		return err
	}

	var org organizationDetail
	if err := json.NewDecoder(resp.Body).Decode(&org); err != nil {
		return fmt.Errorf("failed to decode organization info response: %w", err)
	}

	d.SetId(GetStringValue(org.OrganizationID, orgID))
	organizationSetDetail(d, org)

	log.Printf("[INFO] Successfully read organization with ID: %s", orgID)
	return nil
}

func organizationSetDetail(d *schema.ResourceData, org organizationDetail) {
	d.Set("organization_alias", org.OrganizationAlias)
	d.Set("budget_id", org.BudgetID)
	d.Set("models", org.Models)
	d.Set("spend", org.Spend)

	metadata := map[string]string{}
	for k, v := range org.Metadata {
		if s, ok := v.(string); ok {
			metadata[k] = s
		}
	}
	d.Set("metadata", metadata)

	if org.Budget != nil {
		if org.Budget.MaxBudget != nil {
			d.Set("max_budget", *org.Budget.MaxBudget)
		}
		if org.Budget.SoftBudget != nil {
			d.Set("soft_budget", *org.Budget.SoftBudget)
		}
		if org.Budget.TPMLimit != nil {
			d.Set("tpm_limit", *org.Budget.TPMLimit)
		}
		if org.Budget.RPMLimit != nil {
			d.Set("rpm_limit", *org.Budget.RPMLimit)
		}
		if org.Budget.MaxParallelRequests != nil {
			d.Set("max_parallel_requests", *org.Budget.MaxParallelRequests)
		}
		d.Set("budget_duration", org.Budget.BudgetDuration)
	}
	d.Set("created_at", org.CreatedAt)
	d.Set("updated_at", org.UpdatedAt)
}

func dataSourceLiteLLMOrganizations() *schema.Resource {
	return &schema.Resource{
		Read: dataSourceLiteLLMOrganizationsRead,

		Schema: map[string]*schema.Schema{
			"org_alias": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "Filter organizations by alias",
			},
			"ids": {
				Type:        schema.TypeList,
				Computed:    true,
				Elem:        &schema.Schema{Type: schema.TypeString},
				Description: "IDs of the returned organizations",
			},
			"organizations": {
				Type:     schema.TypeList,
				Computed: true,
				Elem: &schema.Resource{
					Schema: map[string]*schema.Schema{
						"organization_id":    {Type: schema.TypeString, Computed: true},
						"organization_alias": {Type: schema.TypeString, Computed: true},
						"budget_id":          {Type: schema.TypeString, Computed: true},
						"models":             {Type: schema.TypeList, Computed: true, Elem: &schema.Schema{Type: schema.TypeString}},
						"spend":              {Type: schema.TypeFloat, Computed: true},
						"max_budget":         {Type: schema.TypeFloat, Computed: true},
						"tpm_limit":          {Type: schema.TypeInt, Computed: true},
						"rpm_limit":          {Type: schema.TypeInt, Computed: true},
						"budget_duration":    {Type: schema.TypeString, Computed: true},
						"created_at":         {Type: schema.TypeString, Computed: true},
						"updated_at":         {Type: schema.TypeString, Computed: true},
					},
				},
			},
		},
	}
}

func dataSourceLiteLLMOrganizationsRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	endpoint := endpointOrganizationList
	if v, ok := d.GetOk("org_alias"); ok {
		endpoint = fmt.Sprintf("%s?org_alias=%s", endpointOrganizationList, url.QueryEscape(v.(string)))
	}

	resp, err := MakeRequest(client, "GET", endpoint, nil)
	if err != nil {
		return fmt.Errorf("failed to list organizations: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "listing organizations"); err != nil {
		return err
	}

	var orgList []organizationDetail
	if err := json.NewDecoder(resp.Body).Decode(&orgList); err != nil {
		return fmt.Errorf("failed to decode organization list response: %w", err)
	}

	ids := make([]string, 0, len(orgList))
	orgs := make([]map[string]interface{}, 0, len(orgList))
	for _, org := range orgList {
		ids = append(ids, org.OrganizationID)
		item := map[string]interface{}{
			"organization_id":    org.OrganizationID,
			"organization_alias": org.OrganizationAlias,
			"budget_id":          org.BudgetID,
			"models":             org.Models,
			"spend":              org.Spend,
			"created_at":         org.CreatedAt,
			"updated_at":         org.UpdatedAt,
		}
		if org.Budget != nil {
			item["max_budget"] = organizationDerefFloat(org.Budget.MaxBudget)
			item["tpm_limit"] = organizationDerefInt(org.Budget.TPMLimit)
			item["rpm_limit"] = organizationDerefInt(org.Budget.RPMLimit)
			item["budget_duration"] = org.Budget.BudgetDuration
		}
		orgs = append(orgs, item)
	}

	d.SetId(GetStringValue(d.Get("org_alias").(string), "all"))
	d.Set("ids", ids)
	d.Set("organizations", orgs)

	log.Printf("[INFO] Successfully listed %d organizations", len(orgs))
	return nil
}

func organizationDerefFloat(v *float64) float64 {
	if v == nil {
		return 0
	}
	return *v
}

func organizationDerefInt(v *int) int {
	if v == nil {
		return 0
	}
	return *v
}
