package litellm

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

const endpointProjectList = "/project/list"

func dataSourceLiteLLMProject() *schema.Resource {
	return &schema.Resource{
		Read: dataSourceLiteLLMProjectRead,

		Schema: map[string]*schema.Schema{
			"project_id": {
				Type:        schema.TypeString,
				Required:    true,
				Description: "Unique identifier of the project to retrieve",
			},
			"project_alias": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Human-friendly name for the project",
			},
			"description": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Description of the project",
			},
			"team_id": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "The team ID this project belongs to",
			},
			"budget_id": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Budget ID associated with this project",
			},
			"models": {
				Type:        schema.TypeList,
				Computed:    true,
				Elem:        &schema.Schema{Type: schema.TypeString},
				Description: "List of models the project can access",
			},
			"max_budget": {
				Type:        schema.TypeFloat,
				Computed:    true,
				Description: "Maximum budget for this project",
			},
			"soft_budget": {
				Type:        schema.TypeFloat,
				Computed:    true,
				Description: "Soft budget limit for warnings",
			},
			"budget_duration": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Budget reset duration",
			},
			"tpm_limit": {
				Type:        schema.TypeInt,
				Computed:    true,
				Description: "Tokens per minute limit",
			},
			"rpm_limit": {
				Type:        schema.TypeInt,
				Computed:    true,
				Description: "Requests per minute limit",
			},
			"max_parallel_requests": {
				Type:        schema.TypeInt,
				Computed:    true,
				Description: "Maximum parallel requests allowed",
			},
			"blocked": {
				Type:        schema.TypeBool,
				Computed:    true,
				Description: "Whether the project is blocked from making requests",
			},
			"spend": {
				Type:        schema.TypeFloat,
				Computed:    true,
				Description: "Current spend for the project",
			},
			"created_at": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Timestamp when the project was created",
			},
			"updated_at": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Timestamp when the project was last updated",
			},
			"created_by": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "User that created the project",
			},
			"updated_by": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "User that last updated the project",
			},
		},
	}
}

func dataSourceLiteLLMProjectRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)
	projectID := d.Get("project_id").(string)

	resp, err := MakeRequest(client, "GET", fmt.Sprintf("%s?project_id=%s", endpointProjectInfo, projectID), nil)
	if err != nil {
		return fmt.Errorf("failed to read project: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return fmt.Errorf("project '%s' not found", projectID)
	}

	if err := handleResponse(resp, "reading project"); err != nil {
		return err
	}

	var projResp projectResponse
	if err := json.NewDecoder(resp.Body).Decode(&projResp); err != nil {
		return fmt.Errorf("error decoding project info response: %w", err)
	}

	d.SetId(projResp.ProjectID)
	d.Set("project_id", projResp.ProjectID)
	d.Set("project_alias", projResp.ProjectAlias)
	d.Set("description", projResp.Description)
	d.Set("team_id", projResp.TeamID)
	d.Set("budget_id", projResp.BudgetID)
	d.Set("models", projResp.Models)
	d.Set("blocked", projResp.Blocked)
	d.Set("spend", projResp.Spend)
	d.Set("created_at", projResp.CreatedAt)
	d.Set("updated_at", projResp.UpdatedAt)
	d.Set("created_by", projResp.CreatedBy)
	d.Set("updated_by", projResp.UpdatedBy)

	if bt := projResp.LitellmBudgetTable; bt != nil {
		if bt.MaxBudget != nil {
			d.Set("max_budget", *bt.MaxBudget)
		}
		if bt.SoftBudget != nil {
			d.Set("soft_budget", *bt.SoftBudget)
		}
		if bt.MaxParallelRequests != nil {
			d.Set("max_parallel_requests", *bt.MaxParallelRequests)
		}
		if bt.TPMLimit != nil {
			d.Set("tpm_limit", *bt.TPMLimit)
		}
		if bt.RPMLimit != nil {
			d.Set("rpm_limit", *bt.RPMLimit)
		}
		d.Set("budget_duration", bt.BudgetDuration)
	}

	return nil
}

func dataSourceLiteLLMProjects() *schema.Resource {
	return &schema.Resource{
		Read: dataSourceLiteLLMProjectsRead,

		Schema: map[string]*schema.Schema{
			"ids": {
				Type:        schema.TypeList,
				Computed:    true,
				Elem:        &schema.Schema{Type: schema.TypeString},
				Description: "IDs of all projects",
			},
			"projects": {
				Type:        schema.TypeList,
				Computed:    true,
				Description: "List of projects",
				Elem: &schema.Resource{
					Schema: map[string]*schema.Schema{
						"project_id":    {Type: schema.TypeString, Computed: true},
						"project_alias": {Type: schema.TypeString, Computed: true},
						"description":   {Type: schema.TypeString, Computed: true},
						"team_id":       {Type: schema.TypeString, Computed: true},
						"budget_id":     {Type: schema.TypeString, Computed: true},
						"models": {
							Type:     schema.TypeList,
							Computed: true,
							Elem:     &schema.Schema{Type: schema.TypeString},
						},
						"blocked":    {Type: schema.TypeBool, Computed: true},
						"spend":      {Type: schema.TypeFloat, Computed: true},
						"created_at": {Type: schema.TypeString, Computed: true},
						"updated_at": {Type: schema.TypeString, Computed: true},
						"created_by": {Type: schema.TypeString, Computed: true},
						"updated_by": {Type: schema.TypeString, Computed: true},
					},
				},
			},
		},
	}
}

func dataSourceLiteLLMProjectsRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	resp, err := MakeRequest(client, "GET", endpointProjectList, nil)
	if err != nil {
		return fmt.Errorf("failed to list projects: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "listing projects"); err != nil {
		return err
	}

	var projResps []projectResponse
	if err := json.NewDecoder(resp.Body).Decode(&projResps); err != nil {
		return fmt.Errorf("error decoding project list response: %w", err)
	}

	ids := make([]string, 0, len(projResps))
	projects := make([]map[string]interface{}, 0, len(projResps))
	for _, projResp := range projResps {
		ids = append(ids, projResp.ProjectID)
		projects = append(projects, map[string]interface{}{
			"project_id":    projResp.ProjectID,
			"project_alias": projResp.ProjectAlias,
			"description":   projResp.Description,
			"team_id":       projResp.TeamID,
			"budget_id":     projResp.BudgetID,
			"models":        projResp.Models,
			"blocked":       projResp.Blocked,
			"spend":         projResp.Spend,
			"created_at":    projResp.CreatedAt,
			"updated_at":    projResp.UpdatedAt,
			"created_by":    projResp.CreatedBy,
			"updated_by":    projResp.UpdatedBy,
		})
	}

	d.SetId("litellm-projects")
	d.Set("ids", ids)
	d.Set("projects", projects)

	return nil
}
