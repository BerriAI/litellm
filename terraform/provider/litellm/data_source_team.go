package litellm

import (
	"encoding/json"
	"fmt"
	"log"
	"net/url"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

const endpointTeamList = "/team/list"

type teamDetail struct {
	TeamID                string                 `json:"team_id"`
	TeamAlias             string                 `json:"team_alias"`
	OrganizationID        string                 `json:"organization_id"`
	Models                []string               `json:"models"`
	Metadata              map[string]interface{} `json:"metadata"`
	TPMLimit              *int                   `json:"tpm_limit"`
	RPMLimit              *int                   `json:"rpm_limit"`
	MaxParallelRequests   *int                   `json:"max_parallel_requests"`
	MaxBudget             *float64               `json:"max_budget"`
	SoftBudget            *float64               `json:"soft_budget"`
	Spend                 *float64               `json:"spend"`
	BudgetDuration        string                 `json:"budget_duration"`
	Blocked               bool                   `json:"blocked"`
	TeamMemberPermissions []string               `json:"team_member_permissions"`
	CreatedAt             string                 `json:"created_at"`
	UpdatedAt             string                 `json:"updated_at"`
}

type teamInfoEnvelope struct {
	TeamID   string     `json:"team_id"`
	TeamInfo teamDetail `json:"team_info"`
}

func dataSourceLiteLLMTeam() *schema.Resource {
	return &schema.Resource{
		Read: dataSourceLiteLLMTeamRead,

		Schema: map[string]*schema.Schema{
			"team_id": {
				Type:        schema.TypeString,
				Required:    true,
				Description: "Unique identifier of the team to retrieve",
			},
			"team_alias": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"organization_id": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"models": {
				Type:     schema.TypeList,
				Computed: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
			"metadata": {
				Type:     schema.TypeMap,
				Computed: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
			"tags": {
				Type:     schema.TypeList,
				Computed: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
			"soft_budget_alerting_emails": {
				Type:     schema.TypeList,
				Computed: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
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
			"max_budget": {
				Type:     schema.TypeFloat,
				Computed: true,
			},
			"soft_budget": {
				Type:     schema.TypeFloat,
				Computed: true,
			},
			"spend": {
				Type:     schema.TypeFloat,
				Computed: true,
			},
			"budget_duration": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"blocked": {
				Type:     schema.TypeBool,
				Computed: true,
			},
			"team_member_permissions": {
				Type:     schema.TypeList,
				Computed: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
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

func dataSourceLiteLLMTeamRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)
	teamID := d.Get("team_id").(string)

	resp, err := MakeRequest(client, "GET", fmt.Sprintf("%s?team_id=%s", endpointTeamInfo, url.QueryEscape(teamID)), nil)
	if err != nil {
		return fmt.Errorf("failed to read team: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "reading team info"); err != nil {
		return err
	}

	var envelope teamInfoEnvelope
	if err := json.NewDecoder(resp.Body).Decode(&envelope); err != nil {
		return fmt.Errorf("failed to decode team info response: %w", err)
	}
	team := envelope.TeamInfo

	d.SetId(teamID)
	d.Set("team_alias", team.TeamAlias)
	d.Set("organization_id", team.OrganizationID)
	d.Set("models", team.Models)

	metadata, tags, alertEmails := splitTeamMetadata(team.Metadata)
	d.Set("metadata", metadata)
	d.Set("tags", tags)
	d.Set("soft_budget_alerting_emails", alertEmails)

	if team.TPMLimit != nil {
		d.Set("tpm_limit", *team.TPMLimit)
	}
	if team.RPMLimit != nil {
		d.Set("rpm_limit", *team.RPMLimit)
	}
	if team.MaxParallelRequests != nil {
		d.Set("max_parallel_requests", *team.MaxParallelRequests)
	}
	if team.MaxBudget != nil {
		d.Set("max_budget", *team.MaxBudget)
	}
	if team.SoftBudget != nil {
		d.Set("soft_budget", *team.SoftBudget)
	}
	if team.Spend != nil {
		d.Set("spend", *team.Spend)
	}
	d.Set("budget_duration", team.BudgetDuration)
	d.Set("blocked", team.Blocked)
	d.Set("team_member_permissions", team.TeamMemberPermissions)
	d.Set("created_at", team.CreatedAt)
	d.Set("updated_at", team.UpdatedAt)

	log.Printf("[INFO] Successfully read team with ID: %s", teamID)
	return nil
}

func dataSourceLiteLLMTeams() *schema.Resource {
	return &schema.Resource{
		Read: dataSourceLiteLLMTeamsRead,

		Schema: map[string]*schema.Schema{
			"user_id": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "Only return teams this user belongs to",
			},
			"organization_id": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "Only return teams in this organization",
			},
			"ids": {
				Type:        schema.TypeList,
				Computed:    true,
				Elem:        &schema.Schema{Type: schema.TypeString},
				Description: "IDs of the returned teams",
			},
			"teams": {
				Type:     schema.TypeList,
				Computed: true,
				Elem: &schema.Resource{
					Schema: map[string]*schema.Schema{
						"team_id":         {Type: schema.TypeString, Computed: true},
						"team_alias":      {Type: schema.TypeString, Computed: true},
						"organization_id": {Type: schema.TypeString, Computed: true},
						"models":          {Type: schema.TypeList, Computed: true, Elem: &schema.Schema{Type: schema.TypeString}},
						"spend":           {Type: schema.TypeFloat, Computed: true},
						"max_budget":      {Type: schema.TypeFloat, Computed: true},
						"tpm_limit":       {Type: schema.TypeInt, Computed: true},
						"rpm_limit":       {Type: schema.TypeInt, Computed: true},
						"budget_duration": {Type: schema.TypeString, Computed: true},
						"blocked":         {Type: schema.TypeBool, Computed: true},
						"created_at":      {Type: schema.TypeString, Computed: true},
						"updated_at":      {Type: schema.TypeString, Computed: true},
					},
				},
			},
		},
	}
}

func dataSourceLiteLLMTeamsRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	query := url.Values{}
	if v, ok := d.GetOk("user_id"); ok {
		query.Set("user_id", v.(string))
	}
	if v, ok := d.GetOk("organization_id"); ok {
		query.Set("organization_id", v.(string))
	}

	resp, err := MakeRequest(client, "GET", fmt.Sprintf("%s?%s", endpointTeamList, query.Encode()), nil)
	if err != nil {
		return fmt.Errorf("failed to list teams: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "listing teams"); err != nil {
		return err
	}

	var teamList []teamDetail
	if err := json.NewDecoder(resp.Body).Decode(&teamList); err != nil {
		return fmt.Errorf("failed to decode team list response: %w", err)
	}

	ids := make([]string, 0, len(teamList))
	teams := make([]map[string]interface{}, 0, len(teamList))
	for _, team := range teamList {
		ids = append(ids, team.TeamID)
		teams = append(teams, map[string]interface{}{
			"team_id":         team.TeamID,
			"team_alias":      team.TeamAlias,
			"organization_id": team.OrganizationID,
			"models":          team.Models,
			"spend":           teamDerefFloat(team.Spend),
			"max_budget":      teamDerefFloat(team.MaxBudget),
			"tpm_limit":       teamDerefInt(team.TPMLimit),
			"rpm_limit":       teamDerefInt(team.RPMLimit),
			"budget_duration": team.BudgetDuration,
			"blocked":         team.Blocked,
			"created_at":      team.CreatedAt,
			"updated_at":      team.UpdatedAt,
		})
	}

	d.SetId(GetStringValue(query.Encode(), "all"))
	d.Set("ids", ids)
	d.Set("teams", teams)

	log.Printf("[INFO] Successfully listed %d teams", len(teams))
	return nil
}

func teamDerefFloat(v *float64) float64 {
	if v == nil {
		return 0
	}
	return *v
}

func teamDerefInt(v *int) int {
	if v == nil {
		return 0
	}
	return *v
}
