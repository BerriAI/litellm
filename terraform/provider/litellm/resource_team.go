package litellm

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"

	"github.com/google/uuid"
	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

const (
	endpointTeamNew               = "/team/new"
	endpointTeamInfo              = "/team/info"
	endpointTeamUpdate            = "/team/update"
	endpointTeamDelete            = "/team/delete"
	endpointTeamPermissionsList   = "/team/permissions_list"
	endpointTeamPermissionsUpdate = "/team/permissions_update"
)

func ResourceLiteLLMTeam() *schema.Resource {
	return &schema.Resource{
		Create: resourceLiteLLMTeamCreate,
		Read:   resourceLiteLLMTeamRead,
		Update: resourceLiteLLMTeamUpdate,
		Delete: resourceLiteLLMTeamDelete,

		Schema: map[string]*schema.Schema{
			"team_alias": {
				Type:     schema.TypeString,
				Required: true,
			},
			"organization_id": {
				Type:     schema.TypeString,
				Optional: true,
			},
			"metadata": {
				Type:     schema.TypeMap,
				Optional: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
			"tpm_limit": {
				Type:     schema.TypeInt,
				Optional: true,
			},
			"rpm_limit": {
				Type:     schema.TypeInt,
				Optional: true,
			},
			"max_budget": {
				Type:     schema.TypeFloat,
				Optional: true,
			},
			"soft_budget": {
				Type:        schema.TypeFloat,
				Optional:    true,
				Description: "Spend threshold that triggers a soft budget alert without blocking requests",
			},
			"budget_duration": {
				Type:     schema.TypeString,
				Optional: true,
			},
			"models": {
				Type:     schema.TypeList,
				Optional: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
			"blocked": {
				Type:     schema.TypeBool,
				Optional: true,
			},
			"team_member_permissions": {
				Type:        schema.TypeList,
				Optional:    true,
				Elem:        &schema.Schema{Type: schema.TypeString},
				Description: "List of permissions granted to team members",
			},
			"tags": {
				Type:        schema.TypeList,
				Optional:    true,
				Elem:        &schema.Schema{Type: schema.TypeString},
				Description: "Tags for spend tracking and tag-based routing",
			},
			"soft_budget_alerting_emails": {
				Type:        schema.TypeList,
				Optional:    true,
				Elem:        &schema.Schema{Type: schema.TypeString},
				Description: "Email addresses alerted when the team crosses soft_budget",
			},
		},
	}
}

func resourceLiteLLMTeamCreate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	teamID := uuid.New().String()
	teamData := buildTeamData(d, teamID)

	log.Printf("[DEBUG] Create team request payload: %+v", teamData)

	resp, err := MakeRequest(client, "POST", endpointTeamNew, teamData)
	if err != nil {
		return fmt.Errorf("error creating team: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "creating team"); err != nil {
		return err
	}

	d.SetId(teamID)
	log.Printf("[INFO] Team created with ID: %s", teamID)

	return resourceLiteLLMTeamRead(d, m)
}

func resourceLiteLLMTeamRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	log.Printf("[INFO] Reading team with ID: %s", d.Id())

	resp, err := MakeRequest(client, "GET", fmt.Sprintf("%s?team_id=%s", endpointTeamInfo, d.Id()), nil)
	if err != nil {
		return fmt.Errorf("error reading team: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		log.Printf("[WARN] Team with ID %s not found, removing from state", d.Id())
		d.SetId("")
		return nil
	}

	var infoResp TeamInfoResponse
	if err := json.NewDecoder(resp.Body).Decode(&infoResp); err != nil {
		return fmt.Errorf("error decoding team info response: %w", err)
	}
	teamResp := infoResp.TeamInfo

	// Update the state with values from the response or fall back to the data passed in during creation
	d.Set("team_alias", GetStringValue(teamResp.TeamAlias, d.Get("team_alias").(string)))
	d.Set("organization_id", GetStringValue(teamResp.OrganizationID, d.Get("organization_id").(string)))

	metadata, tags, alertEmails := splitTeamMetadata(teamResp.Metadata)
	d.Set("metadata", metadata)
	d.Set("tags", tags)
	d.Set("soft_budget_alerting_emails", alertEmails)

	if teamResp.TPMLimit != nil {
		d.Set("tpm_limit", *teamResp.TPMLimit)
	}
	if teamResp.RPMLimit != nil {
		d.Set("rpm_limit", *teamResp.RPMLimit)
	}
	if teamResp.MaxBudget != nil {
		d.Set("max_budget", *teamResp.MaxBudget)
	}
	d.Set("soft_budget", teamResp.SoftBudget)
	d.Set("budget_duration", GetStringValue(teamResp.BudgetDuration, d.Get("budget_duration").(string)))

	// Handle models separately as it's a list
	if teamResp.Models != nil {
		d.Set("models", teamResp.Models)
	} else {
		d.Set("models", d.Get("models"))
	}

	d.Set("blocked", GetBoolValue(teamResp.Blocked, d.Get("blocked").(bool)))

	// Explicitly fetch the current permissions from the API
	permResp, err := getTeamPermissions(client, d.Id())
	if err != nil {
		log.Printf("[WARN] Error fetching team permissions: %s", err)
		// Fall back to the permissions from the team info response
		if teamResp.TeamMemberPermissions != nil {
			d.Set("team_member_permissions", teamResp.TeamMemberPermissions)
		} else {
			d.Set("team_member_permissions", d.Get("team_member_permissions"))
		}
	} else {
		// Use the permissions from the permissions_list endpoint
		log.Printf("[DEBUG] Team permissions from API: %+v", permResp.TeamMemberPermissions)
		d.Set("team_member_permissions", permResp.TeamMemberPermissions)
	}

	log.Printf("[INFO] Successfully read team with ID: %s", d.Id())
	return nil
}

func resourceLiteLLMTeamUpdate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	teamData := buildTeamData(d, d.Id())
	log.Printf("[DEBUG] Update team request payload: %+v", teamData)

	resp, err := MakeRequest(client, "POST", endpointTeamUpdate, teamData)
	if err != nil {
		return fmt.Errorf("error updating team: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "updating team"); err != nil {
		return err
	}

	// Check if team_member_permissions have changed and explicitly update them
	if d.HasChange("team_member_permissions") {
		_, newPerms := d.GetChange("team_member_permissions")
		if newPerms != nil {
			// Convert interface{} to []string
			var permissions []string
			for _, perm := range newPerms.([]interface{}) {
				permissions = append(permissions, perm.(string))
			}

			log.Printf("[DEBUG] Explicitly updating team permissions: %+v", permissions)
			if err := updateTeamPermissions(client, d.Id(), permissions); err != nil {
				return fmt.Errorf("error updating team permissions: %w", err)
			}
		}
	}

	log.Printf("[INFO] Successfully updated team with ID: %s", d.Id())
	return resourceLiteLLMTeamRead(d, m)
}

func resourceLiteLLMTeamDelete(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	log.Printf("[INFO] Deleting team with ID: %s", d.Id())

	deleteData := map[string]interface{}{
		"team_ids": []string{d.Id()},
	}

	resp, err := MakeRequest(client, "POST", endpointTeamDelete, deleteData)
	if err != nil {
		return fmt.Errorf("error deleting team: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "deleting team"); err != nil {
		return err
	}

	log.Printf("[INFO] Successfully deleted team with ID: %s", d.Id())
	d.SetId("")
	return nil
}

func buildTeamData(d *schema.ResourceData, teamID string) map[string]interface{} {
	teamData := map[string]interface{}{
		"team_id":    teamID,
		"team_alias": d.Get("team_alias").(string),
	}

	for _, key := range []string{"organization_id", "tpm_limit", "rpm_limit", "max_budget", "budget_duration", "models", "blocked", "team_member_permissions"} {
		if v, ok := d.GetOk(key); ok {
			teamData[key] = v
		}
	}

	if v, ok := d.GetOk("soft_budget"); ok {
		teamData["soft_budget"] = v
	} else if d.HasChange("soft_budget") {
		teamData["soft_budget"] = nil
	}

	if v, ok := d.GetOk("tags"); ok || d.HasChange("tags") {
		teamData["tags"] = v
	}

	if metadata := buildTeamMetadata(d); metadata != nil {
		teamData["metadata"] = metadata
	}

	return teamData
}

// /team/update replaces metadata wholesale, so the full map must go out whenever either half changed.
func buildTeamMetadata(d *schema.ResourceData) map[string]interface{} {
	metadata := map[string]interface{}{}
	for k, v := range d.Get("metadata").(map[string]interface{}) {
		metadata[k] = v
	}
	if v, ok := d.GetOk("soft_budget_alerting_emails"); ok {
		metadata["soft_budget_alerting_emails"] = v
	}
	if len(metadata) == 0 && !d.HasChange("metadata") && !d.HasChange("soft_budget_alerting_emails") {
		return nil
	}
	return metadata
}

func splitTeamMetadata(raw map[string]interface{}) (map[string]string, []string, []string) {
	metadata := map[string]string{}
	var tags, alertEmails []string
	for k, v := range raw {
		switch k {
		case "tags":
			tags = toStringSlice(v)
		case "soft_budget_alerting_emails":
			alertEmails = toStringSlice(v)
		case "team_member_budget_id":
		default:
			if s, ok := v.(string); ok {
				metadata[k] = s
			}
		}
	}
	return metadata, tags, alertEmails
}

func toStringSlice(v interface{}) []string {
	items, ok := v.([]interface{})
	if !ok {
		return nil
	}
	out := make([]string, 0, len(items))
	for _, item := range items {
		if s, ok := item.(string); ok {
			out = append(out, s)
		}
	}
	return out
}

func handleResponse(resp *http.Response, action string) error {
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("error %s: %s - %s", action, resp.Status, string(body))
	}
	return nil
}

// TeamPermissionsResponse represents a response from the API containing team permissions information.
type TeamPermissionsResponse struct {
	TeamID                  string   `json:"team_id"`
	TeamMemberPermissions   []string `json:"team_member_permissions"`
	AllAvailablePermissions []string `json:"all_available_permissions"`
}

// getTeamPermissions retrieves the current permissions and available permissions for a team.
func getTeamPermissions(client *Client, teamID string) (*TeamPermissionsResponse, error) {
	log.Printf("[INFO] Getting permissions for team with ID: %s", teamID)

	resp, err := MakeRequest(client, "GET", fmt.Sprintf("%s?team_id=%s", endpointTeamPermissionsList, teamID), nil)
	if err != nil {
		return nil, fmt.Errorf("error getting team permissions: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("error getting team permissions: %s - %s", resp.Status, string(body))
	}

	var permResp TeamPermissionsResponse
	if err := json.NewDecoder(resp.Body).Decode(&permResp); err != nil {
		return nil, fmt.Errorf("error decoding team permissions response: %w", err)
	}

	return &permResp, nil
}

// updateTeamPermissions updates the permissions for a team.
func updateTeamPermissions(client *Client, teamID string, permissions []string) error {
	log.Printf("[INFO] Updating permissions for team with ID: %s", teamID)

	permData := map[string]interface{}{
		"team_id":                 teamID,
		"team_member_permissions": permissions,
	}

	resp, err := MakeRequest(client, "POST", endpointTeamPermissionsUpdate, permData)
	if err != nil {
		return fmt.Errorf("error updating team permissions: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "updating team permissions"); err != nil {
		return err
	}

	log.Printf("[INFO] Successfully updated permissions for team with ID: %s", teamID)
	return nil
}
