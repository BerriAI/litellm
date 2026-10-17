package litellm

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"strings"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/validation"
)

func resourceLiteLLMTeamMember() *schema.Resource {
	return &schema.Resource{
		Create: resourceLiteLLMTeamMemberCreate,
		Read:   resourceLiteLLMTeamMemberRead,
		Update: resourceLiteLLMTeamMemberUpdate,
		Delete: resourceLiteLLMTeamMemberDelete,

		Schema: map[string]*schema.Schema{
			"team_id": {
				Type:     schema.TypeString,
				Required: true,
			},
			"user_id": {
				Type:     schema.TypeString,
				Required: true,
			},
			"user_email": {
				Type:     schema.TypeString,
				Required: true,
			},
			"role": {
				Type:     schema.TypeString,
				Required: true,
				ValidateFunc: validation.StringInSlice([]string{
					"org_admin",
					"internal_user",
					"internal_user_viewer",
					"admin",
					"user",
				}, false),
			},
			"max_budget_in_team": {
				Type:     schema.TypeFloat,
				Optional: true,
			},
		},
	}
}

func resourceLiteLLMTeamMemberCreate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	memberData := map[string]interface{}{
		"member": []map[string]interface{}{
			{
				"role":       d.Get("role").(string),
				"user_id":    d.Get("user_id").(string),
				"user_email": d.Get("user_email").(string),
			},
		},
		"team_id":            d.Get("team_id").(string),
		"max_budget_in_team": d.Get("max_budget_in_team").(float64),
	}

	log.Printf("[DEBUG] Create team member request payload: %+v", memberData)

	resp, err := MakeRequest(client, "POST", "/team/member_add", memberData)
	if err != nil {
		return fmt.Errorf("error creating team member: %v", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "creating team member"); err != nil {
		return err
	}

	// Set a composite ID since there's no specific member ID returned
	d.SetId(fmt.Sprintf("%s:%s", d.Get("team_id").(string), d.Get("user_id").(string)))

	log.Printf("[INFO] Team member created with ID: %s", d.Id())

	return resourceLiteLLMTeamMemberRead(d, m)
}

type teamMemberRoleRow struct {
	UserID    string `json:"user_id"`
	UserEmail string `json:"user_email"`
	Role      string `json:"role"`
}

type teamMemberBudgetRow struct {
	MaxBudget *float64 `json:"max_budget"`
}

type teamMemberMembershipRow struct {
	UserID             string               `json:"user_id"`
	LiteLLMBudgetTable *teamMemberBudgetRow `json:"litellm_budget_table"`
}

// teamMemberInfoAPI is the subset of GET /team/info used to refresh a single member.
type teamMemberInfoAPI struct {
	TeamInfo struct {
		MembersWithRoles []teamMemberRoleRow `json:"members_with_roles"`
	} `json:"team_info"`
	TeamMemberships []teamMemberMembershipRow `json:"team_memberships"`
}

func resourceLiteLLMTeamMemberRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)
	teamID := d.Get("team_id").(string)
	userID := d.Get("user_id").(string)
	if teamID == "" || userID == "" {
		// Fall back to the composite id "team_id:user_id" used on create/import.
		if parts := strings.SplitN(d.Id(), ":", 2); len(parts) == 2 {
			if teamID == "" {
				teamID = parts[0]
			}
			if userID == "" {
				userID = parts[1]
			}
		}
	}

	log.Printf("[INFO] Reading team member team_id=%s user_id=%s", teamID, userID)

	resp, err := MakeRequest(client, "GET", fmt.Sprintf("/team/info?team_id=%s", url.QueryEscape(teamID)), nil)
	if err != nil {
		return fmt.Errorf("error reading team info for member: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		log.Printf("[WARN] Team %s not found, removing team member from state", teamID)
		d.SetId("")
		return nil
	}
	if err := handleResponse(resp, "reading team info for member"); err != nil {
		return err
	}

	var info teamMemberInfoAPI
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return fmt.Errorf("error decoding team info response: %w", err)
	}

	var found *teamMemberRoleRow
	for i := range info.TeamInfo.MembersWithRoles {
		row := &info.TeamInfo.MembersWithRoles[i]
		if row.UserID == userID {
			found = row
			break
		}
	}
	if found == nil {
		log.Printf("[WARN] User %s not in team %s, removing team member from state", userID, teamID)
		d.SetId("")
		return nil
	}

	d.Set("team_id", teamID)
	d.Set("user_id", found.UserID)
	if found.UserEmail != "" {
		d.Set("user_email", found.UserEmail)
	}
	if found.Role != "" {
		d.Set("role", found.Role)
	}

	for _, membership := range info.TeamMemberships {
		if membership.UserID != userID {
			continue
		}
		if membership.LiteLLMBudgetTable != nil && membership.LiteLLMBudgetTable.MaxBudget != nil {
			d.Set("max_budget_in_team", *membership.LiteLLMBudgetTable.MaxBudget)
		}
		break
	}

	d.SetId(fmt.Sprintf("%s:%s", teamID, userID))
	return nil
}

func resourceLiteLLMTeamMemberUpdate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	updateData := map[string]interface{}{
		"user_id":            d.Get("user_id").(string),
		"user_email":         d.Get("user_email").(string),
		"team_id":            d.Get("team_id").(string),
		"role":               d.Get("role").(string),
		"max_budget_in_team": d.Get("max_budget_in_team").(float64),
	}

	log.Printf("[DEBUG] Update team member request payload: %+v", updateData)

	resp, err := MakeRequest(client, "POST", "/team/member_update", updateData)
	if err != nil {
		return fmt.Errorf("error updating team member: %v", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "updating team member"); err != nil {
		return err
	}

	log.Printf("[INFO] Successfully updated team member with ID: %s", d.Id())

	return resourceLiteLLMTeamMemberRead(d, m)
}

func resourceLiteLLMTeamMemberDelete(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	deleteData := map[string]interface{}{
		"user_id":    d.Get("user_id").(string),
		"user_email": d.Get("user_email").(string),
		"team_id":    d.Get("team_id").(string),
	}

	log.Printf("[DEBUG] Delete team member request payload: %+v", deleteData)

	resp, err := MakeRequest(client, "POST", "/team/member_delete", deleteData)
	if err != nil {
		return fmt.Errorf("error deleting team member: %v", err)
	}
	defer resp.Body.Close()

	// Member or team already gone outside Terraform: treat as destroyed so state can converge.
	if resp.StatusCode == http.StatusNotFound || resp.StatusCode == http.StatusBadRequest {
		body, _ := io.ReadAll(resp.Body)
		bodyStr := string(body)
		if resp.StatusCode == http.StatusNotFound ||
			strings.Contains(bodyStr, "User not found in team") ||
			strings.Contains(bodyStr, "does not exist") ||
			strings.Contains(bodyStr, "Team not found") {
			log.Printf("[WARN] Team member already gone (%s): %s", resp.Status, bodyStr)
			d.SetId("")
			return nil
		}
		return fmt.Errorf("error deleting team member: %s - %s", resp.Status, bodyStr)
	}

	if err := handleResponse(resp, "deleting team member"); err != nil {
		return err
	}

	log.Printf("[INFO] Successfully deleted team member with ID: %s", d.Id())

	d.SetId("")
	return nil
}
