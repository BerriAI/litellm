package litellm

import (
	"fmt"
	"log"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/validation"
)

func resourceLiteLLMTeamMemberAdd() *schema.Resource {
	return &schema.Resource{
		Create: resourceLiteLLMTeamMemberAddCreate,
		Read:   resourceLiteLLMTeamMemberAddRead,
		Update: resourceLiteLLMTeamMemberAddUpdate,
		Delete: resourceLiteLLMTeamMemberAddDelete,

		Schema: map[string]*schema.Schema{
			"team_id": {
				Type:     schema.TypeString,
				Required: true,
				ForceNew: true,
			},
			"member": {
				Type:     schema.TypeSet,
				Required: true,
				Elem: &schema.Resource{
					Schema: map[string]*schema.Schema{
						"user_id": {
							Type:     schema.TypeString,
							Optional: true,
						},
						"user_email": {
							Type:     schema.TypeString,
							Optional: true,
						},
						"role": {
							Type:     schema.TypeString,
							Required: true,
							ValidateFunc: validation.StringInSlice([]string{
								"admin",
								"user",
							}, false),
						},
					},
				},
			},
			"max_budget_in_team": {
				Type:     schema.TypeFloat,
				Optional: true,
			},
		},
	}
}

func resourceLiteLLMTeamMemberAddCreate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	teamID := d.Get("team_id").(string)
	members := d.Get("member").(*schema.Set)
	maxBudget := d.Get("max_budget_in_team").(float64)

	// Convert members to the expected format
	membersList := make([]map[string]interface{}, 0, members.Len())
	for _, member := range members.List() {
		m := member.(map[string]interface{})
		memberData := map[string]interface{}{
			"role": m["role"].(string),
		}
		if userID, ok := m["user_id"].(string); ok && userID != "" {
			memberData["user_id"] = userID
		}
		if userEmail, ok := m["user_email"].(string); ok && userEmail != "" {
			memberData["user_email"] = userEmail
		}
		membersList = append(membersList, memberData)
	}

	memberData := map[string]interface{}{
		"member":             membersList,
		"team_id":            teamID,
		"max_budget_in_team": maxBudget,
	}

	log.Printf("[DEBUG] Create team members request payload: %+v", memberData)

	resp, err := MakeRequest(client, "POST", "/team/member_add", memberData)
	if err != nil {
		return fmt.Errorf("error adding team members: %v", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "adding team members"); err != nil {
		return err
	}

	// Set ID as team_id since this resource manages all members for a team
	d.SetId(teamID)

	return resourceLiteLLMTeamMemberAddRead(d, m)
}

func resourceLiteLLMTeamMemberAddRead(d *schema.ResourceData, m interface{}) error {
	// The API doesn't provide a way to read specific team members
	// We'll maintain the state as is
	return nil
}

func resourceLiteLLMTeamMemberAddUpdate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)
	teamID := d.Get("team_id").(string)
	maxBudget := d.Get("max_budget_in_team").(float64)

	o, n := d.GetChange("member")
	oldMembers := o.(*schema.Set)
	newMembers := n.(*schema.Set)

	// Create maps for easier lookup by user identifier
	oldMemberMap := make(map[string]map[string]interface{})
	newMemberMap := make(map[string]map[string]interface{})

	// Build old member map using user_id or user_email as key
	for _, member := range oldMembers.List() {
		m := member.(map[string]interface{})
		key := getMemberKey(m)
		if key != "" {
			oldMemberMap[key] = m
		}
	}

	// Build new member map using user_id or user_email as key
	for _, member := range newMembers.List() {
		m := member.(map[string]interface{})
		key := getMemberKey(m)
		if key != "" {
			newMemberMap[key] = m
		}
	}

	// Track which members have been updated to avoid duplicates
	updatedMembers := make(map[string]bool)

	// Check if max_budget_in_team has changed
	if d.HasChange("max_budget_in_team") {
		log.Printf("[DEBUG] max_budget_in_team changed, updating all existing members with new budget: %f", maxBudget)

		// Update ALL existing members with the new budget
		for key, newMember := range newMemberMap {
			if _, exists := oldMemberMap[key]; exists {
				updateData := map[string]interface{}{
					"team_id":            teamID,
					"role":               newMember["role"].(string),
					"max_budget_in_team": maxBudget,
				}
				if userID, ok := newMember["user_id"].(string); ok && userID != "" {
					updateData["user_id"] = userID
				}
				if userEmail, ok := newMember["user_email"].(string); ok && userEmail != "" {
					updateData["user_email"] = userEmail
				}

				log.Printf("[DEBUG] Update team member budget request payload: %+v", updateData)

				resp, err := MakeRequest(client, "POST", "/team/member_update", updateData)
				if err != nil {
					return fmt.Errorf("error updating team member budget: %v", err)
				}
				defer resp.Body.Close()

				if err := handleResponse(resp, "updating team member budget"); err != nil {
					return err
				}

				// Mark this member as updated
				updatedMembers[key] = true
			}
		}
	}

	// Find members to delete (in old but not in new)
	for key, oldMember := range oldMemberMap {
		if _, exists := newMemberMap[key]; !exists {
			deleteData := map[string]interface{}{
				"team_id": teamID,
			}
			if userID, ok := oldMember["user_id"].(string); ok && userID != "" {
				deleteData["user_id"] = userID
			}
			if userEmail, ok := oldMember["user_email"].(string); ok && userEmail != "" {
				deleteData["user_email"] = userEmail
			}

			log.Printf("[DEBUG] Delete team member request payload: %+v", deleteData)

			resp, err := MakeRequest(client, "POST", "/team/member_delete", deleteData)
			if err != nil {
				return fmt.Errorf("error deleting team member: %v", err)
			}
			defer resp.Body.Close()

			if err := handleResponse(resp, "deleting team member"); err != nil {
				return err
			}
		}
	}

	// Find members to update (exist in both but with different attributes)
	// Skip members that were already updated due to budget change
	for key, newMember := range newMemberMap {
		if oldMember, exists := oldMemberMap[key]; exists {
			// Skip if already updated due to budget change
			if updatedMembers[key] {
				continue
			}

			// Check if member attributes have changed
			if memberAttributesChanged(oldMember, newMember) {
				updateData := map[string]interface{}{
					"team_id":            teamID,
					"role":               newMember["role"].(string),
					"max_budget_in_team": maxBudget,
				}
				if userID, ok := newMember["user_id"].(string); ok && userID != "" {
					updateData["user_id"] = userID
				}
				if userEmail, ok := newMember["user_email"].(string); ok && userEmail != "" {
					updateData["user_email"] = userEmail
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
			}
		}
	}

	// Find members to add (in new but not in old)
	var membersToAdd []map[string]interface{}
	for key, newMember := range newMemberMap {
		if _, exists := oldMemberMap[key]; !exists {
			memberData := map[string]interface{}{
				"role": newMember["role"].(string),
			}
			if userID, ok := newMember["user_id"].(string); ok && userID != "" {
				memberData["user_id"] = userID
			}
			if userEmail, ok := newMember["user_email"].(string); ok && userEmail != "" {
				memberData["user_email"] = userEmail
			}
			membersToAdd = append(membersToAdd, memberData)
		}
	}

	if len(membersToAdd) > 0 {
		memberData := map[string]interface{}{
			"member":             membersToAdd,
			"team_id":            teamID,
			"max_budget_in_team": maxBudget,
		}

		log.Printf("[DEBUG] Adding new team members request payload: %+v", memberData)

		resp, err := MakeRequest(client, "POST", "/team/member_add", memberData)
		if err != nil {
			return fmt.Errorf("error adding team members: %v", err)
		}
		defer resp.Body.Close()

		if err := handleResponse(resp, "adding team members"); err != nil {
			return err
		}
	}

	return resourceLiteLLMTeamMemberAddRead(d, m)
}

// getMemberKey returns a unique key for a member based on user_id or user_email
func getMemberKey(member map[string]interface{}) string {
	if userID, ok := member["user_id"].(string); ok && userID != "" {
		return "id:" + userID
	}
	if userEmail, ok := member["user_email"].(string); ok && userEmail != "" {
		return "email:" + userEmail
	}
	return ""
}

// memberAttributesChanged checks if member attributes have changed between old and new
func memberAttributesChanged(oldMember, newMember map[string]interface{}) bool {
	// Compare role
	oldRole, _ := oldMember["role"].(string)
	newRole, _ := newMember["role"].(string)
	if oldRole != newRole {
		return true
	}

	// Note: max_budget_in_team is handled at the resource level, not per member
	// so we don't need to compare it here

	return false
}

func resourceLiteLLMTeamMemberAddDelete(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)
	teamID := d.Get("team_id").(string)
	members := d.Get("member").(*schema.Set)

	// Delete each member
	for _, member := range members.List() {
		m := member.(map[string]interface{})
		deleteData := map[string]interface{}{
			"team_id": teamID,
		}
		if userID, ok := m["user_id"].(string); ok && userID != "" {
			deleteData["user_id"] = userID
		}
		if userEmail, ok := m["user_email"].(string); ok && userEmail != "" {
			deleteData["user_email"] = userEmail
		}

		resp, err := MakeRequest(client, "POST", "/team/member_delete", deleteData)
		if err != nil {
			return fmt.Errorf("error deleting team member: %v", err)
		}
		defer resp.Body.Close()

		if err := handleResponse(resp, "deleting team member"); err != nil {
			return err
		}
	}

	d.SetId("")
	return nil
}
