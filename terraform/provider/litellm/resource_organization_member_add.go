package litellm

import (
	"fmt"
	"log"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/validation"
)

func resourceLiteLLMOrganizationMemberAdd() *schema.Resource {
	return &schema.Resource{
		Create: resourceLiteLLMOrganizationMemberAddCreate,
		Read:   resourceLiteLLMOrganizationMemberAddRead,
		Update: resourceLiteLLMOrganizationMemberAddUpdate,
		Delete: resourceLiteLLMOrganizationMemberAddDelete,

		Schema: map[string]*schema.Schema{
			"organization_id": {
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
								"org_admin",
								"internal_user",
								"internal_user_viewer",
							}, false),
						},
					},
				},
			},
		},
	}
}

func resourceLiteLLMOrganizationMemberAddCreate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	orgID := d.Get("organization_id").(string)
	members := d.Get("member").(*schema.Set)

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
		"member":          membersList,
		"organization_id": orgID,
	}

	log.Printf("[DEBUG] Create organization members request payload: %+v", memberData)

	resp, err := client.AddOrganizationMember(memberData)
	if err != nil {
		return fmt.Errorf("error adding organization members: %v", err)
	}

	log.Printf("[DEBUG] Create organization members response: %+v", resp)

	// Set ID as organization_id since this resource manages all members for an organization
	d.SetId(orgID)

	return resourceLiteLLMOrganizationMemberAddRead(d, m)
}

func resourceLiteLLMOrganizationMemberAddRead(d *schema.ResourceData, m interface{}) error {
	// The API doesn't provide a way to read specific organization members easily
	// We'll maintain the state as is
	return nil
}

func resourceLiteLLMOrganizationMemberAddUpdate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)
	orgID := d.Get("organization_id").(string)

	o, n := d.GetChange("member")
	oldMembers := o.(*schema.Set)
	newMembers := n.(*schema.Set)

	// Create maps for easier lookup by user identifier
	oldMemberMap := make(map[string]map[string]interface{})
	newMemberMap := make(map[string]map[string]interface{})

	// Build old member map using user_id or user_email as key
	for _, member := range oldMembers.List() {
		m := member.(map[string]interface{})
		key := getOrgMemberKey(m)
		if key != "" {
			oldMemberMap[key] = m
		}
	}

	// Build new member map using user_id or user_email as key
	for _, member := range newMembers.List() {
		m := member.(map[string]interface{})
		key := getOrgMemberKey(m)
		if key != "" {
			newMemberMap[key] = m
		}
	}

	// Find members to delete (in old but not in new)
	for key, oldMember := range oldMemberMap {
		if _, exists := newMemberMap[key]; !exists {
			deleteData := map[string]interface{}{
				"organization_id": orgID,
			}
			if userID, ok := oldMember["user_id"].(string); ok && userID != "" {
				deleteData["user_id"] = userID
			}
			if userEmail, ok := oldMember["user_email"].(string); ok && userEmail != "" {
				deleteData["user_email"] = userEmail
			}

			log.Printf("[DEBUG] Delete organization member request payload: %+v", deleteData)

			_, err := client.DeleteOrganizationMember(deleteData)
			if err != nil {
				return fmt.Errorf("error deleting organization member: %v", err)
			}
		}
	}

	// Find members to update (exist in both but with different attributes)
	for key, newMember := range newMemberMap {
		if oldMember, exists := oldMemberMap[key]; exists {
			// Check if member attributes have changed
			if orgMemberAttributesChanged(oldMember, newMember) {
				updateData := map[string]interface{}{
					"organization_id": orgID,
					"role":            newMember["role"].(string),
				}
				if userID, ok := newMember["user_id"].(string); ok && userID != "" {
					updateData["user_id"] = userID
				}
				if userEmail, ok := newMember["user_email"].(string); ok && userEmail != "" {
					updateData["user_email"] = userEmail
				}

				log.Printf("[DEBUG] Update organization member request payload: %+v", updateData)

				_, err := client.UpdateOrganizationMember(updateData)
				if err != nil {
					return fmt.Errorf("error updating organization member: %v", err)
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
			"member":          membersToAdd,
			"organization_id": orgID,
		}

		log.Printf("[DEBUG] Adding new organization members request payload: %+v", memberData)

		resp, err := client.AddOrganizationMember(memberData)
		if err != nil {
			return fmt.Errorf("error adding organization members: %v", err)
		}

		log.Printf("[DEBUG] Add organization members response: %+v", resp)
	}

	return resourceLiteLLMOrganizationMemberAddRead(d, m)
}

// getOrgMemberKey returns a unique key for a member based on user_id or user_email
func getOrgMemberKey(member map[string]interface{}) string {
	if userID, ok := member["user_id"].(string); ok && userID != "" {
		return "id:" + userID
	}
	if userEmail, ok := member["user_email"].(string); ok && userEmail != "" {
		return "email:" + userEmail
	}
	return ""
}

// orgMemberAttributesChanged checks if member attributes have changed between old and new
func orgMemberAttributesChanged(oldMember, newMember map[string]interface{}) bool {
	// Compare role
	oldRole, _ := oldMember["role"].(string)
	newRole, _ := newMember["role"].(string)
	return oldRole != newRole
}

func resourceLiteLLMOrganizationMemberAddDelete(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)
	orgID := d.Get("organization_id").(string)
	members := d.Get("member").(*schema.Set)

	// Delete each member
	for _, member := range members.List() {
		m := member.(map[string]interface{})
		deleteData := map[string]interface{}{
			"organization_id": orgID,
		}
		if userID, ok := m["user_id"].(string); ok && userID != "" {
			deleteData["user_id"] = userID
		}
		if userEmail, ok := m["user_email"].(string); ok && userEmail != "" {
			deleteData["user_email"] = userEmail
		}

		_, err := client.DeleteOrganizationMember(deleteData)
		if err != nil {
			return fmt.Errorf("error deleting organization member: %v", err)
		}
	}

	d.SetId("")
	return nil
}
