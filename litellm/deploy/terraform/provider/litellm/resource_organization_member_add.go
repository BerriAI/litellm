package litellm

import (
	"fmt"
	"log"
	"strings"

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
			"member_user_ids": {
				Type:     schema.TypeMap,
				Computed: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
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
	client := m.(*Client)
	orgID := d.Get("organization_id").(string)

	members, found, err := fetchOrganizationMembers(client, orgID)
	if err != nil {
		return err
	}
	if !found {
		log.Printf("[WARN] Organization %s not found, removing organization members from state", orgID)
		d.SetId("")
		return nil
	}

	// Keep only the configured members that are still in the organization,
	// each with the role the proxy reports. Members removed out of band drop
	// out of state, so the next apply adds them back.
	resolvedUserIDs := d.Get("member_user_ids").(map[string]interface{})
	current := make([]interface{}, 0)
	currentUserIDs := make(map[string]interface{})
	for _, member := range d.Get("member").(*schema.Set).List() {
		configured := member.(map[string]interface{})
		configuredID, _ := configured["user_id"].(string)
		configuredEmail, _ := configured["user_email"].(string)

		userID, userEmail := orgMemberIdentity(configured, resolvedUserIDs)
		existing, ok := findOrgMember(members, userID, userEmail)
		if !ok {
			log.Printf("[WARN] Organization member (user_id=%q, user_email=%q) not found in organization %s", configuredID, configuredEmail, orgID)
			continue
		}
		current = append(current, map[string]interface{}{
			"user_id":    configuredID,
			"user_email": configuredEmail,
			"role":       existing.Role,
		})
		if configuredID == "" {
			currentUserIDs[strings.ToLower(configuredEmail)] = existing.UserID
		}
	}

	if err := d.Set("member", current); err != nil {
		return fmt.Errorf("error setting member: %w", err)
	}
	if err := d.Set("member_user_ids", currentUserIDs); err != nil {
		return fmt.Errorf("error setting member_user_ids: %w", err)
	}
	return nil
}

func resourceLiteLLMOrganizationMemberAddUpdate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)
	orgID := d.Get("organization_id").(string)

	o, n := d.GetChange("member")
	oldMembers := o.(*schema.Set)
	newMembers := n.(*schema.Set)
	resolvedUserIDs := d.Get("member_user_ids").(map[string]interface{})

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
			userID, userEmail := orgMemberIdentity(oldMember, resolvedUserIDs)
			if err := deleteOrganizationMember(client, orgID, userID, userEmail); err != nil {
				return err
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
				userID, userEmail := orgMemberIdentity(newMember, resolvedUserIDs)
				if userID != "" {
					updateData["user_id"] = userID
				}
				if userEmail != "" {
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
	resolvedUserIDs := d.Get("member_user_ids").(map[string]interface{})

	// Delete each member
	for _, member := range members.List() {
		userID, userEmail := orgMemberIdentity(member.(map[string]interface{}), resolvedUserIDs)
		if err := deleteOrganizationMember(client, orgID, userID, userEmail); err != nil {
			return err
		}
	}

	d.SetId("")
	return nil
}
