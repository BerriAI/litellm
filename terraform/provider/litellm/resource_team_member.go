package litellm

import (
	"fmt"
	"log"

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

func resourceLiteLLMTeamMemberRead(d *schema.ResourceData, m interface{}) error {
	// There's no specific endpoint to read a single team member
	// We might need to read the entire team and find the member
	// For now, we'll just return the data we have in the state
	log.Printf("[INFO] Reading team member with ID: %s", d.Id())
	return nil
}

func resourceLiteLLMTeamMemberUpdate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	updateData := map[string]interface{}{
		"user_id":            d.Get("user_id").(string),
		"user_email":         d.Get("user_email").(string),
		"team_id":            d.Get("team_id").(string),
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

	if err := handleResponse(resp, "deleting team member"); err != nil {
		return err
	}

	log.Printf("[INFO] Successfully deleted team member with ID: %s", d.Id())

	d.SetId("")
	return nil
}
