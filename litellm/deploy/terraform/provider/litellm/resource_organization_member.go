package litellm

import (
	"fmt"
	"log"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/validation"
)

func resourceLiteLLMOrganizationMember() *schema.Resource {
	return &schema.Resource{
		Create: resourceLiteLLMOrganizationMemberCreate,
		Read:   resourceLiteLLMOrganizationMemberRead,
		Update: resourceLiteLLMOrganizationMemberUpdate,
		Delete: resourceLiteLLMOrganizationMemberDelete,

		Schema: map[string]*schema.Schema{
			"organization_id": {
				Type:     schema.TypeString,
				Required: true,
			},
			"user_id": {
				Type:     schema.TypeString,
				Required: true,
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
	}
}

func resourceLiteLLMOrganizationMemberCreate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	memberData := map[string]interface{}{
		"member": []map[string]interface{}{
			{
				"role":       d.Get("role").(string),
				"user_id":    d.Get("user_id").(string),
				"user_email": d.Get("user_email").(string),
			},
		},
		"organization_id": d.Get("organization_id").(string),
	}

	log.Printf("[DEBUG] Create organization member request payload: %+v", memberData)

	resp, err := client.AddOrganizationMember(memberData)
	if err != nil {
		return fmt.Errorf("error creating organization member: %v", err)
	}

	log.Printf("[DEBUG] Create organization member response: %+v", resp)

	// Set a composite ID since there's no specific member ID returned
	d.SetId(fmt.Sprintf("%s:%s", d.Get("organization_id").(string), d.Get("user_id").(string)))

	log.Printf("[INFO] Organization member created with ID: %s", d.Id())

	return resourceLiteLLMOrganizationMemberRead(d, m)
}

func resourceLiteLLMOrganizationMemberRead(d *schema.ResourceData, m interface{}) error {
	// There's no specific endpoint to read a single organization member
	// We'll just return the data we have in the state
	log.Printf("[INFO] Reading organization member with ID: %s", d.Id())
	return nil
}

func resourceLiteLLMOrganizationMemberUpdate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	updateData := map[string]interface{}{
		"user_id":         d.Get("user_id").(string),
		"user_email":      d.Get("user_email").(string),
		"organization_id": d.Get("organization_id").(string),
		"role":            d.Get("role").(string),
	}

	log.Printf("[DEBUG] Update organization member request payload: %+v", updateData)

	resp, err := client.UpdateOrganizationMember(updateData)
	if err != nil {
		return fmt.Errorf("error updating organization member: %v", err)
	}

	log.Printf("[DEBUG] Update organization member response: %+v", resp)

	log.Printf("[INFO] Successfully updated organization member with ID: %s", d.Id())

	return resourceLiteLLMOrganizationMemberRead(d, m)
}

func resourceLiteLLMOrganizationMemberDelete(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	deleteData := map[string]interface{}{
		"user_id":         d.Get("user_id").(string),
		"user_email":      d.Get("user_email").(string),
		"organization_id": d.Get("organization_id").(string),
	}

	log.Printf("[DEBUG] Delete organization member request payload: %+v", deleteData)

	_, err := client.DeleteOrganizationMember(deleteData)
	if err != nil {
		return fmt.Errorf("error deleting organization member: %v", err)
	}

	log.Printf("[INFO] Successfully deleted organization member with ID: %s", d.Id())

	d.SetId("")
	return nil
}
