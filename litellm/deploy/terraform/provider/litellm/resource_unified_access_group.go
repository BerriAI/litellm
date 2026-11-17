package litellm

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

const endpointUnifiedAccessGroupCreate = "/v1/unified_access_group"

var unifiedAccessGroupListFields = []string{
	"access_model_names",
	"access_mcp_server_ids",
	"access_agent_ids",
	"assigned_team_ids",
	"assigned_key_ids",
}

type unifiedAccessGroupResponse struct {
	AccessGroupID      string   `json:"access_group_id"`
	AccessGroupName    string   `json:"access_group_name"`
	Description        *string  `json:"description"`
	AccessModelNames   []string `json:"access_model_names"`
	AccessMCPServerIDs []string `json:"access_mcp_server_ids"`
	AccessAgentIDs     []string `json:"access_agent_ids"`
	AssignedTeamIDs    []string `json:"assigned_team_ids"`
	AssignedKeyIDs     []string `json:"assigned_key_ids"`
	CreatedAt          string   `json:"created_at"`
	CreatedBy          *string  `json:"created_by"`
	UpdatedAt          string   `json:"updated_at"`
	UpdatedBy          *string  `json:"updated_by"`
}

func resourceLiteLLMUnifiedAccessGroup() *schema.Resource {
	return &schema.Resource{
		Create: resourceLiteLLMUnifiedAccessGroupCreate,
		Read:   resourceLiteLLMUnifiedAccessGroupRead,
		Update: resourceLiteLLMUnifiedAccessGroupUpdate,
		Delete: resourceLiteLLMUnifiedAccessGroupDelete,

		Importer: &schema.ResourceImporter{StateContext: schema.ImportStatePassthroughContext},

		Schema: map[string]*schema.Schema{
			"access_group_name": {
				Type:     schema.TypeString,
				Required: true,
			},
			"description": {
				Type:     schema.TypeString,
				Optional: true,
			},
			"access_model_names": {
				Type:     schema.TypeList,
				Optional: true,
				Computed: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
			"access_mcp_server_ids": {
				Type:     schema.TypeList,
				Optional: true,
				Computed: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
			"access_agent_ids": {
				Type:     schema.TypeList,
				Optional: true,
				Computed: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
			"assigned_team_ids": {
				Type:     schema.TypeList,
				Optional: true,
				Computed: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
			"assigned_key_ids": {
				Type:     schema.TypeList,
				Optional: true,
				Computed: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
			"access_group_id": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"created_at": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"created_by": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"updated_at": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"updated_by": {
				Type:     schema.TypeString,
				Computed: true,
			},
		},
	}
}

func buildUnifiedAccessGroupData(d *schema.ResourceData) map[string]interface{} {
	data := map[string]interface{}{
		"access_group_name": d.Get("access_group_name").(string),
	}
	if v, ok := d.GetOk("description"); ok {
		data["description"] = v
	}
	for _, key := range unifiedAccessGroupListFields {
		data[key] = d.Get(key)
	}
	return data
}

func setUnifiedAccessGroupFields(d *schema.ResourceData, group unifiedAccessGroupResponse) {
	d.Set("access_group_id", group.AccessGroupID)
	d.Set("access_group_name", group.AccessGroupName)
	if group.Description != nil {
		d.Set("description", *group.Description)
	}
	d.Set("access_model_names", group.AccessModelNames)
	d.Set("access_mcp_server_ids", group.AccessMCPServerIDs)
	d.Set("access_agent_ids", group.AccessAgentIDs)
	d.Set("assigned_team_ids", group.AssignedTeamIDs)
	d.Set("assigned_key_ids", group.AssignedKeyIDs)
	d.Set("created_at", group.CreatedAt)
	if group.CreatedBy != nil {
		d.Set("created_by", *group.CreatedBy)
	}
	d.Set("updated_at", group.UpdatedAt)
	if group.UpdatedBy != nil {
		d.Set("updated_by", *group.UpdatedBy)
	}
}

func resourceLiteLLMUnifiedAccessGroupCreate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	groupData := buildUnifiedAccessGroupData(d)
	log.Printf("[DEBUG] Create unified access group request payload: %+v", groupData)

	resp, err := MakeRequest(client, "POST", endpointUnifiedAccessGroupCreate, groupData)
	if err != nil {
		return fmt.Errorf("error creating unified access group: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "creating unified access group"); err != nil {
		return err
	}

	var group unifiedAccessGroupResponse
	if err := json.NewDecoder(resp.Body).Decode(&group); err != nil {
		return fmt.Errorf("error decoding unified access group create response: %w", err)
	}

	if group.AccessGroupID == "" {
		return fmt.Errorf("unified access group create response missing access_group_id")
	}

	d.SetId(group.AccessGroupID)
	log.Printf("[INFO] Unified access group created with ID: %s", group.AccessGroupID)

	return resourceLiteLLMUnifiedAccessGroupRead(d, m)
}

func resourceLiteLLMUnifiedAccessGroupRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	log.Printf("[INFO] Reading unified access group with ID: %s", d.Id())

	resp, err := MakeRequest(client, "GET", fmt.Sprintf("/v1/unified_access_group/%s", d.Id()), nil)
	if err != nil {
		return fmt.Errorf("error reading unified access group: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		log.Printf("[WARN] Unified access group with ID %s not found, removing from state", d.Id())
		d.SetId("")
		return nil
	}

	if err := handleResponse(resp, "reading unified access group"); err != nil {
		return err
	}

	var group unifiedAccessGroupResponse
	if err := json.NewDecoder(resp.Body).Decode(&group); err != nil {
		return fmt.Errorf("error decoding unified access group info response: %w", err)
	}

	setUnifiedAccessGroupFields(d, group)

	log.Printf("[INFO] Successfully read unified access group with ID: %s", d.Id())
	return nil
}

func resourceLiteLLMUnifiedAccessGroupUpdate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	groupData := buildUnifiedAccessGroupData(d)
	log.Printf("[DEBUG] Update unified access group request payload: %+v", groupData)

	resp, err := MakeRequest(client, "PUT", fmt.Sprintf("/v1/unified_access_group/%s", d.Id()), groupData)
	if err != nil {
		return fmt.Errorf("error updating unified access group: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "updating unified access group"); err != nil {
		return err
	}

	log.Printf("[INFO] Successfully updated unified access group with ID: %s", d.Id())
	return resourceLiteLLMUnifiedAccessGroupRead(d, m)
}

func resourceLiteLLMUnifiedAccessGroupDelete(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	log.Printf("[INFO] Deleting unified access group with ID: %s", d.Id())

	resp, err := MakeRequest(client, "DELETE", fmt.Sprintf("/v1/unified_access_group/%s", d.Id()), nil)
	if err != nil {
		return fmt.Errorf("error deleting unified access group: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusNoContent {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("error deleting unified access group: %s - %s", resp.Status, string(body))
	}

	log.Printf("[INFO] Successfully deleted unified access group with ID: %s", d.Id())
	d.SetId("")
	return nil
}
