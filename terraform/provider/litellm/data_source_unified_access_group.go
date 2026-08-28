package litellm

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

const endpointUnifiedAccessGroupList = "/v1/unified_access_group"

func unifiedAccessGroupComputedSchema() map[string]*schema.Schema {
	return map[string]*schema.Schema{
		"access_group_name": {
			Type:     schema.TypeString,
			Computed: true,
		},
		"description": {
			Type:     schema.TypeString,
			Computed: true,
		},
		"access_model_names": {
			Type:     schema.TypeList,
			Computed: true,
			Elem:     &schema.Schema{Type: schema.TypeString},
		},
		"access_mcp_server_ids": {
			Type:     schema.TypeList,
			Computed: true,
			Elem:     &schema.Schema{Type: schema.TypeString},
		},
		"access_agent_ids": {
			Type:     schema.TypeList,
			Computed: true,
			Elem:     &schema.Schema{Type: schema.TypeString},
		},
		"assigned_team_ids": {
			Type:     schema.TypeList,
			Computed: true,
			Elem:     &schema.Schema{Type: schema.TypeString},
		},
		"assigned_key_ids": {
			Type:     schema.TypeList,
			Computed: true,
			Elem:     &schema.Schema{Type: schema.TypeString},
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
	}
}

func dataSourceLiteLLMUnifiedAccessGroup() *schema.Resource {
	dsSchema := unifiedAccessGroupComputedSchema()
	dsSchema["access_group_id"] = &schema.Schema{
		Type:        schema.TypeString,
		Required:    true,
		Description: "ID of the unified access group to retrieve",
	}

	return &schema.Resource{
		Read:   dataSourceLiteLLMUnifiedAccessGroupRead,
		Schema: dsSchema,
	}
}

func dataSourceLiteLLMUnifiedAccessGroupRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)
	groupID := d.Get("access_group_id").(string)

	resp, err := MakeRequest(client, "GET", fmt.Sprintf("/v1/unified_access_group/%s", groupID), nil)
	if err != nil {
		return fmt.Errorf("error reading unified access group: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return fmt.Errorf("unified access group '%s' not found", groupID)
	}

	if err := handleResponse(resp, "reading unified access group"); err != nil {
		return err
	}

	var group unifiedAccessGroupResponse
	if err := json.NewDecoder(resp.Body).Decode(&group); err != nil {
		return fmt.Errorf("error decoding unified access group info response: %w", err)
	}

	d.SetId(GetStringValue(group.AccessGroupID, groupID))
	setUnifiedAccessGroupFields(d, group)

	return nil
}

func dataSourceLiteLLMUnifiedAccessGroups() *schema.Resource {
	itemSchema := unifiedAccessGroupComputedSchema()
	itemSchema["access_group_id"] = &schema.Schema{
		Type:     schema.TypeString,
		Computed: true,
	}

	return &schema.Resource{
		Read: dataSourceLiteLLMUnifiedAccessGroupsRead,

		Schema: map[string]*schema.Schema{
			"access_groups": {
				Type:     schema.TypeList,
				Computed: true,
				Elem:     &schema.Resource{Schema: itemSchema},
			},
			"ids": {
				Type:     schema.TypeList,
				Computed: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
		},
	}
}

func dataSourceLiteLLMUnifiedAccessGroupsRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	resp, err := MakeRequest(client, "GET", endpointUnifiedAccessGroupList, nil)
	if err != nil {
		return fmt.Errorf("error listing unified access groups: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "listing unified access groups"); err != nil {
		return err
	}

	var groups []unifiedAccessGroupResponse
	if err := json.NewDecoder(resp.Body).Decode(&groups); err != nil {
		return fmt.Errorf("error decoding unified access group list response: %w", err)
	}

	items := make([]map[string]interface{}, 0, len(groups))
	ids := make([]string, 0, len(groups))
	for _, group := range groups {
		items = append(items, unifiedAccessGroupFlatten(group))
		ids = append(ids, group.AccessGroupID)
	}

	d.SetId("unified_access_groups")
	d.Set("access_groups", items)
	d.Set("ids", ids)

	return nil
}

func unifiedAccessGroupFlatten(group unifiedAccessGroupResponse) map[string]interface{} {
	item := map[string]interface{}{
		"access_group_id":       group.AccessGroupID,
		"access_group_name":     group.AccessGroupName,
		"access_model_names":    group.AccessModelNames,
		"access_mcp_server_ids": group.AccessMCPServerIDs,
		"access_agent_ids":      group.AccessAgentIDs,
		"assigned_team_ids":     group.AssignedTeamIDs,
		"assigned_key_ids":      group.AssignedKeyIDs,
		"created_at":            group.CreatedAt,
		"updated_at":            group.UpdatedAt,
	}
	if group.Description != nil {
		item["description"] = *group.Description
	}
	if group.CreatedBy != nil {
		item["created_by"] = *group.CreatedBy
	}
	if group.UpdatedBy != nil {
		item["updated_by"] = *group.UpdatedBy
	}
	return item
}
