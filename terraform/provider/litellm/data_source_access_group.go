package litellm

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

const endpointAccessGroupList = "/access_group/list"

type accessGroupListResponse struct {
	AccessGroups []accessGroupInfoResponse `json:"access_groups"`
}

func dataSourceLiteLLMAccessGroup() *schema.Resource {
	return &schema.Resource{
		Read: dataSourceLiteLLMAccessGroupRead,

		Schema: map[string]*schema.Schema{
			"access_group": {
				Type:        schema.TypeString,
				Required:    true,
				Description: "Name of the access group to retrieve",
			},
			"model_names": {
				Type:     schema.TypeList,
				Computed: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
			"deployment_count": {
				Type:     schema.TypeInt,
				Computed: true,
			},
		},
	}
}

func dataSourceLiteLLMAccessGroupRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)
	name := d.Get("access_group").(string)

	resp, err := MakeRequest(client, "GET", fmt.Sprintf("/access_group/%s/info", name), nil)
	if err != nil {
		return fmt.Errorf("error reading access group: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return fmt.Errorf("access group '%s' not found", name)
	}

	if err := handleResponse(resp, "reading access group"); err != nil {
		return err
	}

	var info accessGroupInfoResponse
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return fmt.Errorf("error decoding access group info response: %w", err)
	}

	d.SetId(GetStringValue(info.AccessGroup, name))
	d.Set("access_group", GetStringValue(info.AccessGroup, name))
	d.Set("model_names", info.ModelNames)
	d.Set("deployment_count", info.DeploymentCount)

	return nil
}

func dataSourceLiteLLMAccessGroups() *schema.Resource {
	return &schema.Resource{
		Read: dataSourceLiteLLMAccessGroupsRead,

		Schema: map[string]*schema.Schema{
			"access_groups": {
				Type:     schema.TypeList,
				Computed: true,
				Elem: &schema.Resource{
					Schema: map[string]*schema.Schema{
						"access_group": {
							Type:     schema.TypeString,
							Computed: true,
						},
						"model_names": {
							Type:     schema.TypeList,
							Computed: true,
							Elem:     &schema.Schema{Type: schema.TypeString},
						},
						"deployment_count": {
							Type:     schema.TypeInt,
							Computed: true,
						},
					},
				},
			},
			"ids": {
				Type:     schema.TypeList,
				Computed: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
		},
	}
}

func dataSourceLiteLLMAccessGroupsRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	resp, err := MakeRequest(client, "GET", endpointAccessGroupList, nil)
	if err != nil {
		return fmt.Errorf("error listing access groups: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "listing access groups"); err != nil {
		return err
	}

	var listResp accessGroupListResponse
	if err := json.NewDecoder(resp.Body).Decode(&listResp); err != nil {
		return fmt.Errorf("error decoding access group list response: %w", err)
	}

	groups := make([]map[string]interface{}, 0, len(listResp.AccessGroups))
	ids := make([]string, 0, len(listResp.AccessGroups))
	for _, group := range listResp.AccessGroups {
		groups = append(groups, map[string]interface{}{
			"access_group":     group.AccessGroup,
			"model_names":      group.ModelNames,
			"deployment_count": group.DeploymentCount,
		})
		ids = append(ids, group.AccessGroup)
	}

	d.SetId("access_groups")
	d.Set("access_groups", groups)
	d.Set("ids", ids)

	return nil
}
