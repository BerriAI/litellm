package litellm

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

const endpointAccessGroupNew = "/access_group/new"

type accessGroupInfoResponse struct {
	AccessGroup     string   `json:"access_group"`
	ModelNames      []string `json:"model_names"`
	DeploymentCount int      `json:"deployment_count"`
}

func resourceLiteLLMAccessGroup() *schema.Resource {
	return &schema.Resource{
		Create: resourceLiteLLMAccessGroupCreate,
		Read:   resourceLiteLLMAccessGroupRead,
		Update: resourceLiteLLMAccessGroupUpdate,
		Delete: resourceLiteLLMAccessGroupDelete,

		Importer: &schema.ResourceImporter{StateContext: schema.ImportStatePassthroughContext},

		Schema: map[string]*schema.Schema{
			"access_group": {
				Type:     schema.TypeString,
				Required: true,
				ForceNew: true,
			},
			"model_names": {
				Type:     schema.TypeList,
				Optional: true,
				Computed: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
			"model_ids": {
				Type:     schema.TypeList,
				Optional: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
			"deployment_count": {
				Type:     schema.TypeInt,
				Computed: true,
			},
		},
	}
}

func buildAccessGroupData(d *schema.ResourceData) map[string]interface{} {
	data := map[string]interface{}{}
	for _, key := range []string{"model_names", "model_ids"} {
		if v, ok := d.GetOk(key); ok {
			data[key] = v
		}
	}
	return data
}

func resourceLiteLLMAccessGroupCreate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	name := d.Get("access_group").(string)
	groupData := buildAccessGroupData(d)
	groupData["access_group"] = name

	log.Printf("[DEBUG] Create access group request payload: %+v", groupData)

	resp, err := MakeRequest(client, "POST", endpointAccessGroupNew, groupData)
	if err != nil {
		return fmt.Errorf("error creating access group: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "creating access group"); err != nil {
		return err
	}

	d.SetId(name)
	log.Printf("[INFO] Access group created with name: %s", name)

	return resourceLiteLLMAccessGroupRead(d, m)
}

func resourceLiteLLMAccessGroupRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	log.Printf("[INFO] Reading access group: %s", d.Id())

	resp, err := MakeRequest(client, "GET", fmt.Sprintf("/access_group/%s/info", d.Id()), nil)
	if err != nil {
		return fmt.Errorf("error reading access group: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		log.Printf("[WARN] Access group %s not found, removing from state", d.Id())
		d.SetId("")
		return nil
	}

	if err := handleResponse(resp, "reading access group"); err != nil {
		return err
	}

	var info accessGroupInfoResponse
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return fmt.Errorf("error decoding access group info response: %w", err)
	}

	d.Set("access_group", GetStringValue(info.AccessGroup, d.Id()))
	d.Set("model_names", info.ModelNames)
	d.Set("deployment_count", info.DeploymentCount)

	log.Printf("[INFO] Successfully read access group: %s", d.Id())
	return nil
}

func resourceLiteLLMAccessGroupUpdate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	groupData := buildAccessGroupData(d)
	log.Printf("[DEBUG] Update access group request payload: %+v", groupData)

	resp, err := MakeRequest(client, "PUT", fmt.Sprintf("/access_group/%s/update", d.Id()), groupData)
	if err != nil {
		return fmt.Errorf("error updating access group: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "updating access group"); err != nil {
		return err
	}

	log.Printf("[INFO] Successfully updated access group: %s", d.Id())
	return resourceLiteLLMAccessGroupRead(d, m)
}

func resourceLiteLLMAccessGroupDelete(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	log.Printf("[INFO] Deleting access group: %s", d.Id())

	resp, err := MakeRequest(client, "DELETE", fmt.Sprintf("/access_group/%s/delete", d.Id()), nil)
	if err != nil {
		return fmt.Errorf("error deleting access group: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "deleting access group"); err != nil {
		return err
	}

	log.Printf("[INFO] Successfully deleted access group: %s", d.Id())
	d.SetId("")
	return nil
}
