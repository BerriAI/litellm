package litellm

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/url"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

const (
	endpointTeamBlock   = "/team/block"
	endpointTeamUnblock = "/team/unblock"
)

type TeamBlockInfoResponse struct {
	TeamInfo struct {
		Blocked *bool `json:"blocked"`
	} `json:"team_info"`
}

func resourceLiteLLMTeamBlock() *schema.Resource {
	return &schema.Resource{
		Create: resourceLiteLLMTeamBlockCreate,
		Read:   resourceLiteLLMTeamBlockRead,
		Delete: resourceLiteLLMTeamBlockDelete,

		Importer: &schema.ResourceImporter{
			StateContext: schema.ImportStatePassthroughContext,
		},

		Schema: map[string]*schema.Schema{
			"team_id": {
				Type:        schema.TypeString,
				Required:    true,
				ForceNew:    true,
				Description: "The ID of the team to block. Destroying this resource unblocks the team",
			},
			"blocked": {
				Type:        schema.TypeBool,
				Computed:    true,
				Description: "Whether the team is currently blocked",
			},
		},
	}
}

func resourceLiteLLMTeamBlockCreate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)
	teamID := d.Get("team_id").(string)

	log.Printf("[INFO] Blocking team with ID: %s", teamID)

	resp, err := MakeRequest(client, "POST", endpointTeamBlock, map[string]interface{}{"team_id": teamID})
	if err != nil {
		return fmt.Errorf("error blocking team: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "blocking team"); err != nil {
		return err
	}

	d.SetId(teamID)
	return resourceLiteLLMTeamBlockRead(d, m)
}

func resourceLiteLLMTeamBlockRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)
	teamID := d.Id()

	log.Printf("[INFO] Reading block state for team with ID: %s", teamID)

	resp, err := MakeRequest(client, "GET", fmt.Sprintf("/team/info?team_id=%s", url.QueryEscape(teamID)), nil)
	if err != nil {
		return fmt.Errorf("error reading team info: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		log.Printf("[WARN] Team with ID %s not found, removing team block from state", teamID)
		d.SetId("")
		return nil
	}

	if err := handleResponse(resp, "reading team info"); err != nil {
		return err
	}

	var infoResp TeamBlockInfoResponse
	if err := json.NewDecoder(resp.Body).Decode(&infoResp); err != nil {
		return fmt.Errorf("error decoding team info response: %w", err)
	}

	if infoResp.TeamInfo.Blocked == nil || !*infoResp.TeamInfo.Blocked {
		log.Printf("[WARN] Team with ID %s is no longer blocked, removing team block from state", teamID)
		d.SetId("")
		return nil
	}

	d.Set("team_id", teamID)
	d.Set("blocked", true)
	return nil
}

func resourceLiteLLMTeamBlockDelete(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	log.Printf("[INFO] Unblocking team with ID: %s", d.Id())

	resp, err := MakeRequest(client, "POST", endpointTeamUnblock, map[string]interface{}{"team_id": d.Id()})
	if err != nil {
		return fmt.Errorf("error unblocking team: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNotFound {
		if err := handleResponse(resp, "unblocking team"); err != nil {
			return err
		}
	}

	d.SetId("")
	return nil
}
