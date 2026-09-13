package litellm

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func dataSourceLiteLLMAgent() *schema.Resource {
	return &schema.Resource{
		Read: dataSourceLiteLLMAgentRead,

		Schema: map[string]*schema.Schema{
			"agent_id": {
				Type:        schema.TypeString,
				Required:    true,
				Description: "Unique identifier of the agent to retrieve.",
			},
			"agent_name": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"agent_card_params": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "A2A agent card as a JSON object string.",
			},
			"object_permission": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Access control permissions as a JSON object string.",
			},
			"extra_headers": {
				Type:     schema.TypeList,
				Computed: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
			"tpm_limit": {
				Type:     schema.TypeInt,
				Computed: true,
			},
			"rpm_limit": {
				Type:     schema.TypeInt,
				Computed: true,
			},
			"session_tpm_limit": {
				Type:     schema.TypeInt,
				Computed: true,
			},
			"session_rpm_limit": {
				Type:     schema.TypeInt,
				Computed: true,
			},
			"spend": {
				Type:     schema.TypeFloat,
				Computed: true,
			},
			"created_at": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"updated_at": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"created_by": {
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

func dataSourceLiteLLMAgentRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)
	agentID := d.Get("agent_id").(string)

	resp, err := MakeRequest(client, "GET", fmt.Sprintf(endpointAgentByID, agentID), nil)
	if err != nil {
		return fmt.Errorf("error reading agent: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return fmt.Errorf("agent '%s' not found", agentID)
	}

	if err := handleResponse(resp, "reading agent"); err != nil {
		return err
	}

	var agentResp agentAPIResponse
	if err := json.NewDecoder(resp.Body).Decode(&agentResp); err != nil {
		return fmt.Errorf("error decoding agent info response: %w", err)
	}

	d.SetId(agentResp.AgentID)
	d.Set("agent_name", agentResp.AgentName)

	if agentResp.AgentCardParams != nil {
		cardJSON, err := json.Marshal(agentResp.AgentCardParams)
		if err != nil {
			return fmt.Errorf("error encoding agent_card_params: %w", err)
		}
		d.Set("agent_card_params", string(cardJSON))
	}
	if agentResp.ObjectPermission != nil {
		permJSON, err := json.Marshal(agentResp.ObjectPermission)
		if err != nil {
			return fmt.Errorf("error encoding object_permission: %w", err)
		}
		d.Set("object_permission", string(permJSON))
	}

	if agentResp.ExtraHeaders != nil {
		d.Set("extra_headers", agentResp.ExtraHeaders)
	}
	if agentResp.TPMLimit != nil {
		d.Set("tpm_limit", *agentResp.TPMLimit)
	}
	if agentResp.RPMLimit != nil {
		d.Set("rpm_limit", *agentResp.RPMLimit)
	}
	if agentResp.SessionTPMLimit != nil {
		d.Set("session_tpm_limit", *agentResp.SessionTPMLimit)
	}
	if agentResp.SessionRPMLimit != nil {
		d.Set("session_rpm_limit", *agentResp.SessionRPMLimit)
	}
	if agentResp.Spend != nil {
		d.Set("spend", *agentResp.Spend)
	}
	d.Set("created_at", agentResp.CreatedAt)
	d.Set("updated_at", agentResp.UpdatedAt)
	d.Set("created_by", agentResp.CreatedBy)
	d.Set("updated_by", agentResp.UpdatedBy)

	return nil
}

func dataSourceLiteLLMAgents() *schema.Resource {
	return &schema.Resource{
		Read: dataSourceLiteLLMAgentsRead,

		Schema: map[string]*schema.Schema{
			"health_check": {
				Type:     schema.TypeBool,
				Optional: true,
				Default:  false,
				Description: "When true, the proxy probes each agent's URL and only returns agents that are " +
					"reachable or have no URL.",
			},
			"ids": {
				Type:     schema.TypeList,
				Computed: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
			"agents": {
				Type:     schema.TypeList,
				Computed: true,
				Elem: &schema.Resource{
					Schema: map[string]*schema.Schema{
						"agent_id": {
							Type:     schema.TypeString,
							Computed: true,
						},
						"agent_name": {
							Type:     schema.TypeString,
							Computed: true,
						},
						"tpm_limit": {
							Type:     schema.TypeInt,
							Computed: true,
						},
						"rpm_limit": {
							Type:     schema.TypeInt,
							Computed: true,
						},
						"session_tpm_limit": {
							Type:     schema.TypeInt,
							Computed: true,
						},
						"session_rpm_limit": {
							Type:     schema.TypeInt,
							Computed: true,
						},
						"spend": {
							Type:     schema.TypeFloat,
							Computed: true,
						},
						"created_at": {
							Type:     schema.TypeString,
							Computed: true,
						},
						"updated_at": {
							Type:     schema.TypeString,
							Computed: true,
						},
						"created_by": {
							Type:     schema.TypeString,
							Computed: true,
						},
						"updated_by": {
							Type:     schema.TypeString,
							Computed: true,
						},
					},
				},
			},
		},
	}
}

func dataSourceLiteLLMAgentsRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	endpoint := endpointAgents
	if d.Get("health_check").(bool) {
		endpoint = fmt.Sprintf("%s?health_check=true", endpointAgents)
	}

	resp, err := MakeRequest(client, "GET", endpoint, nil)
	if err != nil {
		return fmt.Errorf("error listing agents: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "listing agents"); err != nil {
		return err
	}

	var agentResps []agentAPIResponse
	if err := json.NewDecoder(resp.Body).Decode(&agentResps); err != nil {
		return fmt.Errorf("error decoding agents list response: %w", err)
	}

	ids := make([]string, 0, len(agentResps))
	agents := make([]map[string]interface{}, 0, len(agentResps))
	for _, agentResp := range agentResps {
		ids = append(ids, agentResp.AgentID)

		agent := map[string]interface{}{
			"agent_id":   agentResp.AgentID,
			"agent_name": agentResp.AgentName,
			"created_at": agentResp.CreatedAt,
			"updated_at": agentResp.UpdatedAt,
			"created_by": agentResp.CreatedBy,
			"updated_by": agentResp.UpdatedBy,
		}
		if agentResp.TPMLimit != nil {
			agent["tpm_limit"] = *agentResp.TPMLimit
		}
		if agentResp.RPMLimit != nil {
			agent["rpm_limit"] = *agentResp.RPMLimit
		}
		if agentResp.SessionTPMLimit != nil {
			agent["session_tpm_limit"] = *agentResp.SessionTPMLimit
		}
		if agentResp.SessionRPMLimit != nil {
			agent["session_rpm_limit"] = *agentResp.SessionRPMLimit
		}
		if agentResp.Spend != nil {
			agent["spend"] = *agentResp.Spend
		}
		agents = append(agents, agent)
	}

	d.SetId(strconv.FormatInt(time.Now().UnixNano(), 10))
	d.Set("ids", ids)
	d.Set("agents", agents)

	return nil
}
