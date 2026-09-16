package litellm

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"reflect"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

const (
	endpointAgents    = "/v1/agents"
	endpointAgentByID = "/v1/agents/%s"
)

type agentAPIResponse struct {
	AgentID          string                 `json:"agent_id"`
	AgentName        string                 `json:"agent_name"`
	AgentCardParams  map[string]interface{} `json:"agent_card_params"`
	ObjectPermission map[string]interface{} `json:"object_permission"`
	ExtraHeaders     []string               `json:"extra_headers"`
	TPMLimit         *int                   `json:"tpm_limit"`
	RPMLimit         *int                   `json:"rpm_limit"`
	SessionTPMLimit  *int                   `json:"session_tpm_limit"`
	SessionRPMLimit  *int                   `json:"session_rpm_limit"`
	Spend            *float64               `json:"spend"`
	CreatedAt        string                 `json:"created_at"`
	UpdatedAt        string                 `json:"updated_at"`
	CreatedBy        string                 `json:"created_by"`
	UpdatedBy        string                 `json:"updated_by"`
}

func agentSuppressEquivalentJSON(k, oldValue, newValue string, d *schema.ResourceData) bool {
	var oldObj, newObj interface{}
	if err := json.Unmarshal([]byte(oldValue), &oldObj); err != nil {
		return false
	}
	if err := json.Unmarshal([]byte(newValue), &newObj); err != nil {
		return false
	}
	return reflect.DeepEqual(oldObj, newObj)
}

func agentParseJSONObject(raw, field string) (map[string]interface{}, error) {
	var obj map[string]interface{}
	if err := json.Unmarshal([]byte(raw), &obj); err != nil {
		return nil, fmt.Errorf("%s must be a JSON object: %w", field, err)
	}
	return obj, nil
}

func resourceLiteLLMAgent() *schema.Resource {
	return &schema.Resource{
		Create: resourceLiteLLMAgentCreate,
		Read:   resourceLiteLLMAgentRead,
		Update: resourceLiteLLMAgentUpdate,
		Delete: resourceLiteLLMAgentDelete,

		Importer: &schema.ResourceImporter{StateContext: schema.ImportStatePassthroughContext},

		Schema: map[string]*schema.Schema{
			"agent_name": {
				Type:        schema.TypeString,
				Required:    true,
				Description: "Name of the agent.",
			},
			"agent_card_params": {
				Type:             schema.TypeString,
				Required:         true,
				DiffSuppressFunc: agentSuppressEquivalentJSON,
				Description: "A2A agent card as a JSON object string (name, description, url, version, " +
					"capabilities, skills, ...). The proxy merges in LiteLLM-fronting fields, so the configured " +
					"value stays authoritative in state.",
			},
			"litellm_params": {
				Type:             schema.TypeString,
				Optional:         true,
				Sensitive:        true,
				DiffSuppressFunc: agentSuppressEquivalentJSON,
				Description: "LiteLLM-specific parameters as a JSON object string (may include model, api_key, ...). " +
					"Never read back from the API.",
			},
			"object_permission": {
				Type:             schema.TypeString,
				Optional:         true,
				DiffSuppressFunc: agentSuppressEquivalentJSON,
				Description: "Access control permissions as a JSON object string " +
					"(mcp_servers, mcp_access_groups, mcp_tool_permissions, models, agents).",
			},
			"static_headers": {
				Type:        schema.TypeMap,
				Optional:    true,
				Sensitive:   true,
				Elem:        &schema.Schema{Type: schema.TypeString},
				Description: "Static headers sent with agent requests (may hold tokens). Never read back from the API.",
			},
			"extra_headers": {
				Type:        schema.TypeList,
				Optional:    true,
				Elem:        &schema.Schema{Type: schema.TypeString},
				Description: "Names of incoming request headers to forward to the agent.",
			},
			"tpm_limit": {
				Type:     schema.TypeInt,
				Optional: true,
			},
			"rpm_limit": {
				Type:     schema.TypeInt,
				Optional: true,
			},
			"session_tpm_limit": {
				Type:     schema.TypeInt,
				Optional: true,
			},
			"session_rpm_limit": {
				Type:     schema.TypeInt,
				Optional: true,
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

func buildAgentData(d *schema.ResourceData) (map[string]interface{}, error) {
	card, err := agentParseJSONObject(d.Get("agent_card_params").(string), "agent_card_params")
	if err != nil {
		return nil, err
	}

	agentData := map[string]interface{}{
		"agent_name":        d.Get("agent_name").(string),
		"agent_card_params": card,
	}

	for _, key := range []string{"litellm_params", "object_permission"} {
		raw, ok := d.GetOk(key)
		if !ok || raw.(string) == "" {
			continue
		}
		obj, err := agentParseJSONObject(raw.(string), key)
		if err != nil {
			return nil, err
		}
		agentData[key] = obj
	}

	for _, key := range []string{"static_headers", "extra_headers", "tpm_limit", "rpm_limit", "session_tpm_limit", "session_rpm_limit"} {
		if v, ok := d.GetOk(key); ok {
			agentData[key] = v
		}
	}

	return agentData, nil
}

func resourceLiteLLMAgentCreate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	agentData, err := buildAgentData(d)
	if err != nil {
		return err
	}

	log.Printf("[DEBUG] Create agent request for: %s", d.Get("agent_name").(string))

	resp, err := MakeRequest(client, "POST", endpointAgents, agentData)
	if err != nil {
		return fmt.Errorf("error creating agent: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "creating agent"); err != nil {
		return err
	}

	var agentResp agentAPIResponse
	if err := json.NewDecoder(resp.Body).Decode(&agentResp); err != nil {
		return fmt.Errorf("error decoding create agent response: %w", err)
	}
	if agentResp.AgentID == "" {
		return fmt.Errorf("create agent response did not contain an agent_id")
	}

	d.SetId(agentResp.AgentID)
	log.Printf("[INFO] Agent created with ID: %s", agentResp.AgentID)

	return resourceLiteLLMAgentRead(d, m)
}

func resourceLiteLLMAgentRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	log.Printf("[INFO] Reading agent with ID: %s", d.Id())

	resp, err := MakeRequest(client, "GET", fmt.Sprintf(endpointAgentByID, d.Id()), nil)
	if err != nil {
		return fmt.Errorf("error reading agent: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		log.Printf("[WARN] Agent with ID %s not found, removing from state", d.Id())
		d.SetId("")
		return nil
	}

	if err := handleResponse(resp, "reading agent"); err != nil {
		return err
	}

	var agentResp agentAPIResponse
	if err := json.NewDecoder(resp.Body).Decode(&agentResp); err != nil {
		return fmt.Errorf("error decoding agent info response: %w", err)
	}

	d.Set("agent_name", agentResp.AgentName)

	// The proxy merges LiteLLM-fronting fields into the stored card, so the configured
	// JSON stays authoritative; only populate from the API when importing.
	if d.Get("agent_card_params").(string) == "" && agentResp.AgentCardParams != nil {
		cardJSON, err := json.Marshal(agentResp.AgentCardParams)
		if err != nil {
			return fmt.Errorf("error encoding agent_card_params: %w", err)
		}
		d.Set("agent_card_params", string(cardJSON))
	}
	if d.Get("object_permission").(string) == "" && agentResp.ObjectPermission != nil {
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
	d.Set("created_at", agentResp.CreatedAt)
	d.Set("updated_at", agentResp.UpdatedAt)
	d.Set("created_by", agentResp.CreatedBy)
	d.Set("updated_by", agentResp.UpdatedBy)

	log.Printf("[INFO] Successfully read agent with ID: %s", d.Id())
	return nil
}

func resourceLiteLLMAgentUpdate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	agentData, err := buildAgentData(d)
	if err != nil {
		return err
	}

	log.Printf("[DEBUG] Update agent request for ID: %s", d.Id())

	resp, err := MakeRequest(client, "PATCH", fmt.Sprintf(endpointAgentByID, d.Id()), agentData)
	if err != nil {
		return fmt.Errorf("error updating agent: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "updating agent"); err != nil {
		return err
	}

	log.Printf("[INFO] Successfully updated agent with ID: %s", d.Id())
	return resourceLiteLLMAgentRead(d, m)
}

func resourceLiteLLMAgentDelete(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	log.Printf("[INFO] Deleting agent with ID: %s", d.Id())

	resp, err := MakeRequest(client, "DELETE", fmt.Sprintf(endpointAgentByID, d.Id()), nil)
	if err != nil {
		return fmt.Errorf("error deleting agent: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNotFound {
		if err := handleResponse(resp, "deleting agent"); err != nil {
			return err
		}
	}

	log.Printf("[INFO] Successfully deleted agent with ID: %s", d.Id())
	d.SetId("")
	return nil
}
