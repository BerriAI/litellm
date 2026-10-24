package litellm

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

const endpointGuardrailList = "/guardrails/list"

func dataSourceLiteLLMGuardrail() *schema.Resource {
	return &schema.Resource{
		Read: dataSourceLiteLLMGuardrailRead,

		Schema: map[string]*schema.Schema{
			"guardrail_id": {
				Type:        schema.TypeString,
				Required:    true,
				Description: "Unique identifier of the guardrail to retrieve",
			},
			"guardrail_name": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"guardrail_info": {
				Type:     schema.TypeMap,
				Computed: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
			"guardrail_definition_location": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Where the guardrail is defined: 'config' or 'db'",
			},
			"created_at": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"updated_at": {
				Type:     schema.TypeString,
				Computed: true,
			},
		},
	}
}

type guardrailListItemAPIResponse struct {
	GuardrailID                 string                 `json:"guardrail_id"`
	GuardrailName               string                 `json:"guardrail_name"`
	GuardrailInfo               map[string]interface{} `json:"guardrail_info"`
	GuardrailDefinitionLocation string                 `json:"guardrail_definition_location"`
	CreatedAt                   string                 `json:"created_at"`
	UpdatedAt                   string                 `json:"updated_at"`
}

func dataSourceLiteLLMGuardrailRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)
	guardrailID := d.Get("guardrail_id").(string)

	resp, err := MakeRequest(client, "GET", fmt.Sprintf(endpointGuardrailInfo, guardrailID), nil)
	if err != nil {
		return fmt.Errorf("failed to read guardrail: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return fmt.Errorf("guardrail '%s' not found", guardrailID)
	}

	if err := handleResponse(resp, "reading guardrail"); err != nil {
		return err
	}

	var info guardrailListItemAPIResponse
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return fmt.Errorf("error decoding guardrail info response: %w", err)
	}

	d.SetId(guardrailID)
	d.Set("guardrail_name", info.GuardrailName)
	d.Set("guardrail_info", guardrailInfoToStringMap(info.GuardrailInfo))
	d.Set("guardrail_definition_location", info.GuardrailDefinitionLocation)
	d.Set("created_at", info.CreatedAt)
	d.Set("updated_at", info.UpdatedAt)
	// litellm_params is intentionally not exposed: it can carry API keys.

	return nil
}

func dataSourceLiteLLMGuardrails() *schema.Resource {
	return &schema.Resource{
		Read: dataSourceLiteLLMGuardrailsRead,

		Schema: map[string]*schema.Schema{
			"guardrails": {
				Type:     schema.TypeList,
				Computed: true,
				Elem: &schema.Resource{
					Schema: map[string]*schema.Schema{
						"guardrail_id": {
							Type:     schema.TypeString,
							Computed: true,
						},
						"guardrail_name": {
							Type:     schema.TypeString,
							Computed: true,
						},
						"guardrail_info": {
							Type:     schema.TypeMap,
							Computed: true,
							Elem:     &schema.Schema{Type: schema.TypeString},
						},
						"guardrail_definition_location": {
							Type:     schema.TypeString,
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

func dataSourceLiteLLMGuardrailsRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	resp, err := MakeRequest(client, "GET", endpointGuardrailList, nil)
	if err != nil {
		return fmt.Errorf("failed to list guardrails: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "listing guardrails"); err != nil {
		return err
	}

	var listResp struct {
		Guardrails []guardrailListItemAPIResponse `json:"guardrails"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&listResp); err != nil {
		return fmt.Errorf("error decoding guardrails list response: %w", err)
	}

	guardrails := make([]map[string]interface{}, 0, len(listResp.Guardrails))
	ids := make([]string, 0, len(listResp.Guardrails))
	for _, g := range listResp.Guardrails {
		guardrails = append(guardrails, map[string]interface{}{
			"guardrail_id":                  g.GuardrailID,
			"guardrail_name":                g.GuardrailName,
			"guardrail_info":                guardrailInfoToStringMap(g.GuardrailInfo),
			"guardrail_definition_location": g.GuardrailDefinitionLocation,
			"created_at":                    g.CreatedAt,
			"updated_at":                    g.UpdatedAt,
		})
		ids = append(ids, g.GuardrailID)
	}

	d.SetId("guardrails")
	d.Set("guardrails", guardrails)
	d.Set("ids", ids)

	return nil
}
