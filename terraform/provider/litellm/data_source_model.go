package litellm

import (
	"encoding/json"
	"fmt"
	"log"
	"net/url"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

const endpointModelInfoV1 = "/v1/model/info"

// modelInfoParams intentionally maps only the non-sensitive litellm_params fields;
// credentials (api_key, aws_secret_access_key, ...) must never reach state.
type modelInfoParams struct {
	Model             string `json:"model"`
	CustomLLMProvider string `json:"custom_llm_provider"`
	APIBase           string `json:"api_base"`
	APIVersion        string `json:"api_version"`
	TPM               int    `json:"tpm"`
	RPM               int    `json:"rpm"`
}

type modelInfoMeta struct {
	ID        string `json:"id"`
	DBModel   bool   `json:"db_model"`
	BaseModel string `json:"base_model"`
	Tier      string `json:"tier"`
	Mode      string `json:"mode"`
	TeamID    string `json:"team_id"`
	CreatedAt string `json:"created_at"`
	UpdatedAt string `json:"updated_at"`
}

type modelInfoEntry struct {
	ModelName     string          `json:"model_name"`
	LiteLLMParams modelInfoParams `json:"litellm_params"`
	ModelInfo     modelInfoMeta   `json:"model_info"`
}

type modelInfoEnvelope struct {
	Data json.RawMessage `json:"data"`
}

// /v1/model/info returns data as a single object on the DB path and as a
// one-element list on the config path, so both shapes must be handled.
func modelDecodeInfoEntries(raw json.RawMessage) ([]modelInfoEntry, error) {
	var single modelInfoEntry
	if err := json.Unmarshal(raw, &single); err == nil {
		return []modelInfoEntry{single}, nil
	}
	var list []modelInfoEntry
	if err := json.Unmarshal(raw, &list); err != nil {
		return nil, fmt.Errorf("failed to decode model info data: %w", err)
	}
	return list, nil
}

func dataSourceLiteLLMModel() *schema.Resource {
	return &schema.Resource{
		Read: dataSourceLiteLLMModelRead,

		Schema: map[string]*schema.Schema{
			"model_id": {
				Type:        schema.TypeString,
				Required:    true,
				Description: "LiteLLM model ID (the x-litellm-model-id response header value)",
			},
			"model_name": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"model": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "The underlying litellm_params model, e.g. openai/gpt-4o",
			},
			"custom_llm_provider": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"model_api_base": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"api_version": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"tpm": {
				Type:     schema.TypeInt,
				Computed: true,
			},
			"rpm": {
				Type:     schema.TypeInt,
				Computed: true,
			},
			"base_model": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"tier": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"mode": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"team_id": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"db_model": {
				Type:     schema.TypeBool,
				Computed: true,
			},
		},
	}
}

func dataSourceLiteLLMModelRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)
	modelID := d.Get("model_id").(string)

	endpoint := fmt.Sprintf("%s?litellm_model_id=%s", endpointModelInfoV1, url.QueryEscape(modelID))
	resp, err := MakeRequest(client, "GET", endpoint, nil)
	if err != nil {
		return fmt.Errorf("failed to read model info: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "reading model info"); err != nil {
		return err
	}

	var envelope modelInfoEnvelope
	if err := json.NewDecoder(resp.Body).Decode(&envelope); err != nil {
		return fmt.Errorf("failed to decode model info response: %w", err)
	}

	entries, err := modelDecodeInfoEntries(envelope.Data)
	if err != nil {
		return err
	}
	if len(entries) == 0 {
		return fmt.Errorf("model with id %q not found", modelID)
	}
	entry := entries[0]

	d.SetId(GetStringValue(entry.ModelInfo.ID, modelID))
	d.Set("model_name", entry.ModelName)
	d.Set("model", entry.LiteLLMParams.Model)
	d.Set("custom_llm_provider", entry.LiteLLMParams.CustomLLMProvider)
	d.Set("model_api_base", entry.LiteLLMParams.APIBase)
	d.Set("api_version", entry.LiteLLMParams.APIVersion)
	d.Set("tpm", entry.LiteLLMParams.TPM)
	d.Set("rpm", entry.LiteLLMParams.RPM)
	d.Set("base_model", entry.ModelInfo.BaseModel)
	d.Set("tier", entry.ModelInfo.Tier)
	d.Set("mode", entry.ModelInfo.Mode)
	d.Set("team_id", entry.ModelInfo.TeamID)
	d.Set("db_model", entry.ModelInfo.DBModel)

	log.Printf("[INFO] Successfully read model with ID: %s", modelID)
	return nil
}

func dataSourceLiteLLMModels() *schema.Resource {
	return &schema.Resource{
		Read: dataSourceLiteLLMModelsRead,

		Schema: map[string]*schema.Schema{
			"team_id": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "Filter models to those accessible by this team",
			},
			"ids": {
				Type:        schema.TypeList,
				Computed:    true,
				Elem:        &schema.Schema{Type: schema.TypeString},
				Description: "LiteLLM model IDs of the returned models",
			},
			"models": {
				Type:     schema.TypeList,
				Computed: true,
				Elem: &schema.Resource{
					Schema: map[string]*schema.Schema{
						"id":                  {Type: schema.TypeString, Computed: true},
						"model_name":          {Type: schema.TypeString, Computed: true},
						"model":               {Type: schema.TypeString, Computed: true},
						"custom_llm_provider": {Type: schema.TypeString, Computed: true},
						"model_api_base":      {Type: schema.TypeString, Computed: true},
						"base_model":          {Type: schema.TypeString, Computed: true},
						"tier":                {Type: schema.TypeString, Computed: true},
						"mode":                {Type: schema.TypeString, Computed: true},
						"team_id":             {Type: schema.TypeString, Computed: true},
						"db_model":            {Type: schema.TypeBool, Computed: true},
					},
				},
			},
		},
	}
}

func dataSourceLiteLLMModelsRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	endpoint := endpointModelInfoV1
	if v, ok := d.GetOk("team_id"); ok {
		endpoint = fmt.Sprintf("%s?teamId=%s", endpointModelInfoV1, url.QueryEscape(v.(string)))
	}

	resp, err := MakeRequest(client, "GET", endpoint, nil)
	if err != nil {
		return fmt.Errorf("failed to list models: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "listing models"); err != nil {
		return err
	}

	var envelope modelInfoEnvelope
	if err := json.NewDecoder(resp.Body).Decode(&envelope); err != nil {
		return fmt.Errorf("failed to decode model list response: %w", err)
	}

	entries, err := modelDecodeInfoEntries(envelope.Data)
	if err != nil {
		return err
	}

	ids := make([]string, 0, len(entries))
	models := make([]map[string]interface{}, 0, len(entries))
	for _, entry := range entries {
		ids = append(ids, entry.ModelInfo.ID)
		models = append(models, map[string]interface{}{
			"id":                  entry.ModelInfo.ID,
			"model_name":          entry.ModelName,
			"model":               entry.LiteLLMParams.Model,
			"custom_llm_provider": entry.LiteLLMParams.CustomLLMProvider,
			"model_api_base":      entry.LiteLLMParams.APIBase,
			"base_model":          entry.ModelInfo.BaseModel,
			"tier":                entry.ModelInfo.Tier,
			"mode":                entry.ModelInfo.Mode,
			"team_id":             entry.ModelInfo.TeamID,
			"db_model":            entry.ModelInfo.DBModel,
		})
	}

	d.SetId(GetStringValue(d.Get("team_id").(string), "all"))
	d.Set("ids", ids)
	d.Set("models", models)

	log.Printf("[INFO] Successfully listed %d models", len(models))
	return nil
}
