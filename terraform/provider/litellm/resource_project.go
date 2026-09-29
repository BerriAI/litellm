package litellm

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

const (
	endpointProjectNew    = "/project/new"
	endpointProjectInfo   = "/project/info"
	endpointProjectUpdate = "/project/update"
	endpointProjectDelete = "/project/delete"
)

type projectBudgetTable struct {
	MaxBudget           *float64 `json:"max_budget"`
	SoftBudget          *float64 `json:"soft_budget"`
	MaxParallelRequests *int     `json:"max_parallel_requests"`
	TPMLimit            *int     `json:"tpm_limit"`
	RPMLimit            *int     `json:"rpm_limit"`
	BudgetDuration      string   `json:"budget_duration"`
}

type projectResponse struct {
	ProjectID          string                 `json:"project_id"`
	ProjectAlias       string                 `json:"project_alias"`
	Description        string                 `json:"description"`
	TeamID             string                 `json:"team_id"`
	BudgetID           string                 `json:"budget_id"`
	Metadata           map[string]interface{} `json:"metadata"`
	Models             []string               `json:"models"`
	Spend              float64                `json:"spend"`
	Blocked            bool                   `json:"blocked"`
	CreatedBy          string                 `json:"created_by"`
	UpdatedBy          string                 `json:"updated_by"`
	CreatedAt          string                 `json:"created_at"`
	UpdatedAt          string                 `json:"updated_at"`
	LitellmBudgetTable *projectBudgetTable    `json:"litellm_budget_table"`
}

func resourceLiteLLMProject() *schema.Resource {
	return &schema.Resource{
		Create: resourceLiteLLMProjectCreate,
		Read:   resourceLiteLLMProjectRead,
		Update: resourceLiteLLMProjectUpdate,
		Delete: resourceLiteLLMProjectDelete,

		Importer: &schema.ResourceImporter{StateContext: schema.ImportStatePassthroughContext},

		Schema: map[string]*schema.Schema{
			"team_id": {
				Type:        schema.TypeString,
				Required:    true,
				ForceNew:    true,
				Description: "The team ID this project belongs to.",
			},
			"project_alias": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "Human-friendly name for the project.",
			},
			"description": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "Description of the project's purpose and use case.",
			},
			"models": {
				Type:        schema.TypeList,
				Optional:    true,
				Elem:        &schema.Schema{Type: schema.TypeString},
				Description: "List of models the project can access.",
			},
			"metadata": {
				Type:        schema.TypeMap,
				Optional:    true,
				Elem:        &schema.Schema{Type: schema.TypeString},
				Description: "Metadata for the project.",
			},
			"tags": {
				Type:        schema.TypeList,
				Optional:    true,
				Elem:        &schema.Schema{Type: schema.TypeString},
				Description: "Tags associated with the project.",
			},
			"max_budget": {
				Type:        schema.TypeFloat,
				Optional:    true,
				Description: "Maximum budget for this project.",
			},
			"soft_budget": {
				Type:        schema.TypeFloat,
				Optional:    true,
				Description: "Soft budget limit for warnings.",
			},
			"budget_duration": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "Budget reset duration (e.g. '30d', '1h').",
			},
			"budget_id": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "Budget ID to associate with this project.",
			},
			"tpm_limit": {
				Type:        schema.TypeInt,
				Optional:    true,
				Description: "Tokens per minute limit.",
			},
			"rpm_limit": {
				Type:        schema.TypeInt,
				Optional:    true,
				Description: "Requests per minute limit.",
			},
			"max_parallel_requests": {
				Type:        schema.TypeInt,
				Optional:    true,
				Description: "Maximum parallel requests allowed.",
			},
			"model_max_budget": {
				Type:        schema.TypeMap,
				Optional:    true,
				Elem:        &schema.Schema{Type: schema.TypeFloat},
				Description: "Per-model budget limits.",
			},
			"model_rpm_limit": {
				Type:        schema.TypeMap,
				Optional:    true,
				Elem:        &schema.Schema{Type: schema.TypeInt},
				Description: "Per-model RPM limits.",
			},
			"model_tpm_limit": {
				Type:        schema.TypeMap,
				Optional:    true,
				Elem:        &schema.Schema{Type: schema.TypeInt},
				Description: "Per-model TPM limits.",
			},
			"blocked": {
				Type:        schema.TypeBool,
				Optional:    true,
				Description: "Whether the project is blocked from making requests.",
			},
			"spend": {
				Type:        schema.TypeFloat,
				Computed:    true,
				Description: "Current spend for the project.",
			},
			"created_at": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Timestamp when the project was created.",
			},
			"updated_at": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Timestamp when the project was last updated.",
			},
			"created_by": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "User that created the project.",
			},
			"updated_by": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "User that last updated the project.",
			},
		},
	}
}

func buildProjectData(d *schema.ResourceData) map[string]interface{} {
	projectData := map[string]interface{}{
		"team_id": d.Get("team_id").(string),
	}

	for _, key := range []string{"project_alias", "description", "models", "metadata", "tags",
		"max_budget", "soft_budget", "budget_duration", "budget_id", "tpm_limit", "rpm_limit",
		"max_parallel_requests", "model_max_budget", "model_rpm_limit", "model_tpm_limit", "blocked"} {
		if v, ok := d.GetOk(key); ok {
			projectData[key] = v
		}
	}

	return projectData
}

func resourceLiteLLMProjectCreate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	projectData := buildProjectData(d)
	log.Printf("[DEBUG] Create project request payload: %+v", projectData)

	resp, err := MakeRequest(client, "POST", endpointProjectNew, projectData)
	if err != nil {
		return fmt.Errorf("error creating project: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("error reading create project response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("error creating project: %s - %s", resp.Status, string(body))
	}

	var projResp projectResponse
	if err := json.Unmarshal(body, &projResp); err != nil {
		return fmt.Errorf("error decoding create project response: %w", err)
	}
	if projResp.ProjectID == "" {
		return fmt.Errorf("create project response did not contain a project_id: %s", string(body))
	}

	d.SetId(projResp.ProjectID)
	log.Printf("[INFO] Project created with ID: %s", projResp.ProjectID)

	return resourceLiteLLMProjectRead(d, m)
}

func resourceLiteLLMProjectRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	log.Printf("[INFO] Reading project with ID: %s", d.Id())

	resp, err := MakeRequest(client, "GET", fmt.Sprintf("%s?project_id=%s", endpointProjectInfo, d.Id()), nil)
	if err != nil {
		return fmt.Errorf("error reading project: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		log.Printf("[WARN] Project with ID %s not found, removing from state", d.Id())
		d.SetId("")
		return nil
	}

	if err := handleResponse(resp, "reading project"); err != nil {
		return err
	}

	var projResp projectResponse
	if err := json.NewDecoder(resp.Body).Decode(&projResp); err != nil {
		return fmt.Errorf("error decoding project info response: %w", err)
	}

	d.Set("team_id", GetStringValue(projResp.TeamID, d.Get("team_id").(string)))
	d.Set("project_alias", GetStringValue(projResp.ProjectAlias, d.Get("project_alias").(string)))
	d.Set("description", GetStringValue(projResp.Description, d.Get("description").(string)))
	d.Set("budget_id", GetStringValue(projResp.BudgetID, d.Get("budget_id").(string)))
	if projResp.Models != nil {
		d.Set("models", projResp.Models)
	}
	setProjectMetadataAndTags(d, projResp.Metadata)

	d.Set("blocked", projResp.Blocked)
	d.Set("spend", projResp.Spend)
	d.Set("created_at", projResp.CreatedAt)
	d.Set("updated_at", projResp.UpdatedAt)
	d.Set("created_by", projResp.CreatedBy)
	d.Set("updated_by", projResp.UpdatedBy)

	if bt := projResp.LitellmBudgetTable; bt != nil {
		if bt.MaxBudget != nil {
			d.Set("max_budget", *bt.MaxBudget)
		}
		if bt.SoftBudget != nil {
			d.Set("soft_budget", *bt.SoftBudget)
		}
		if bt.MaxParallelRequests != nil {
			d.Set("max_parallel_requests", *bt.MaxParallelRequests)
		}
		if bt.TPMLimit != nil {
			d.Set("tpm_limit", *bt.TPMLimit)
		}
		if bt.RPMLimit != nil {
			d.Set("rpm_limit", *bt.RPMLimit)
		}
		d.Set("budget_duration", GetStringValue(bt.BudgetDuration, d.Get("budget_duration").(string)))
	}

	log.Printf("[INFO] Successfully read project with ID: %s", d.Id())
	return nil
}

// The proxy stores project tags inside metadata; split them back out so state matches the config shape.
func setProjectMetadataAndTags(d *schema.ResourceData, metadata map[string]interface{}) {
	if metadata == nil {
		return
	}

	if tags, ok := metadata["tags"].([]interface{}); ok {
		d.Set("tags", tags)
	}

	stringMetadata := map[string]interface{}{}
	for k, v := range metadata {
		if s, ok := v.(string); ok {
			stringMetadata[k] = s
		}
	}
	d.Set("metadata", stringMetadata)
}

func resourceLiteLLMProjectUpdate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	projectData := buildProjectData(d)
	projectData["project_id"] = d.Id()
	log.Printf("[DEBUG] Update project request payload: %+v", projectData)

	resp, err := MakeRequest(client, "POST", endpointProjectUpdate, projectData)
	if err != nil {
		return fmt.Errorf("error updating project: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "updating project"); err != nil {
		return err
	}

	log.Printf("[INFO] Successfully updated project with ID: %s", d.Id())
	return resourceLiteLLMProjectRead(d, m)
}

func resourceLiteLLMProjectDelete(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	log.Printf("[INFO] Deleting project with ID: %s", d.Id())

	resp, err := MakeRequest(client, "DELETE", endpointProjectDelete, map[string]interface{}{
		"project_ids": []string{d.Id()},
	})
	if err != nil {
		return fmt.Errorf("error deleting project: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "deleting project"); err != nil {
		return err
	}

	log.Printf("[INFO] Successfully deleted project with ID: %s", d.Id())
	d.SetId("")
	return nil
}
