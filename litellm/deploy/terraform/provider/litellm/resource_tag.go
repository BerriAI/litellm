package litellm

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

const (
	endpointTagNew    = "/tag/new"
	endpointTagInfo   = "/tag/info"
	endpointTagUpdate = "/tag/update"
	endpointTagDelete = "/tag/delete"
)

type tagBudgetTable struct {
	BudgetID            string   `json:"budget_id"`
	MaxBudget           *float64 `json:"max_budget"`
	SoftBudget          *float64 `json:"soft_budget"`
	MaxParallelRequests *int     `json:"max_parallel_requests"`
	TPMLimit            *int     `json:"tpm_limit"`
	RPMLimit            *int     `json:"rpm_limit"`
	BudgetDuration      string   `json:"budget_duration"`
}

type tagInfoEntry struct {
	Name               string          `json:"name"`
	Description        string          `json:"description"`
	Models             []string        `json:"models"`
	CreatedAt          string          `json:"created_at"`
	UpdatedAt          string          `json:"updated_at"`
	CreatedBy          string          `json:"created_by"`
	LitellmBudgetTable *tagBudgetTable `json:"litellm_budget_table"`
}

func resourceLiteLLMTag() *schema.Resource {
	return &schema.Resource{
		Create: resourceLiteLLMTagCreate,
		Read:   resourceLiteLLMTagRead,
		Update: resourceLiteLLMTagUpdate,
		Delete: resourceLiteLLMTagDelete,

		Importer: &schema.ResourceImporter{StateContext: schema.ImportStatePassthroughContext},

		Schema: map[string]*schema.Schema{
			"name": {
				Type:        schema.TypeString,
				Required:    true,
				ForceNew:    true,
				Description: "Unique name of the tag. Also used as the resource ID.",
			},
			"description": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "Description of the tag.",
			},
			"models": {
				Type:        schema.TypeList,
				Optional:    true,
				Elem:        &schema.Schema{Type: schema.TypeString},
				Description: "List of model IDs this tag applies to.",
			},
			"budget_id": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "Existing budget ID to associate with this tag.",
			},
			"max_budget": {
				Type:        schema.TypeFloat,
				Optional:    true,
				Description: "Max budget in USD for this tag.",
			},
			"soft_budget": {
				Type:        schema.TypeFloat,
				Optional:    true,
				Description: "Soft budget in USD for this tag.",
			},
			"max_parallel_requests": {
				Type:        schema.TypeInt,
				Optional:    true,
				Description: "Max concurrent requests allowed for this tag.",
			},
			"tpm_limit": {
				Type:        schema.TypeInt,
				Optional:    true,
				Description: "Max tokens per minute for this tag.",
			},
			"rpm_limit": {
				Type:        schema.TypeInt,
				Optional:    true,
				Description: "Max requests per minute for this tag.",
			},
			"budget_duration": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "Duration for budget reset (e.g. '1h', '1d', '30d').",
			},
			"model_max_budget": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "JSON object string with per-model budget configuration.",
			},
		},
	}
}

func buildTagData(d *schema.ResourceData, name string) (map[string]interface{}, error) {
	tagData := map[string]interface{}{
		"name": name,
	}

	for _, key := range []string{"description", "models", "budget_id", "max_budget", "soft_budget",
		"max_parallel_requests", "tpm_limit", "rpm_limit", "budget_duration"} {
		if v, ok := d.GetOk(key); ok {
			tagData[key] = v
		}
	}

	if v, ok := d.GetOk("model_max_budget"); ok {
		var modelMaxBudget map[string]interface{}
		if err := json.Unmarshal([]byte(v.(string)), &modelMaxBudget); err != nil {
			return nil, fmt.Errorf("model_max_budget must be a JSON object: %w", err)
		}
		tagData["model_max_budget"] = modelMaxBudget
	}

	return tagData, nil
}

// fetchTagInfo returns the tag entry, or gone=true when the proxy reports the tag missing.
func fetchTagInfo(client *Client, name string) (*tagInfoEntry, bool, error) {
	resp, err := MakeRequest(client, "POST", endpointTagInfo, map[string]interface{}{
		"names": []string{name},
	})
	if err != nil {
		return nil, false, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, false, fmt.Errorf("failed to read tag info response: %w", err)
	}

	if resp.StatusCode == http.StatusNotFound ||
		(resp.StatusCode != http.StatusOK && strings.Contains(string(body), "Tags not found")) {
		return nil, true, nil
	}
	if resp.StatusCode != http.StatusOK {
		return nil, false, fmt.Errorf("error reading tag: %s - %s", resp.Status, string(body))
	}

	var tags map[string]tagInfoEntry
	if err := json.Unmarshal(body, &tags); err != nil {
		return nil, false, fmt.Errorf("error decoding tag info response: %w", err)
	}

	entry, ok := tags[name]
	if !ok {
		return nil, true, nil
	}
	return &entry, false, nil
}

func resourceLiteLLMTagCreate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	name := d.Get("name").(string)
	tagData, err := buildTagData(d, name)
	if err != nil {
		return err
	}

	log.Printf("[DEBUG] Create tag request payload: %+v", tagData)

	resp, err := MakeRequest(client, "POST", endpointTagNew, tagData)
	if err != nil {
		return fmt.Errorf("error creating tag: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "creating tag"); err != nil {
		return err
	}

	d.SetId(name)
	log.Printf("[INFO] Tag created with name: %s", name)

	return resourceLiteLLMTagRead(d, m)
}

func resourceLiteLLMTagRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	log.Printf("[INFO] Reading tag with name: %s", d.Id())

	entry, gone, err := fetchTagInfo(client, d.Id())
	if err != nil {
		return err
	}
	if gone {
		log.Printf("[WARN] Tag %s not found, removing from state", d.Id())
		d.SetId("")
		return nil
	}

	d.Set("name", d.Id())
	d.Set("description", GetStringValue(entry.Description, d.Get("description").(string)))
	if entry.Models != nil {
		d.Set("models", entry.Models)
	}

	if bt := entry.LitellmBudgetTable; bt != nil {
		d.Set("budget_id", GetStringValue(bt.BudgetID, d.Get("budget_id").(string)))
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

	log.Printf("[INFO] Successfully read tag with name: %s", d.Id())
	return nil
}

func resourceLiteLLMTagUpdate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	tagData, err := buildTagData(d, d.Id())
	if err != nil {
		return err
	}
	log.Printf("[DEBUG] Update tag request payload: %+v", tagData)

	resp, err := MakeRequest(client, "POST", endpointTagUpdate, tagData)
	if err != nil {
		return fmt.Errorf("error updating tag: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "updating tag"); err != nil {
		return err
	}

	log.Printf("[INFO] Successfully updated tag with name: %s", d.Id())
	return resourceLiteLLMTagRead(d, m)
}

func resourceLiteLLMTagDelete(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	log.Printf("[INFO] Deleting tag with name: %s", d.Id())

	resp, err := MakeRequest(client, "POST", endpointTagDelete, map[string]interface{}{
		"name": d.Id(),
	})
	if err != nil {
		return fmt.Errorf("error deleting tag: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "deleting tag"); err != nil {
		return err
	}

	log.Printf("[INFO] Successfully deleted tag with name: %s", d.Id())
	d.SetId("")
	return nil
}
