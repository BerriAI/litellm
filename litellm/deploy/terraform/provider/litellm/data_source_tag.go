package litellm

import (
	"encoding/json"
	"fmt"
	"net/url"
	"strings"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

const endpointTagList = "/tag/list"

func dataSourceLiteLLMTag() *schema.Resource {
	return &schema.Resource{
		Read: dataSourceLiteLLMTagRead,

		Schema: map[string]*schema.Schema{
			"name": {
				Type:        schema.TypeString,
				Required:    true,
				Description: "Name of the tag to retrieve",
			},
			"description": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Description of the tag",
			},
			"models": {
				Type:        schema.TypeList,
				Computed:    true,
				Elem:        &schema.Schema{Type: schema.TypeString},
				Description: "Model IDs this tag applies to",
			},
			"budget_id": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Budget ID associated with this tag",
			},
			"max_budget": {
				Type:        schema.TypeFloat,
				Computed:    true,
				Description: "Max budget in USD for this tag",
			},
			"soft_budget": {
				Type:        schema.TypeFloat,
				Computed:    true,
				Description: "Soft budget in USD for this tag",
			},
			"max_parallel_requests": {
				Type:        schema.TypeInt,
				Computed:    true,
				Description: "Max concurrent requests allowed for this tag",
			},
			"tpm_limit": {
				Type:        schema.TypeInt,
				Computed:    true,
				Description: "Max tokens per minute for this tag",
			},
			"rpm_limit": {
				Type:        schema.TypeInt,
				Computed:    true,
				Description: "Max requests per minute for this tag",
			},
			"budget_duration": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Duration for budget reset",
			},
			"created_at": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Timestamp when the tag was created",
			},
			"updated_at": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Timestamp when the tag was last updated",
			},
			"created_by": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "User that created the tag",
			},
		},
	}
}

func dataSourceLiteLLMTagRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)
	name := d.Get("name").(string)

	entry, gone, err := fetchTagInfo(client, name)
	if err != nil {
		return fmt.Errorf("failed to read tag: %w", err)
	}
	if gone {
		return fmt.Errorf("tag '%s' not found", name)
	}

	d.SetId(name)
	d.Set("description", entry.Description)
	d.Set("models", entry.Models)
	d.Set("created_at", entry.CreatedAt)
	d.Set("updated_at", entry.UpdatedAt)
	d.Set("created_by", entry.CreatedBy)

	if bt := entry.LitellmBudgetTable; bt != nil {
		d.Set("budget_id", bt.BudgetID)
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
		d.Set("budget_duration", bt.BudgetDuration)
	}

	return nil
}

func dataSourceLiteLLMTags() *schema.Resource {
	return &schema.Resource{
		Read: dataSourceLiteLLMTagsRead,

		Schema: map[string]*schema.Schema{
			"start_date": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "Optional start date (YYYY-MM-DD) limiting dynamic tags to those active in the window",
			},
			"end_date": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "Optional end date (YYYY-MM-DD), must be given with start_date",
			},
			"ids": {
				Type:        schema.TypeList,
				Computed:    true,
				Elem:        &schema.Schema{Type: schema.TypeString},
				Description: "Names of all tags (tag names are their IDs)",
			},
			"tags": {
				Type:        schema.TypeList,
				Computed:    true,
				Description: "List of tags",
				Elem: &schema.Resource{
					Schema: map[string]*schema.Schema{
						"name":        {Type: schema.TypeString, Computed: true},
						"description": {Type: schema.TypeString, Computed: true},
						"models": {
							Type:     schema.TypeList,
							Computed: true,
							Elem:     &schema.Schema{Type: schema.TypeString},
						},
						"budget_id":             {Type: schema.TypeString, Computed: true},
						"max_budget":            {Type: schema.TypeFloat, Computed: true},
						"soft_budget":           {Type: schema.TypeFloat, Computed: true},
						"max_parallel_requests": {Type: schema.TypeInt, Computed: true},
						"tpm_limit":             {Type: schema.TypeInt, Computed: true},
						"rpm_limit":             {Type: schema.TypeInt, Computed: true},
						"budget_duration":       {Type: schema.TypeString, Computed: true},
						"created_at":            {Type: schema.TypeString, Computed: true},
						"updated_at":            {Type: schema.TypeString, Computed: true},
						"created_by":            {Type: schema.TypeString, Computed: true},
					},
				},
			},
		},
	}
}

func dataSourceLiteLLMTagsRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	endpoint := endpointTagList
	if startDate, ok := d.GetOk("start_date"); ok {
		endpoint = fmt.Sprintf("%s?start_date=%s&end_date=%s", endpointTagList,
			url.QueryEscape(startDate.(string)), url.QueryEscape(d.Get("end_date").(string)))
	}

	resp, err := MakeRequest(client, "GET", endpoint, nil)
	if err != nil {
		return fmt.Errorf("failed to list tags: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "listing tags"); err != nil {
		return err
	}

	var entries []tagInfoEntry
	if err := json.NewDecoder(resp.Body).Decode(&entries); err != nil {
		return fmt.Errorf("error decoding tag list response: %w", err)
	}

	ids := make([]string, 0, len(entries))
	tags := make([]map[string]interface{}, 0, len(entries))
	for _, entry := range entries {
		ids = append(ids, entry.Name)

		tag := map[string]interface{}{
			"name":        entry.Name,
			"description": entry.Description,
			"models":      entry.Models,
			"created_at":  entry.CreatedAt,
			"updated_at":  entry.UpdatedAt,
			"created_by":  entry.CreatedBy,
		}
		if bt := entry.LitellmBudgetTable; bt != nil {
			tag["budget_id"] = bt.BudgetID
			tag["budget_duration"] = bt.BudgetDuration
			if bt.MaxBudget != nil {
				tag["max_budget"] = *bt.MaxBudget
			}
			if bt.SoftBudget != nil {
				tag["soft_budget"] = *bt.SoftBudget
			}
			if bt.MaxParallelRequests != nil {
				tag["max_parallel_requests"] = *bt.MaxParallelRequests
			}
			if bt.TPMLimit != nil {
				tag["tpm_limit"] = *bt.TPMLimit
			}
			if bt.RPMLimit != nil {
				tag["rpm_limit"] = *bt.RPMLimit
			}
		}
		tags = append(tags, tag)
	}

	d.SetId(strings.Join([]string{"litellm-tags", d.Get("start_date").(string), d.Get("end_date").(string)}, "-"))
	d.Set("ids", ids)
	d.Set("tags", tags)

	return nil
}
