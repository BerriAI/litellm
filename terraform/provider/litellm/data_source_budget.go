package litellm

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

const endpointBudgetList = "/budget/list"

func dataSourceLiteLLMBudget() *schema.Resource {
	return &schema.Resource{
		Read: dataSourceLiteLLMBudgetRead,

		Schema: map[string]*schema.Schema{
			"budget_id": {
				Type:        schema.TypeString,
				Required:    true,
				Description: "ID of the budget to retrieve",
			},
			"max_budget": {
				Type:        schema.TypeFloat,
				Computed:    true,
				Description: "Hard budget limit in USD",
			},
			"soft_budget": {
				Type:        schema.TypeFloat,
				Computed:    true,
				Description: "Soft budget limit in USD that triggers alerts",
			},
			"max_parallel_requests": {
				Type:        schema.TypeInt,
				Computed:    true,
				Description: "Maximum concurrent requests allowed for this budget",
			},
			"tpm_limit": {
				Type:        schema.TypeInt,
				Computed:    true,
				Description: "Maximum tokens per minute allowed for this budget",
			},
			"rpm_limit": {
				Type:        schema.TypeInt,
				Computed:    true,
				Description: "Maximum requests per minute allowed for this budget",
			},
			"budget_duration": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Budget reset period",
			},
			"model_max_budget": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "JSON string of per-model budget config",
			},
			"budget_reset_at": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Datetime when the budget is reset",
			},
		},
	}
}

func dataSourceLiteLLMBudgetRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)
	budgetID := d.Get("budget_id").(string)

	resp, err := MakeRequest(client, "POST", endpointBudgetInfo, map[string]interface{}{
		"budgets": []string{budgetID},
	})
	if err != nil {
		return fmt.Errorf("failed to read budget: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return fmt.Errorf("budget '%s' not found", budgetID)
	}

	if err := handleResponse(resp, "reading budget"); err != nil {
		return err
	}

	var budgetResps []budgetResponse
	if err := json.NewDecoder(resp.Body).Decode(&budgetResps); err != nil {
		return fmt.Errorf("error decoding budget info response: %w", err)
	}
	if len(budgetResps) == 0 {
		return fmt.Errorf("budget '%s' not found", budgetID)
	}

	d.SetId(budgetID)
	setBudgetState(d, budgetResps[0])

	return nil
}

func dataSourceLiteLLMBudgets() *schema.Resource {
	return &schema.Resource{
		Read: dataSourceLiteLLMBudgetsRead,

		Schema: map[string]*schema.Schema{
			"budgets": {
				Type:        schema.TypeList,
				Computed:    true,
				Description: "All budgets configured on the proxy",
				Elem: &schema.Resource{
					Schema: map[string]*schema.Schema{
						"budget_id":             {Type: schema.TypeString, Computed: true},
						"max_budget":            {Type: schema.TypeFloat, Computed: true},
						"soft_budget":           {Type: schema.TypeFloat, Computed: true},
						"max_parallel_requests": {Type: schema.TypeInt, Computed: true},
						"tpm_limit":             {Type: schema.TypeInt, Computed: true},
						"rpm_limit":             {Type: schema.TypeInt, Computed: true},
						"budget_duration":       {Type: schema.TypeString, Computed: true},
						"model_max_budget":      {Type: schema.TypeString, Computed: true},
						"budget_reset_at":       {Type: schema.TypeString, Computed: true},
					},
				},
			},
			"ids": {
				Type:        schema.TypeList,
				Computed:    true,
				Elem:        &schema.Schema{Type: schema.TypeString},
				Description: "IDs of all budgets configured on the proxy",
			},
		},
	}
}

func budgetListEntry(budgetResp budgetResponse) map[string]interface{} {
	entry := map[string]interface{}{
		"budget_id": budgetResp.BudgetID,
	}
	if budgetResp.MaxBudget != nil {
		entry["max_budget"] = *budgetResp.MaxBudget
	}
	if budgetResp.SoftBudget != nil {
		entry["soft_budget"] = *budgetResp.SoftBudget
	}
	if budgetResp.MaxParallelRequests != nil {
		entry["max_parallel_requests"] = *budgetResp.MaxParallelRequests
	}
	if budgetResp.TPMLimit != nil {
		entry["tpm_limit"] = *budgetResp.TPMLimit
	}
	if budgetResp.RPMLimit != nil {
		entry["rpm_limit"] = *budgetResp.RPMLimit
	}
	if budgetResp.BudgetDuration != nil {
		entry["budget_duration"] = *budgetResp.BudgetDuration
	}
	if encoded, ok := budgetModelMaxBudgetString(budgetResp.ModelMaxBudget); ok {
		entry["model_max_budget"] = encoded
	}
	if budgetResp.BudgetResetAt != nil {
		entry["budget_reset_at"] = *budgetResp.BudgetResetAt
	}
	return entry
}

func dataSourceLiteLLMBudgetsRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	resp, err := MakeRequest(client, "GET", endpointBudgetList, nil)
	if err != nil {
		return fmt.Errorf("failed to list budgets: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "listing budgets"); err != nil {
		return err
	}

	var budgetResps []budgetResponse
	if err := json.NewDecoder(resp.Body).Decode(&budgetResps); err != nil {
		return fmt.Errorf("error decoding budget list response: %w", err)
	}

	budgets := make([]map[string]interface{}, 0, len(budgetResps))
	ids := make([]string, 0, len(budgetResps))
	for _, budgetResp := range budgetResps {
		budgets = append(budgets, budgetListEntry(budgetResp))
		ids = append(ids, budgetResp.BudgetID)
	}

	d.SetId("budgets")
	d.Set("budgets", budgets)
	d.Set("ids", ids)

	return nil
}
