package litellm

import (
	"context"
	"fmt"

	"github.com/hashicorp/go-cty/cty"
	"github.com/hashicorp/terraform-plugin-sdk/v2/diag"
	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func resourceKey() *schema.Resource {
	return &schema.Resource{
		CreateContext: resourceKeyCreate,
		ReadContext:   resourceKeyRead,
		UpdateContext: resourceKeyUpdate,
		DeleteContext: resourceKeyDelete,
		Importer: &schema.ResourceImporter{
			StateContext: schema.ImportStatePassthroughContext,
		},
		Schema: map[string]*schema.Schema{
			"key": {
				Type:      schema.TypeString,
				Optional:  true,
				WriteOnly: true,
				Sensitive: true,
			},
			"token_id": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"models": {
				Type:     schema.TypeList,
				Optional: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
			"max_budget": {
				Type:     schema.TypeFloat,
				Optional: true,
				Computed: true,
			},
			"user_id": {
				Type:     schema.TypeString,
				Optional: true,
			},
			"team_id": {
				Type:     schema.TypeString,
				Optional: true,
			},
			"max_parallel_requests": {
				Type:     schema.TypeInt,
				Optional: true,
				Computed: true,
			},
			"metadata": {
				Type:     schema.TypeMap,
				Optional: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
			"tpm_limit": {
				Type:     schema.TypeInt,
				Optional: true,
				Computed: true,
			},
			"rpm_limit": {
				Type:     schema.TypeInt,
				Optional: true,
				Computed: true,
			},
			"budget_duration": {
				Type:     schema.TypeString,
				Optional: true,
			},
			"allowed_cache_controls": {
				Type:     schema.TypeList,
				Optional: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
			"soft_budget": {
				Type:     schema.TypeFloat,
				Optional: true,
				Computed: true,
			},
			"key_alias": {
				Type:     schema.TypeString,
				Optional: true,
			},
			"duration": {
				Type:     schema.TypeString,
				Optional: true,
			},
			"aliases": {
				Type:     schema.TypeMap,
				Optional: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
			"config": {
				Type:     schema.TypeMap,
				Optional: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
			"permissions": {
				Type:     schema.TypeMap,
				Optional: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
			"model_max_budget": {
				Type:     schema.TypeMap,
				Optional: true,
				Elem:     &schema.Schema{Type: schema.TypeFloat, Computed: true},
			},
			"model_rpm_limit": {
				Type:     schema.TypeMap,
				Optional: true,
				Elem:     &schema.Schema{Type: schema.TypeInt, Computed: true},
			},
			"model_tpm_limit": {
				Type:     schema.TypeMap,
				Optional: true,
				Elem:     &schema.Schema{Type: schema.TypeInt, Computed: true},
			},
			"guardrails": {
				Type:     schema.TypeList,
				Optional: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
			"blocked": {
				Type:     schema.TypeBool,
				Optional: true,
			},
			"tags": {
				Type:     schema.TypeList,
				Optional: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
			"spend": {
				Type:     schema.TypeFloat,
				Computed: true,
			},
			"budget_id": {
				Type:     schema.TypeString,
				Optional: true,
			},
			"enforced_params": {
				Type:     schema.TypeList,
				Optional: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
			"allowed_routes": {
				Type:     schema.TypeList,
				Optional: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
			"allowed_passthrough_routes": {
				Type:     schema.TypeList,
				Optional: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
			"rpm_limit_type": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "One of 'guaranteed_throughput', 'best_effort_throughput' or 'dynamic'",
			},
			"tpm_limit_type": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "One of 'guaranteed_throughput', 'best_effort_throughput' or 'dynamic'",
			},
			"prompts": {
				Type:     schema.TypeList,
				Optional: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
			"organization_id": {
				Type:     schema.TypeString,
				Optional: true,
			},
			"project_id": {
				Type:     schema.TypeString,
				Optional: true,
				ForceNew: true,
			},
		},
	}
}

func resourceKeyCreate(ctx context.Context, d *schema.ResourceData, m interface{}) diag.Diagnostics {
	c := m.(*Client)

	key := &Key{}
	mapResourceDataToKey(d, key)
	// A config-supplied key value becomes the key itself; when absent the
	// proxy generates one. Write-only attributes are invisible to d.Get in
	// real Terraform runs, so read the raw config first.
	if raw, err := d.GetRawConfigAt(cty.GetAttrPath("key")); err == nil && !raw.IsNull() && raw.Type() == cty.String && raw.AsString() != "" {
		key.Key = raw.AsString()
	} else if v := d.Get("key").(string); v != "" {
		key.Key = v
	}

	createdKey, err := c.CreateKey(key)
	if err != nil {
		return diag.FromErr(fmt.Errorf("error creating key: %s", err))
	}

	d.SetId(createdKey.TokenID)
	// Set the write-only key value so it's available during this apply
	// but will not be persisted to state.
	d.Set("key", createdKey.Key)
	return resourceKeyRead(ctx, d, m)
}

func resourceKeyRead(ctx context.Context, d *schema.ResourceData, m interface{}) diag.Diagnostics {
	c := m.(*Client)

	key, err := c.GetKey(d.Id())
	if err != nil {
		return diag.FromErr(fmt.Errorf("error reading key: %s", err))
	}

	if key == nil {
		d.SetId("")
		return nil
	}

	mapKeyToResourceData(d, key)
	return nil
}

func resourceKeyUpdate(ctx context.Context, d *schema.ResourceData, m interface{}) diag.Diagnostics {
	c := m.(*Client)

	key := &Key{Key: d.Id()}
	mapResourceDataToKey(d, key)

	_, err := c.UpdateKey(key)
	if err != nil {
		return diag.FromErr(fmt.Errorf("error updating key: %s", err))
	}

	return resourceKeyRead(ctx, d, m)
}

func resourceKeyDelete(ctx context.Context, d *schema.ResourceData, m interface{}) diag.Diagnostics {
	c := m.(*Client)

	err := c.DeleteKey(d.Id())
	if err != nil {
		return diag.FromErr(fmt.Errorf("error deleting key: %s", err))
	}

	d.SetId("")
	return nil
}

func mapResourceDataToKey(d *schema.ResourceData, key *Key) {
	key.Models = expandStringList(d.Get("models").([]interface{}))
	if v, ok := d.GetOk("max_budget"); ok {
		val := v.(float64)
		key.MaxBudget = &val
	}
	key.UserID = d.Get("user_id").(string)
	key.TeamID = d.Get("team_id").(string)
	if v, ok := d.GetOk("max_parallel_requests"); ok {
		val := v.(int)
		key.MaxParallelRequests = &val
	}
	key.Metadata = d.Get("metadata").(map[string]interface{})
	if v, ok := d.GetOk("tpm_limit"); ok {
		val := v.(int)
		key.TPMLimit = &val
	}
	if v, ok := d.GetOk("rpm_limit"); ok {
		val := v.(int)
		key.RPMLimit = &val
	}
	key.BudgetDuration = d.Get("budget_duration").(string)
	key.AllowedCacheControls = expandStringList(d.Get("allowed_cache_controls").([]interface{}))
	if v, ok := d.GetOk("soft_budget"); ok {
		val := v.(float64)
		key.SoftBudget = &val
	}
	key.KeyAlias = d.Get("key_alias").(string)
	key.Duration = d.Get("duration").(string)
	key.Aliases = d.Get("aliases").(map[string]interface{})
	key.Config = d.Get("config").(map[string]interface{})
	key.Permissions = d.Get("permissions").(map[string]interface{})
	key.ModelMaxBudget = d.Get("model_max_budget").(map[string]interface{})
	key.ModelRPMLimit = d.Get("model_rpm_limit").(map[string]interface{})
	key.ModelTPMLimit = d.Get("model_tpm_limit").(map[string]interface{})
	key.Guardrails = expandStringList(d.Get("guardrails").([]interface{}))
	key.Blocked = d.Get("blocked").(bool)
	key.Tags = expandStringList(d.Get("tags").([]interface{}))
	key.BudgetID = d.Get("budget_id").(string)
	key.EnforcedParams = expandStringList(d.Get("enforced_params").([]interface{}))
	key.AllowedRoutes = expandStringList(d.Get("allowed_routes").([]interface{}))
	key.AllowedPassthroughRoutes = expandStringList(d.Get("allowed_passthrough_routes").([]interface{}))
	key.RPMLimitType = d.Get("rpm_limit_type").(string)
	key.TPMLimitType = d.Get("tpm_limit_type").(string)
	key.Prompts = expandStringList(d.Get("prompts").([]interface{}))
	key.OrganizationID = d.Get("organization_id").(string)
	key.ProjectID = d.Get("project_id").(string)
}

func mapKeyToResourceData(d *schema.ResourceData, key *Key) {
	// token_id is the SHA-256 hash of the key, used as the resource ID.
	// It is safe to store in state since it cannot be used to authenticate.
	d.Set("token_id", d.Id())

	// Note: "key" is write-only and must not be set here (Read operations).
	// It is only set during Create so it is available during apply.

	if len(key.Models) > 0 {
		d.Set("models", key.Models)
	}
	if key.MaxBudget != nil {
		d.Set("max_budget", *key.MaxBudget)
	}
	if key.UserID != "" {
		d.Set("user_id", key.UserID)
	}
	if key.TeamID != "" {
		d.Set("team_id", key.TeamID)
	}
	if key.MaxParallelRequests != nil {
		d.Set("max_parallel_requests", *key.MaxParallelRequests)
	}
	if key.Metadata != nil {
		d.Set("metadata", key.Metadata)
	}
	if key.TPMLimit != nil {
		d.Set("tpm_limit", *key.TPMLimit)
	}
	if key.RPMLimit != nil {
		d.Set("rpm_limit", *key.RPMLimit)
	}
	if key.BudgetDuration != "" {
		d.Set("budget_duration", key.BudgetDuration)
	}
	if len(key.AllowedCacheControls) > 0 {
		d.Set("allowed_cache_controls", key.AllowedCacheControls)
	}
	if key.SoftBudget != nil {
		d.Set("soft_budget", *key.SoftBudget)
	}
	if key.KeyAlias != "" {
		d.Set("key_alias", key.KeyAlias)
	}
	if key.Duration != "" {
		d.Set("duration", key.Duration)
	}
	if key.Aliases != nil {
		d.Set("aliases", key.Aliases)
	}
	if key.Config != nil {
		d.Set("config", key.Config)
	}
	if key.Permissions != nil {
		d.Set("permissions", key.Permissions)
	}
	if key.ModelMaxBudget != nil {
		d.Set("model_max_budget", key.ModelMaxBudget)
	}
	if key.ModelRPMLimit != nil {
		d.Set("model_rpm_limit", key.ModelRPMLimit)
	}
	if key.ModelTPMLimit != nil {
		d.Set("model_tpm_limit", key.ModelTPMLimit)
	}
	if len(key.Guardrails) > 0 {
		d.Set("guardrails", key.Guardrails)
	}
	d.Set("blocked", key.Blocked)
	if len(key.Tags) > 0 {
		d.Set("tags", key.Tags)
	}
	if key.Spend != 0 {
		d.Set("spend", key.Spend)
	}
	if key.BudgetID != "" {
		d.Set("budget_id", key.BudgetID)
	}
	if len(key.EnforcedParams) > 0 {
		d.Set("enforced_params", key.EnforcedParams)
	}
	if len(key.AllowedRoutes) > 0 {
		d.Set("allowed_routes", key.AllowedRoutes)
	}
	if len(key.AllowedPassthroughRoutes) > 0 {
		d.Set("allowed_passthrough_routes", key.AllowedPassthroughRoutes)
	}
	if key.RPMLimitType != "" {
		d.Set("rpm_limit_type", key.RPMLimitType)
	}
	if key.TPMLimitType != "" {
		d.Set("tpm_limit_type", key.TPMLimitType)
	}
	if len(key.Prompts) > 0 {
		d.Set("prompts", key.Prompts)
	}
	if key.OrganizationID != "" {
		d.Set("organization_id", key.OrganizationID)
	}
	if key.ProjectID != "" {
		d.Set("project_id", key.ProjectID)
	}
}
