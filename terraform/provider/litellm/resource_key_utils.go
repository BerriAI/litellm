package litellm

import (
	"fmt"
	"log"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func buildKeyData(d *schema.ResourceData) map[string]interface{} {
	keyData := make(map[string]interface{})

	if v, ok := d.GetOkExists("models"); ok {
		models := expandStringList(v.([]interface{}))
		if len(models) > 0 {
			keyData["models"] = models
		}
	}
	if v, ok := d.GetOkExists("max_budget"); ok {
		keyData["max_budget"] = v.(float64)
	}
	if v, ok := d.GetOkExists("user_id"); ok {
		keyData["user_id"] = v.(string)
	}
	if v, ok := d.GetOkExists("team_id"); ok {
		keyData["team_id"] = v.(string)
	}
	if v, ok := d.GetOkExists("max_parallel_requests"); ok {
		keyData["max_parallel_requests"] = v.(int)
	}
	if v, ok := d.GetOkExists("metadata"); ok {
		keyData["metadata"] = v.(map[string]interface{})
	}
	if v, ok := d.GetOkExists("tpm_limit"); ok {
		keyData["tpm_limit"] = v.(int)
	}
	if v, ok := d.GetOkExists("rpm_limit"); ok {
		keyData["rpm_limit"] = v.(int)
	}
	if v, ok := d.GetOkExists("budget_duration"); ok {
		keyData["budget_duration"] = v.(string)
	}
	if v, ok := d.GetOkExists("allowed_cache_controls"); ok {
		cacheControls := expandStringList(v.([]interface{}))
		if len(cacheControls) > 0 {
			keyData["allowed_cache_controls"] = cacheControls
		}
	}
	if v, ok := d.GetOkExists("soft_budget"); ok {
		keyData["soft_budget"] = v.(float64)
	}
	if v, ok := d.GetOkExists("key_alias"); ok {
		keyData["key_alias"] = v.(string)
	}
	if v, ok := d.GetOkExists("duration"); ok {
		keyData["duration"] = v.(string)
	}
	if v, ok := d.GetOkExists("aliases"); ok {
		keyData["aliases"] = v.(map[string]interface{})
	}
	if v, ok := d.GetOkExists("config"); ok {
		keyData["config"] = v.(map[string]interface{})
	}
	if v, ok := d.GetOkExists("permissions"); ok {
		keyData["permissions"] = v.(map[string]interface{})
	}
	if v, ok := d.GetOkExists("model_max_budget"); ok {
		keyData["model_max_budget"] = v.(map[string]interface{})
	}
	if v, ok := d.GetOkExists("model_rpm_limit"); ok {
		keyData["model_rpm_limit"] = v.(map[string]interface{})
	}
	if v, ok := d.GetOkExists("model_tpm_limit"); ok {
		keyData["model_tpm_limit"] = v.(map[string]interface{})
	}
	if v, ok := d.GetOkExists("guardrails"); ok {
		guardrails := expandStringList(v.([]interface{}))
		if len(guardrails) > 0 {
			keyData["guardrails"] = guardrails
		}
	}
	if v, ok := d.GetOkExists("blocked"); ok {
		keyData["blocked"] = v.(bool)
	}
	if v, ok := d.GetOkExists("tags"); ok {
		tags := expandStringList(v.([]interface{}))
		if len(tags) > 0 {
			keyData["tags"] = tags
		}
	}

	return keyData
}

func setKeyResourceData(d *schema.ResourceData, key *Key) error {
	fields := map[string]interface{}{
		"key":                    key.Key,
		"models":                 key.Models,
		"spend":                  key.Spend,
		"user_id":                key.UserID,
		"team_id":                key.TeamID,
		"metadata":               key.Metadata,
		"budget_duration":        key.BudgetDuration,
		"allowed_cache_controls": key.AllowedCacheControls,
		"key_alias":              key.KeyAlias,
		"duration":               key.Duration,
		"aliases":                key.Aliases,
		"config":                 key.Config,
		"permissions":            key.Permissions,
		"model_max_budget":       key.ModelMaxBudget,
		"model_rpm_limit":        key.ModelRPMLimit,
		"model_tpm_limit":        key.ModelTPMLimit,
		"guardrails":             key.Guardrails,
		"blocked":                key.Blocked,
		"tags":                   key.Tags,
	}

	for field, value := range fields {
		if err := d.Set(field, value); err != nil {
			log.Printf("[WARN] Error setting %s: %s", field, err)
			return fmt.Errorf("error setting %s: %s", field, err)
		}
	}

	// Handle pointer fields separately - only set if not nil
	if key.MaxBudget != nil {
		if err := d.Set("max_budget", *key.MaxBudget); err != nil {
			return fmt.Errorf("error setting max_budget: %s", err)
		}
	}
	if key.SoftBudget != nil {
		if err := d.Set("soft_budget", *key.SoftBudget); err != nil {
			return fmt.Errorf("error setting soft_budget: %s", err)
		}
	}
	if key.MaxParallelRequests != nil {
		if err := d.Set("max_parallel_requests", *key.MaxParallelRequests); err != nil {
			return fmt.Errorf("error setting max_parallel_requests: %s", err)
		}
	}
	if key.TPMLimit != nil {
		if err := d.Set("tpm_limit", *key.TPMLimit); err != nil {
			return fmt.Errorf("error setting tpm_limit: %s", err)
		}
	}
	if key.RPMLimit != nil {
		if err := d.Set("rpm_limit", *key.RPMLimit); err != nil {
			return fmt.Errorf("error setting rpm_limit: %s", err)
		}
	}

	return nil
}

func expandStringList(list []interface{}) []string {
	result := make([]string, len(list))
	for i, v := range list {
		result[i] = v.(string)
	}
	return result
}

func mapToKey(data map[string]interface{}) *Key {
	key := &Key{}
	for k, v := range data {
		switch k {
		case "key":
			key.Key = v.(string)
		case "models":
			key.Models = v.([]string)
		case "max_budget":
			if v, ok := v.(float64); ok {
				key.MaxBudget = &v
			}
		case "user_id":
			key.UserID = v.(string)
		case "team_id":
			key.TeamID = v.(string)
		case "max_parallel_requests":
			if v, ok := v.(int); ok {
				key.MaxParallelRequests = &v
			}
		case "metadata":
			key.Metadata = v.(map[string]interface{})
		case "tpm_limit":
			if v, ok := v.(int); ok {
				key.TPMLimit = &v
			}
		case "rpm_limit":
			if v, ok := v.(int); ok {
				key.RPMLimit = &v
			}
		case "budget_duration":
			key.BudgetDuration = v.(string)
		case "allowed_cache_controls":
			key.AllowedCacheControls = v.([]string)
		case "soft_budget":
			if v, ok := v.(float64); ok {
				key.SoftBudget = &v
			}
		case "key_alias":
			key.KeyAlias = v.(string)
		case "duration":
			key.Duration = v.(string)
		case "aliases":
			key.Aliases = v.(map[string]interface{})
		case "config":
			key.Config = v.(map[string]interface{})
		case "permissions":
			key.Permissions = v.(map[string]interface{})
		case "model_max_budget":
			key.ModelMaxBudget = v.(map[string]interface{})
		case "model_rpm_limit":
			key.ModelRPMLimit = v.(map[string]interface{})
		case "model_tpm_limit":
			key.ModelTPMLimit = v.(map[string]interface{})
		case "guardrails":
			key.Guardrails = v.([]string)
		case "blocked":
			key.Blocked = v.(bool)
		case "tags":
			key.Tags = v.([]string)
		}
	}
	return key
}

func buildKeyForCreation(data map[string]interface{}) *Key {
	return mapToKey(data)
}
