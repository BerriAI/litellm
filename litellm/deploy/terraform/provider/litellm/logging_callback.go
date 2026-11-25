package litellm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"

	"github.com/hashicorp/go-cty/cty"
	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/validation"
)

const loggingCallbackAttr = "logging_callback"
const defaultLoggingCallbackType = "success_and_failure"

func loggingCallbackSchema() *schema.Schema {
	return &schema.Schema{
		Type:        schema.TypeList,
		Optional:    true,
		Description: "Logging callbacks managed on this resource. Requires Terraform 1.11 or later.",
		Elem: &schema.Resource{Schema: map[string]*schema.Schema{
			"callback_name": {
				Type:         schema.TypeString,
				Required:     true,
				ValidateFunc: validation.StringIsNotEmpty,
			},
			"callback_type": {
				Type:         schema.TypeString,
				Optional:     true,
				Default:      defaultLoggingCallbackType,
				ValidateFunc: validation.StringInSlice([]string{"success", "failure", defaultLoggingCallbackType}, false),
			},
			"callback_vars_wo": {
				Type:      schema.TypeString,
				Required:  true,
				Sensitive: true,
				WriteOnly: true,
				ValidateFunc: func(value interface{}, _ string) ([]string, []error) {
					if _, err := parseCallbackVars(value.(string)); err != nil {
						return nil, []error{err}
					}
					return nil, nil
				},
				Description: "JSON object of string values. Use jsonencode to supply it. Never stored in plan or state.",
			},
			"callback_vars_wo_version": {
				Type:        schema.TypeInt,
				Required:    true,
				Description: "Change this number whenever callback_vars_wo changes to send the new values.",
			},
		}},
	}
}

func parseCallbackVars(raw string) (map[string]*string, error) {
	var vars map[string]*string
	if err := json.Unmarshal([]byte(raw), &vars); err != nil || vars == nil {
		return nil, errors.New("callback_vars_wo must be a JSON object with string values")
	}
	for _, value := range vars {
		if value == nil {
			return nil, errors.New("callback_vars_wo must be a JSON object with string values")
		}
	}
	return vars, nil
}

func validateLoggingCallbackConfig(d interface{ Get(string) interface{} }) error {
	if _, legacy := d.Get("metadata").(map[string]interface{})["logging"]; legacy && len(d.Get(loggingCallbackAttr).([]interface{})) != 0 {
		return errors.New("logging_callback cannot be used together with metadata.logging")
	}
	return nil
}

func validateLoggingCallbackDiff(_ context.Context, d *schema.ResourceDiff, _ interface{}) error {
	return validateLoggingCallbackConfig(d)
}

type loggingCallbackEntry struct {
	Name string             `json:"callback_name"`
	Type string             `json:"callback_type"`
	Vars map[string]*string `json:"callback_vars"`
}

type loggingCallbackIdentity struct {
	name         string
	callbackType string
}

func callbackIdentity(block map[string]interface{}) loggingCallbackIdentity {
	name, _ := block["callback_name"].(string)
	callbackType, _ := block["callback_type"].(string)
	if callbackType == "" {
		callbackType = defaultLoggingCallbackType
	}
	return loggingCallbackIdentity{name, callbackType}
}

func writeLoggingCallbackMetadata(d *schema.ResourceData, metadata map[string]interface{}) error {
	if err := validateLoggingCallbackConfig(d); err != nil {
		return err
	}
	blocks := d.Get(loggingCallbackAttr).([]interface{})
	if len(blocks) == 0 {
		if _, legacy := d.Get("metadata").(map[string]interface{})["logging"]; legacy {
			return nil
		}
	}

	old, _ := d.GetChange(loggingCallbackAttr)
	prior, _ := old.([]interface{})
	managed := make(map[loggingCallbackIdentity]bool, len(prior)+len(blocks))
	for _, item := range prior {
		block, _ := item.(map[string]interface{})
		managed[callbackIdentity(block)] = true
	}
	stored, _ := metadata["logging"].([]interface{})
	entries := make([]interface{}, 0, len(blocks)+len(stored))
	for i, item := range blocks {
		block := item.(map[string]interface{})
		id := callbackIdentity(block)
		managed[id] = true
		raw := nestedCallbackVars(d, i)
		vars, err := parseCallbackVars(raw)
		if err != nil {
			return fmt.Errorf("logging_callback[%d]: %w", i, err)
		}
		entries = append(entries, loggingCallbackEntry{
			Name: id.name,
			Type: id.callbackType,
			Vars: vars,
		})
	}
	for _, item := range stored {
		entry, _ := item.(map[string]interface{})
		if !managed[callbackIdentity(entry)] {
			entries = append(entries, item)
		}
	}
	if len(entries) == 0 {
		delete(metadata, "logging")
	} else {
		metadata["logging"] = entries
	}
	return nil
}

func nestedCallbackVars(d *schema.ResourceData, i int) string {
	path := cty.GetAttrPath(loggingCallbackAttr).IndexInt(i).GetAttr("callback_vars_wo")
	raw, diags := d.GetRawConfigAt(path)
	if !diags.HasError() && raw.IsKnown() && !raw.IsNull() && raw.Type() == cty.String {
		return raw.AsString()
	}
	value, _ := d.Get(fmt.Sprintf("%s.%d.callback_vars_wo", loggingCallbackAttr, i)).(string)
	return value
}

func readLoggingCallbacks(d *schema.ResourceData, metadata map[string]interface{}) error {
	prior := d.Get(loggingCallbackAttr).([]interface{})
	if len(prior) == 0 {
		return nil
	}
	stored, _ := metadata["logging"].([]interface{})
	remaining := make(map[loggingCallbackIdentity]int, len(stored))
	for _, item := range stored {
		entry, _ := item.(map[string]interface{})
		id := callbackIdentity(entry)
		if id.name != "" {
			remaining[id]++
		}
	}
	blocks := make([]map[string]interface{}, 0, len(prior))
	for _, item := range prior {
		block, _ := item.(map[string]interface{})
		id := callbackIdentity(block)
		if remaining[id] == 0 {
			continue
		}
		remaining[id]--
		blocks = append(blocks, map[string]interface{}{
			"callback_name": id.name, "callback_type": id.callbackType,
			"callback_vars_wo_version": block["callback_vars_wo_version"],
		})
	}
	return d.Set(loggingCallbackAttr, blocks)
}

func callbackMetadataForState(slot string, value interface{}) (interface{}, bool) {
	switch slot {
	case "logging":
		entries, ok := value.([]interface{})
		if !ok {
			return nil, false
		}
		safe := make([]map[string]interface{}, 0, len(entries))
		for _, item := range entries {
			if entry, ok := item.(map[string]interface{}); ok {
				safe = append(safe, withoutCallbackVars(entry))
			}
		}
		return safe, true
	case "callback_settings":
		settings, ok := value.(map[string]interface{})
		if !ok {
			return nil, false
		}
		return withoutCallbackVars(settings), true
	default:
		return value, true
	}
}

func withoutCallbackVars(entry map[string]interface{}) map[string]interface{} {
	safe := maps.Clone(entry)
	delete(safe, "callback_vars")
	return safe
}
