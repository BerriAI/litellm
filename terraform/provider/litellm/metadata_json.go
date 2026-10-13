package litellm

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"sort"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func metadataJSONSchema(description string) *schema.Schema {
	return &schema.Schema{
		Type:             schema.TypeString,
		Optional:         true,
		ValidateFunc:     validateMetadataJSON,
		DiffSuppressFunc: budgetSuppressEquivalentJSON,
		Description:      description,
	}
}

func computedMetadataJSONSchema(description string) *schema.Schema {
	return &schema.Schema{
		Type:        schema.TypeString,
		Computed:    true,
		Description: description,
	}
}

func validateMetadataJSON(v interface{}, k string) ([]string, []error) {
	var parsed map[string]interface{}
	if err := json.Unmarshal([]byte(v.(string)), &parsed); err != nil || parsed == nil {
		return nil, []error{fmt.Errorf("%q must be a JSON object, got %s", k, v)}
	}
	return nil, nil
}

func decodeMetadataJSON(raw string) map[string]interface{} {
	var parsed map[string]interface{}
	if err := json.Unmarshal([]byte(raw), &parsed); err != nil || parsed == nil {
		return map[string]interface{}{}
	}
	return parsed
}

// declaredMetadata is the set of metadata keys Terraform owns.
func declaredMetadata(metadata map[string]interface{}, metadataJSON string) map[string]interface{} {
	result := make(map[string]interface{}, len(metadata))
	for k, v := range metadata {
		result[k] = v
	}
	for k, v := range decodeMetadataJSON(metadataJSON) {
		result[k] = v
	}
	return result
}

// metadataJSONFromServer keeps the configured string when the server matches it.
func metadataJSONFromServer(server map[string]interface{}, configured string) string {
	declared := decodeMetadataJSON(configured)
	if len(declared) == 0 {
		return configured
	}
	actual := make(map[string]interface{}, len(declared))
	for k := range declared {
		if v, ok := server[k]; ok {
			actual[k] = v
		}
	}
	if reflect.DeepEqual(actual, declared) {
		return configured
	}
	encoded, err := json.Marshal(actual)
	if err != nil {
		return configured
	}
	return string(encoded)
}

func encodeMetadataJSON(metadata map[string]interface{}) string {
	if len(metadata) == 0 {
		return ""
	}
	encoded, err := json.Marshal(metadata)
	if err != nil {
		return ""
	}
	return string(encoded)
}

func withoutKeys(m map[string]string, drop map[string]interface{}) map[string]string {
	for k := range drop {
		delete(m, k)
	}
	return m
}

func validateMetadataJSONKeys(reserved []string) schema.CustomizeDiffFunc {
	return func(_ context.Context, d *schema.ResourceDiff, _ interface{}) error {
		raw, _ := d.Get("metadata_json").(string)
		jsonKeys := decodeMetadataJSON(raw)
		metadata, _ := d.Get("metadata").(map[string]interface{})
		var conflicts, reservedHits []string
		for k := range jsonKeys {
			if _, ok := metadata[k]; ok {
				conflicts = append(conflicts, k)
			}
			for _, r := range reserved {
				if k == r {
					reservedHits = append(reservedHits, k)
				}
			}
		}
		sort.Strings(conflicts)
		sort.Strings(reservedHits)
		if len(conflicts) > 0 {
			return fmt.Errorf("metadata and metadata_json both set %v; set each key in one of them", conflicts)
		}
		if len(reservedHits) > 0 {
			return fmt.Errorf("metadata_json sets %v, which have attributes of their own; use those instead", reservedHits)
		}
		return nil
	}
}
