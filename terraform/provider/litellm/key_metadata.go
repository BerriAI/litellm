package litellm

// The proxy merges these into the same metadata blob a caller's own metadata lives in, so a key
// that sets one shows a permanent diff without this filter. Only fields with their own schema
// attribute belong here; see TestReservedKeyMetadataFieldsAllHaveSchemaAttribute.
var reservedKeyMetadataFields = map[string]struct{}{
	"model_rpm_limit":            {},
	"model_tpm_limit":            {},
	"rpm_limit_type":             {},
	"tpm_limit_type":             {},
	"enforced_params":            {},
	"guardrails":                 {},
	"tags":                       {},
	"prompts":                    {},
	"allowed_passthrough_routes": {},
}

func withoutReservedKeyMetadataFields(raw map[string]interface{}) map[string]interface{} {
	if raw == nil {
		return nil
	}
	out := make(map[string]interface{}, len(raw))
	for k, v := range raw {
		if _, reserved := reservedKeyMetadataFields[k]; reserved {
			continue
		}
		out[k] = v
	}
	return out
}
