package litellm

// The proxy has no dedicated column for these; /key/generate and /key/update merge them into
// the same metadata JSON blob a caller's own metadata lives in (litellm/proxy/management_endpoints/
// key_management_endpoints.py, "Add model_rpm_limit and model_tpm_limit to metadata"). Filtering
// them out of what this resource reads back into its metadata attribute stops the permanent diff
// that otherwise shows on any key that sets one, since config never repeats the value inside its
// metadata block.
//
// Deliberately narrower than litellm/proxy/_types.py's full LiteLLM_ManagementEndpoint_MetadataFields
// (+ its Premium counterpart): this resource only has its own argument for some of them. A name
// belongs here only once it also has a schema attribute in resource_key.go, so a value with no
// attribute to hold it is never silently dropped from what this resource exposes; it stays visible
// through metadata until this resource gains a real argument for it.
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

// withoutReservedKeyMetadataFields returns a copy of raw with the proxy-owned fields above removed,
// leaving only what the caller's own metadata config actually declares. Never mutates raw.
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
