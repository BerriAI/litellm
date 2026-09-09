package litellm

// The proxy has no dedicated column for these; /key/generate and /key/update merge them into
// the same metadata JSON blob a caller's own metadata lives in (litellm/proxy/management_endpoints/
// key_management_endpoints.py, "Add model_rpm_limit and model_tpm_limit to metadata"). Each already
// has its own schema attribute here, so leaving them in the metadata map read back from the API
// makes every key that sets one of them show a permanent metadata diff: config never repeats the
// value inside its metadata block, but the API always returns it there too.
//
// Names mirror litellm/proxy/_types.py's LiteLLM_ManagementEndpoint_MetadataFields,
// LiteLLM_ManagementEndpoint_MetadataFields_Premium and LiteLLM_Reserved_Metadata_Fields.
var reservedKeyMetadataFields = map[string]struct{}{
	"model_rpm_limit":                           {},
	"model_tpm_limit":                           {},
	"model_itpm_limit":                          {},
	"model_otpm_limit":                          {},
	"default_estimated_output_tokens":           {},
	"default_estimated_output_tokens_per_model": {},
	"mcp_rpm_limit":                             {},
	"tag_rpm_limit":                             {},
	"rpm_limit_type":                            {},
	"tpm_limit_type":                            {},
	"enforced_params":                           {},
	"temp_budget_increase":                      {},
	"temp_budget_expiry":                        {},
	"allowed_vector_store_indexes":              {},
	"enforced_batch_output_expires_after":       {},
	"enforced_file_expires_after":               {},
	"throttle_on_budget_exceeded":               {},
	"enable_prompt_caching":                     {},
	"disable_global_guardrails":                 {},
	"guardrails":                                {},
	"policies":                                  {},
	"tags":                                      {},
	"team_member_key_duration":                  {},
	"prompts":                                   {},
	"logging":                                   {},
	"secret_manager_settings":                   {},
	"allowed_passthrough_routes":                {},
	"service_account_id":                        {},
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
