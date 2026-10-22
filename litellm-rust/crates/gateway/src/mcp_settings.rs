use litellm_config::Config;

pub(crate) fn first_unsupported(config: &Config) -> Option<&'static str> {
    let general = &config.general_settings.mcp;
    let litellm = &config.litellm_settings.mcp;
    [
        (
            "mcp_advertised_versions",
            general.mcp_advertised_versions.is_some(),
        ),
        (
            "mcp_disabled_capabilities",
            general.mcp_disabled_capabilities.is_some(),
        ),
        ("mcp_allowed_clients", general.mcp_allowed_clients.is_some()),
        (
            "mcp_require_explicit_server_scope",
            general.mcp_require_explicit_server_scope.is_some(),
        ),
        (
            "mcp_client_id_header",
            general.mcp_client_id_header.is_some(),
        ),
        (
            "mcp_client_side_auth_header_name",
            general.mcp_client_side_auth_header_name.is_some(),
        ),
        (
            "mcp_allow_all_keys_respects_mcp_scope",
            general.mcp_allow_all_keys_respects_mcp_scope.is_some(),
        ),
        (
            "mcp_prefer_client_id_metadata_document",
            general.mcp_prefer_client_id_metadata_document.is_some(),
        ),
        (
            "mcp_internal_ip_ranges",
            general.mcp_internal_ip_ranges.is_some(),
        ),
        (
            "mcp_trusted_proxy_ranges",
            general.mcp_trusted_proxy_ranges.is_some(),
        ),
        (
            "mcp_xff_num_trusted_hops",
            general.mcp_xff_num_trusted_hops.is_some(),
        ),
        ("mcp_required_fields", general.mcp_required_fields.is_some()),
        (
            "mcp_session_token_signing",
            general.mcp_session_token_signing.is_some(),
        ),
        ("mcp_aliases", litellm.mcp_aliases.is_some()),
        (
            "mcp_semantic_tool_filter",
            litellm.mcp_semantic_tool_filter.is_some(),
        ),
        ("mcp_tool_search", litellm.mcp_tool_search.is_some()),
    ]
    .into_iter()
    .find_map(|(key, set)| set.then_some(key))
}
