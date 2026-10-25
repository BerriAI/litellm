use crate::{Raw, Spelled};

#[macro_rules_attribute::apply(crate::config_section)]
pub struct McpSettings {
    pub mcp_allowed_hosts: Option<Box<[String]>>,
    pub mcp_allowed_origins: Box<[String]>,
    pub mcp_advertised_versions: Option<Raw>,
    pub mcp_disabled_capabilities: Option<Raw>,
    pub mcp_allowed_clients: Option<Raw>,
    pub mcp_require_explicit_server_scope: Option<Spelled<bool>>,
    pub mcp_client_id_header: Option<String>,
    pub mcp_client_side_auth_header_name: Option<String>,
    pub mcp_allow_all_keys_respects_mcp_scope: Option<Spelled<bool>>,
    pub mcp_prefer_client_id_metadata_document: Option<Spelled<bool>>,
    pub mcp_internal_ip_ranges: Option<Raw>,
    pub mcp_trusted_proxy_ranges: Option<Raw>,
    pub mcp_xff_num_trusted_hops: Option<Spelled<i64>>,
    pub mcp_required_fields: Option<Raw>,
    pub mcp_session_token_signing: Option<Raw>,
}
