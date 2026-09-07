#[derive(Clone, Debug, Default, PartialEq)]
pub struct RequestAttribution {
    pub user_api_key_hash: Option<String>,
    pub user_api_key_user_id: Option<String>,
    pub user_api_key_team_id: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RequestCapabilities {
    pub execution_mode: Option<String>,
    pub stream: bool,
    pub has_agentic_hook: bool,
    pub has_custom_client: bool,
    pub request_format: Option<String>,
    pub input_source_kind: Option<String>,
    pub native_response_format: bool,
    pub websocket_mode: Option<String>,
    pub requires_connection: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LiteLlmRequestContext {
    pub litellm_call_id: Option<String>,
    pub trace_id: Option<String>,
    pub request_model: Option<String>,
    pub attribution: RequestAttribution,
    pub capabilities: RequestCapabilities,
}
