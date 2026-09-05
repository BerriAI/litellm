use serde_json::{Map, Value};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RequestAttribution {
    pub user_api_key_hash: Option<String>,
    pub user_api_key_user_id: Option<String>,
    pub user_api_key_team_id: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LiteLlmRequestContext {
    pub metadata: Option<Map<String, Value>>,
    pub litellm_metadata: Option<Map<String, Value>>,
    pub request_metadata_fields: Vec<String>,
    pub litellm_call_id: Option<String>,
    pub request_model: Option<String>,
    pub attribution: RequestAttribution,
}
