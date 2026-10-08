use litellm_auth_types::{Setting, setting};

pub const MODEL: Setting = setting(&["model"], &[]);
pub const CUSTOM_LLM_PROVIDER: Setting = setting(&["custom_llm_provider"], &[]);
pub const EXTRA_BODY: Setting = setting(&["extra_body"], &[]);
pub const DROP_PARAMS: Setting = setting(&["drop_params"], &[]);
pub const ADDITIONAL_DROP_PARAMS: Setting = setting(&["additional_drop_params"], &[]);
pub const REQ_FORMAT: Setting = setting(&["req_format"], &[]);
pub const MAX_RESPONSE_BYTES: Setting = setting(&["max_response_bytes"], &[]);

pub const SETTINGS: &[Setting] = &[
    MODEL,
    CUSTOM_LLM_PROVIDER,
    EXTRA_BODY,
    DROP_PARAMS,
    ADDITIONAL_DROP_PARAMS,
    REQ_FORMAT,
    MAX_RESPONSE_BYTES,
];
