use litellm_auth_types::{Setting, secret, setting};

pub const API_KEY: Setting = secret(&["api_key"], &[]);
pub const API_BASE: Setting = setting(&["api_base", "base_url"], &[]);
pub const CUSTOM_ENDPOINT: Setting = setting(&["custom_endpoint"], &[]);
pub const EXTRA_HEADERS: Setting = setting(&["extra_headers"], &[]);
pub const TIMEOUT: Setting = setting(&["timeout", "request_timeout", "timeout_seconds"], &[]);
pub const MAX_RETRIES: Setting = setting(&["max_retries"], &[]);

pub const SETTINGS: &[Setting] = &[
    API_KEY,
    API_BASE,
    CUSTOM_ENDPOINT,
    EXTRA_HEADERS,
    TIMEOUT,
    MAX_RETRIES,
];
