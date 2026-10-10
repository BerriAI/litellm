pub mod batches;
pub mod beta;
pub mod count_tokens;

pub const API_BASE: &str = "https://api.anthropic.com";
pub const MESSAGES_PATH: &str = "/v1/messages";
pub const BATCHES_PATH: &str = "/v1/messages/batches";
pub const COUNT_TOKENS_PATH: &str = "/v1/messages/count_tokens";

pub const API_KEY_HEADER: &str = "x-api-key";
pub const BETA_HEADER: &str = "anthropic-beta";
pub const VERSION_HEADER: &str = "anthropic-version";
pub const DIRECT_BROWSER_ACCESS_HEADER: &str = "anthropic-dangerous-direct-browser-access";
pub const API_VERSION: &str = "2023-06-01";

pub const DEFAULT_HEADERS: &[(&str, &str)] = &[
    (VERSION_HEADER, API_VERSION),
    ("content-type", "application/json"),
];
