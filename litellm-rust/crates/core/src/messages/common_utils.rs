pub use crate::providers::dispatch::messages_provider_config;

use crate::Error;
use crate::http_utils::string_headers as shared_string_headers;
use serde_json::{Map, Value};

pub use crate::http_utils::{has_bearer_auth, has_header, truncate_error_body};

const HEADER_CONTEXT: &str = "messages";

pub fn string_headers(
    extra_headers: Option<Map<String, Value>>,
) -> Result<Vec<(String, String)>, Error> {
    shared_string_headers(HEADER_CONTEXT, extra_headers)
}
