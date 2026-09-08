pub(crate) use crate::providers::dispatch::chat_completions_provider_config;

use crate::Error;
use crate::http_utils::string_headers as shared_string_headers;
use serde_json::{Map, Value};

const HEADER_CONTEXT: &str = "chat completions";

pub(super) fn string_headers(
    extra_headers: Option<Map<String, Value>>,
) -> Result<Vec<(String, String)>, Error> {
    shared_string_headers(HEADER_CONTEXT, extra_headers)
}
