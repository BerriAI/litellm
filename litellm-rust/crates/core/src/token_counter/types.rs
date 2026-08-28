use serde_json::{Map, Value};

pub struct TokenCounterRequest<'a> {
    pub model: &'a str,
    pub text: Option<&'a str>,
    pub messages: Option<&'a [Map<String, Value>]>,
    pub tools: Option<&'a [Map<String, Value>]>,
    pub tool_choice: Option<&'a Value>,
    pub count_response_tokens: bool,
    pub default_token_count: Option<usize>,
}
