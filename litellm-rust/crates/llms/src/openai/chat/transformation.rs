//! `litellm/llms/openai/chat/gpt_transformation.py` as a passthrough: the caller's body is
//! already OpenAI-shaped, so only the model name and LiteLLM's own request keys change, and
//! the response bytes reach the caller untouched.

use serde_json::{Map, Value};

use crate::{
    Error, ErrorDetail,
    base_llm::auth::{Headers, ValidatedEnvironment},
    openai::common_utils,
};

pub const OPENAI_CHAT_COMPLETIONS_PATH: &str = "/chat/completions";

/// The keys of Python's `all_litellm_params`, and the `completion()` arguments that decide which
/// params are sent, that a proxy caller can put in a chat request body. Python consumes them
/// before the OpenAI call, so they never reach the provider.
const LITELLM_OWNED_PARAMS: &[&str] = &[
    "additional_drop_params",
    "allowed_openai_params",
    "api_base",
    "api_key",
    "api_version",
    "base_model",
    "cache",
    "caching",
    "context_window_fallback_dict",
    "custom_llm_provider",
    "drop_params",
    "extra_headers",
    "fallbacks",
    "guardrails",
    "litellm_metadata",
    "litellm_session_id",
    "litellm_trace_id",
    "max_retries",
    "metadata",
    "mock_response",
    "mock_timeout",
    "model_info",
    "no-log",
    "num_retries",
    "prompt_id",
    "prompt_variables",
    "prompt_version",
    "request_timeout",
    "stream_timeout",
    "tags",
    "timeout",
];
const INTERNAL_PARAM_PREFIX: &str = "_litellm_";
const EXTRA_BODY: &str = "extra_body";

pub struct OpenAiChatConfig;

pub const OPENAI_CHAT_COMPLETIONS_CONFIG: OpenAiChatConfig = OpenAiChatConfig;

impl OpenAiChatConfig {
    pub fn secret_names(
        &self,
        api_key: Option<&str>,
        api_base: Option<&str>,
    ) -> &'static [&'static str] {
        common_utils::secret_names(api_key, api_base)
    }

    pub fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        common_utils::bearer_environment(headers, api_key, lookup)
    }

    pub fn get_complete_url(
        &self,
        api_base: Option<&str>,
        lookup: &dyn Fn(&str) -> Option<String>,
    ) -> String {
        common_utils::complete_url(api_base, OPENAI_CHAT_COMPLETIONS_PATH, lookup)
    }

    /// The OpenAI SDK merges `extra_body` over the body, and the deployment's model name
    /// replaces the public one.
    pub fn transform_request(
        &self,
        model: &str,
        body: Map<String, Value>,
    ) -> Result<Map<String, Value>, Error> {
        let mut body = body;
        let extra_body = take_extra_body(&mut body)?;
        let mut body: Map<String, Value> = body
            .into_iter()
            .chain(extra_body)
            .filter(|(name, _)| !is_litellm_owned(name))
            .collect();
        body.insert("model".into(), Value::String(model.into()));
        Ok(body)
    }
}

/// Whether the body asks for SSE. Anything but a boolean or null is the caller's mistake,
/// since the call cannot tell which response shape to expect.
pub fn is_stream(body: &Map<String, Value>) -> Result<bool, Error> {
    match body.get("stream") {
        None | Some(Value::Null) => Ok(false),
        Some(Value::Bool(stream)) => Ok(*stream),
        Some(other) => Err(Error::InvalidRequest(ErrorDetail::InvalidValue {
            field: "stream",
            expected: "a boolean",
            actual: other.clone(),
        })),
    }
}

/// Removes the body's `extra_body`, which the OpenAI SDK merges over the body it sends.
pub fn take_extra_body(body: &mut Map<String, Value>) -> Result<Map<String, Value>, Error> {
    match body.remove(EXTRA_BODY) {
        None | Some(Value::Null) => Ok(Map::new()),
        Some(Value::Object(extra)) => Ok(extra),
        Some(other) => Err(Error::InvalidRequest(ErrorDetail::InvalidValue {
            field: EXTRA_BODY,
            expected: "an object",
            actual: other,
        })),
    }
}

/// Whether Python consumes `name` before the provider call, so it never reaches the wire.
pub fn is_litellm_owned(name: &str) -> bool {
    LITELLM_OWNED_PARAMS.contains(&name) || name.starts_with(INTERNAL_PARAM_PREFIX)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn object(value: Value) -> Map<String, Value> {
        let Value::Object(map) = value else {
            panic!("expected an object, got {value}");
        };
        map
    }

    #[rstest]
    fn the_body_passes_through_with_only_the_model_renamed() {
        let body = json!({
            "model": "public/gpt",
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "u"}}]}],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
            "stream": true,
            "stream_options": {"include_usage": true},
            "reasoning_effort": "high",
            "a_future_param": {"nested": [1, 2]},
            "user": "end-user"
        });
        let sent = OPENAI_CHAT_COMPLETIONS_CONFIG
            .transform_request("gpt-6-astra", object(body.clone()))
            .unwrap();
        let mut expected = object(body);
        expected.insert("model".into(), json!("gpt-6-astra"));
        assert_eq!(sent, expected);
    }

    #[rstest]
    #[case::metadata("metadata")]
    #[case::mock_response("mock_response")]
    #[case::fallbacks("fallbacks")]
    #[case::drop_params("drop_params")]
    #[case::additional_drop_params("additional_drop_params")]
    #[case::allowed_openai_params("allowed_openai_params")]
    #[case::credentials("api_key")]
    #[case::timeout("timeout")]
    #[case::no_log("no-log")]
    #[case::internal("_litellm_control")]
    fn litellm_owned_keys_never_reach_openai(#[case] name: &str) {
        let body = object(json!({"model": "m", "messages": [], name: "x"}));
        let sent = OPENAI_CHAT_COMPLETIONS_CONFIG
            .transform_request("gpt", body)
            .unwrap();
        assert_eq!(sent, object(json!({"model": "gpt", "messages": []})));
    }

    #[rstest]
    fn extra_body_is_merged_over_the_body_but_not_over_the_model() {
        let body = object(json!({
            "model": "m",
            "temperature": 1,
            "extra_body": {"temperature": 0, "top_k": 5, "model": "other", "metadata": {}}
        }));
        let sent = OPENAI_CHAT_COMPLETIONS_CONFIG
            .transform_request("gpt", body)
            .unwrap();
        assert_eq!(
            sent,
            object(json!({"model": "gpt", "temperature": 0, "top_k": 5}))
        );
    }

    #[rstest]
    fn a_non_object_extra_body_is_rejected() {
        let body = object(json!({"model": "m", "extra_body": "x"}));
        assert!(matches!(
            OPENAI_CHAT_COMPLETIONS_CONFIG.transform_request("gpt", body),
            Err(Error::InvalidRequest(ErrorDetail::InvalidValue {
                field: "extra_body",
                ..
            }))
        ));
    }

    #[rstest]
    #[case::absent(json!({}), Some(false))]
    #[case::null(json!({"stream": null}), Some(false))]
    #[case::off(json!({"stream": false}), Some(false))]
    #[case::on(json!({"stream": true}), Some(true))]
    #[case::string(json!({"stream": "true"}), None)]
    fn the_stream_flag_must_be_a_boolean(#[case] body: Value, #[case] expected: Option<bool>) {
        assert_eq!(is_stream(&object(body)).ok(), expected);
    }
}
