//! Python's `get_optional_params` for the providers chat translates: the request's keys are
//! sorted into OpenAI params, keys LiteLLM consumes itself, and provider-specific keys that
//! pass through; an OpenAI param the provider does not take is dropped under `drop_params`
//! and refused otherwise.

use serde_json::{Map, Value};

use crate::{
    Error, ErrorDetail, base_llm::model_info::ModelInfo,
    openai::chat::transformation::is_litellm_owned,
};

/// The keys of `DEFAULT_CHAT_COMPLETION_PARAM_VALUES` a request carries as OpenAI params, in
/// that table's order, which is the order Python maps them in.
pub const OPENAI_PARAMS: &[&str] = &[
    "functions",
    "function_call",
    "temperature",
    "top_p",
    "n",
    "stream",
    "stream_options",
    "stop",
    "max_tokens",
    "max_completion_tokens",
    "modalities",
    "prediction",
    "audio",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "user",
    "response_format",
    "seed",
    "tools",
    "tool_choice",
    "max_retries",
    "logprobs",
    "top_logprobs",
    "extra_headers",
    "parallel_tool_calls",
    "reasoning_effort",
    "verbosity",
    "thinking",
    "web_search_options",
    "include_server_side_tool_invocations",
    "service_tier",
    "safety_identifier",
    "prompt_cache_key",
    "prompt_cache_retention",
    "store",
    "metadata",
    "context_management",
];

/// `PROVIDER_UNVALIDATED_PARAMS`: never refused, only left out when unsupported.
pub const UNVALIDATED_PARAMS: &[&str] = &["user", "stream_options", "stream", "max_retries"];

/// The tool Python adds when `response_format` asks for JSON and the model has no native
/// structured output.
pub const RESPONSE_FORMAT_TOOL_NAME: &str = "json_tool_call";

/// Python's `DEFAULT_MAX_TOKENS`, added to a thinking budget when the caller set no limit.
pub const DEFAULT_MAX_TOKENS: u64 = 4096;

/// The request's OpenAI params in `DEFAULT_CHAT_COMPLETION_PARAM_VALUES` order.
pub fn ordered(params: &Map<String, Value>) -> impl Iterator<Item = (&'static str, &Value)> {
    OPENAI_PARAMS
        .iter()
        .filter_map(|name| params.get(*name).map(|value| (*name, value)))
}

/// Python's truthiness for a JSON value.
pub fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64() != Some(0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(object) => !object.is_empty(),
    }
}

/// `normalize_drop_params`: a boolean, or a config's `"true"`.
pub fn request_drop_params(value: Option<Value>) -> bool {
    match value {
        Some(Value::Bool(flag)) => flag,
        Some(Value::String(text)) => text.trim().eq_ignore_ascii_case("true"),
        _ => false,
    }
}

pub fn string_list(value: Option<Value>) -> Vec<String> {
    match value {
        Some(Value::Array(items)) => items
            .into_iter()
            .filter_map(|item| match item {
                Value::String(text) => Some(text),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// What the host tells a config about the request beyond its params.
#[derive(Clone, Copy, Default)]
pub struct ChatRequestContext<'a> {
    /// `drop_params` from the proxy settings or the deployment. `None` when the host cannot
    /// say, as when Python hosts the call: a request Python would drop or refuse a param of
    /// then goes back to Python instead.
    pub drop_params: Option<bool>,
    /// The model cost map, which several param rules read.
    pub model_info: Option<&'a dyn ModelInfo>,
}

/// Whether a param the provider does not take is dropped, refused, or left to Python.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DropPolicy(Option<bool>);

impl DropPolicy {
    pub fn new(drop_params: Option<bool>) -> Self {
        Self(drop_params)
    }

    /// Whether unsupported params are dropped.
    pub fn drops(self) -> bool {
        self.0 == Some(true)
    }

    /// Whether the host said what `drop_params` is.
    pub fn known(self) -> bool {
        self.0.is_some()
    }

    /// A param Python drops under `drop_params` and refuses otherwise: `Ok` means drop it.
    pub fn unsupported(self, refusal: impl FnOnce() -> Error) -> Result<(), Error> {
        match self.0 {
            Some(true) => Ok(()),
            Some(false) => Err(refusal()),
            None => Err(Error::Unsupported(
                "a param Python drops or refuses by drop_params",
            )),
        }
    }
}

/// Python's `UnsupportedParamsError` for one param with its own message.
pub fn unsupported_value(message: impl Into<String>) -> Error {
    Error::InvalidRequest(ErrorDetail::Message(message.into()))
}

/// A request's params as `get_optional_params` sorts them.
#[derive(Debug, Default)]
pub struct SortedParams {
    /// The OpenAI params, refused or dropped ones removed.
    pub openai: Map<String, Value>,
    /// Every other key, sent to the provider as given.
    pub provider: Map<String, Value>,
    /// Keys the config consumes as call configuration.
    pub config: Map<String, Value>,
    /// `allowed_openai_params`: sent as given even when the provider does not list them.
    pub allowed: Vec<String>,
    pub drop: DropPolicy,
}

/// Sorts `params` and refuses or drops the OpenAI params `supported` leaves out.
pub fn sort_params(
    params: Map<String, Value>,
    config_params: &[&str],
    context: &ChatRequestContext<'_>,
    supported: impl FnOnce(&Map<String, Value>) -> Result<Vec<&'static str>, Error>,
    provider: &'static str,
    model: &str,
) -> Result<SortedParams, Error> {
    let mut params = params;
    let request_drop = request_drop_params(params.remove("drop_params"));
    let drop = DropPolicy::new(match context.drop_params {
        Some(configured) => Some(configured || request_drop),
        None if request_drop => Some(true),
        None => None,
    });
    let allowed = string_list(params.remove("allowed_openai_params"));
    let additional_drop = string_list(params.remove("additional_drop_params"));
    let mut sorted = SortedParams {
        allowed,
        drop,
        ..SortedParams::default()
    };
    for (name, value) in params {
        if config_params.contains(&name.as_str()) {
            sorted.config.insert(name, value);
            continue;
        }
        if value.is_null() || is_litellm_owned(&name) || additional_drop.contains(&name) {
            continue;
        }
        match OPENAI_PARAMS.contains(&name.as_str()) {
            true => sorted.openai.insert(name, value),
            false => sorted.provider.insert(name, value),
        };
    }
    let supported = supported(&sorted.openai)?;
    let unsupported: Vec<String> = sorted
        .openai
        .iter()
        .filter(|(name, _)| {
            !supported.contains(&name.as_str()) && !sorted.allowed.iter().any(|a| a == *name)
        })
        .map(|(name, _)| name.clone())
        .collect();
    let refused: Vec<String> = unsupported
        .iter()
        .filter(|name| {
            !UNVALIDATED_PARAMS.contains(&name.as_str())
                && !(name.as_str() == "n"
                    && sorted.openai.get(name.as_str()).and_then(Value::as_f64) == Some(1.0))
        })
        .cloned()
        .collect();
    if !refused.is_empty() {
        drop.unsupported(|| {
            Error::InvalidRequest(ErrorDetail::UnsupportedParams {
                provider,
                params: refused.clone(),
                model: model.to_string(),
            })
        })?;
    }
    for name in unsupported {
        sorted.openai.remove(&name);
    }
    Ok(sorted)
}

/// A capability flag the cost map sets for the first of `candidates` it knows.
///
/// Without a cost map Python's own lookups have nothing to go on either, so the answer is
/// `None`, except when the host could still fall back to Python, which does have one.
pub fn capability(
    context: &ChatRequestContext<'_>,
    candidates: &[String],
    key: &str,
) -> Result<Option<bool>, Error> {
    let Some(info) = context.model_info else {
        return match context.drop_params {
            None => Err(Error::Unsupported(
                "a request whose translation reads the model cost map",
            )),
            Some(_) => Ok(None),
        };
    };
    Ok(candidates
        .iter()
        .filter_map(|candidate| info.entry(candidate))
        .find_map(|entry| entry.get(key).and_then(Value::as_bool)))
}

/// A numeric field of the first of `candidates` the cost map knows.
pub fn model_field(
    context: &ChatRequestContext<'_>,
    candidates: &[String],
    key: &str,
) -> Option<Value> {
    let info = context.model_info?;
    candidates
        .iter()
        .filter_map(|candidate| info.entry(candidate))
        .find_map(|entry| entry.get(key).filter(|value| !value.is_null()).cloned())
}

/// Whether the cost map has an entry for any of `candidates`.
pub fn model_known(context: &ChatRequestContext<'_>, candidates: &[String]) -> bool {
    context
        .model_info
        .is_some_and(|info| candidates.iter().any(|c| info.entry(c).is_some()))
}

/// `json.dumps(value)` with Python's default separators and ASCII escaping.
pub fn python_dumps(value: &Value) -> String {
    litellm_python_compat::json::dumps(&litellm_python_compat::json::from_json(value.clone()))
        .unwrap_or_else(|_| value.to_string())
}

/// `json.loads` of tool call arguments, as `parse_tool_call_arguments` reads them: empty
/// arguments are `{}`.
pub fn parse_tool_arguments(arguments: Option<&Value>) -> Result<Value, Error> {
    match arguments {
        None | Some(Value::Null) => Ok(Value::Object(Map::new())),
        Some(Value::String(text)) if text.trim().is_empty() => Ok(Value::Object(Map::new())),
        Some(Value::String(text)) => serde_json::from_str(text).map_err(|error| {
            Error::InvalidRequest(ErrorDetail::Message(format!(
                "tool call arguments are not valid JSON: {error}"
            )))
        }),
        Some(other) => Ok(other.clone()),
    }
}

/// `is_thinking_enabled` over a params map.
pub fn is_thinking_enabled(params: &Map<String, Value>) -> bool {
    let thinking = params.get("thinking");
    thinking == Some(&Value::Bool(true))
        || thinking
            .and_then(|thinking| thinking.get("type"))
            .and_then(Value::as_str)
            == Some("enabled")
        || params
            .get("reasoning_effort")
            .is_some_and(|effort| !effort.is_null())
}

/// The JSON schema a `response_format` names: `response_schema`, or `json_schema.schema`.
pub fn response_format_schema(value: &Value) -> Option<Value> {
    if let Some(schema) = value.get("response_schema") {
        return Some(schema.clone());
    }
    value
        .get("json_schema")
        .and_then(|json_schema| json_schema.get("schema"))
        .cloned()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn object(value: Value) -> Map<String, Value> {
        match value {
            Value::Object(map) => map,
            other => panic!("expected an object, got {other}"),
        }
    }

    fn sort(params: Value, drop_params: Option<bool>) -> Result<SortedParams, Error> {
        sort_params(
            object(params),
            &["aws_region_name"],
            &ChatRequestContext {
                drop_params,
                model_info: None,
            },
            |_| Ok(vec!["temperature", "max_tokens"]),
            "anthropic",
            "claude",
        )
    }

    #[rstest]
    fn params_sort_into_openai_provider_config_and_owned() {
        let sorted = sort(
            json!({"temperature": 0.5, "top_k": 3, "aws_region_name": "us-east-1",
                "metadata": {"x": 1}, "mock_response": "hi", "_litellm_x": 1, "user": "u", "seed": null}),
            Some(false),
        )
        .unwrap();
        assert_eq!(sorted.openai, object(json!({"temperature": 0.5})));
        assert_eq!(sorted.provider, object(json!({"top_k": 3})));
        assert_eq!(
            sorted.config,
            object(json!({"aws_region_name": "us-east-1"}))
        );
    }

    #[rstest]
    #[case::refused(Some(false), false)]
    #[case::dropped(Some(true), true)]
    #[case::request_drop(None, true)]
    fn an_unsupported_param_is_refused_or_dropped(
        #[case] drop_params: Option<bool>,
        #[case] dropped: bool,
    ) {
        let mut body = json!({"temperature": 0.5, "seed": 3});
        if drop_params.is_none() {
            body["drop_params"] = json!(true);
        }
        let result = sort(body, drop_params);
        match dropped {
            true => assert_eq!(result.unwrap().openai, object(json!({"temperature": 0.5}))),
            false => assert!(matches!(
                result,
                Err(Error::InvalidRequest(ErrorDetail::UnsupportedParams { ref params, .. }))
                    if params == &vec!["seed".to_string()]
            )),
        }
    }

    #[rstest]
    fn an_unknown_drop_policy_falls_back() {
        assert!(matches!(
            sort(json!({"seed": 3}), None),
            Err(Error::Unsupported(_))
        ));
    }

    #[rstest]
    fn allowed_and_additional_drop_params_are_honoured() {
        let sorted = sort(
            json!({"seed": 3, "top_k": 2, "allowed_openai_params": ["seed"], "additional_drop_params": ["top_k"]}),
            Some(false),
        )
        .unwrap();
        assert_eq!(sorted.openai, object(json!({"seed": 3})));
        assert!(sorted.provider.is_empty());
        assert_eq!(sorted.allowed, vec!["seed".to_string()]);
    }

    #[rstest]
    #[case::n_of_one(json!({"n": 1}), true)]
    #[case::n_of_two(json!({"n": 2}), false)]
    #[case::user(json!({"user": "u"}), true)]
    fn unvalidated_params_and_one_choice_are_never_refused(
        #[case] body: Value,
        #[case] accepted: bool,
    ) {
        let result = sort(body, Some(false));
        assert_eq!(result.is_ok(), accepted, "{result:?}");
        if let Ok(sorted) = result {
            assert!(sorted.openai.is_empty());
        }
    }

    #[rstest]
    fn python_dumps_matches_json_dumps() {
        assert_eq!(
            python_dumps(&json!({"b": "é", "a": [1, 2.5]})),
            r#"{"b": "\u00e9", "a": [1, 2.5]}"#
        );
    }
}
