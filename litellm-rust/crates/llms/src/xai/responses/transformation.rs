//! xAI Responses, matching `litellm/llms/xai/responses/transformation.py`

use litellm_auth::{CredentialPlacement, SecretValue};
use litellm_llms_types::formats::responses::ResponsesApiResponse;
use litellm_python_compat::truthy::json_truthy;
use serde_json::{Map, Value, json};

use crate::{
    Error,
    base_llm::{
        auth::{AuthScheme, ValidatedEnvironment},
        responses::transformation::BaseResponsesApiConfig,
    },
    openai::responses::{
        request::ResponsesHost,
        transformation::{responses_request, responses_response},
    },
};

const API_KEY_ENV: &str = "XAI_API_KEY";
const API_BASE_ENV: &str = "XAI_API_BASE";
const DEFAULT_BASE: &str = "https://api.x.ai/v1";
/// xAI bills in ticks of a dollar: https://docs.x.ai/developers/cost-tracking
const USD_TICKS_PER_DOLLAR: f64 = 10_000_000_000.0;
const WEB_SEARCH_FILTERS: [&str; 2] = ["allowed_domains", "excluded_domains"];
const X_SEARCH_FIELDS: [&str; 6] = [
    "allowed_x_handles",
    "excluded_x_handles",
    "from_date",
    "to_date",
    "enable_image_understanding",
    "enable_video_understanding",
];

pub struct XaiResponsesApiConfig;
pub const XAI_RESPONSES_CONFIG: XaiResponsesApiConfig = XaiResponsesApiConfig;

impl BaseResponsesApiConfig for XaiResponsesApiConfig {
    fn secret_names(
        &self,
        api_key: Option<&str>,
        api_base: Option<&str>,
    ) -> &'static [&'static str] {
        match (
            api_key.is_some_and(|key| !key.is_empty()),
            api_base.is_some_and(|base| !base.is_empty()),
        ) {
            (true, true) => &[],
            (true, false) => &[API_BASE_ENV],
            (false, true) => &[API_KEY_ENV],
            (false, false) => &[API_KEY_ENV, API_BASE_ENV],
        }
    }

    fn validate_environment(
        &self,
        headers: Vec<(String, String)>,
        api_key: Option<&str>,
        lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        let key =
            resolved(api_key, lookup, API_KEY_ENV).ok_or(litellm_auth::Error::MissingApiKey {
                provider: "XAI",
                environment_variable: API_KEY_ENV,
            })?;
        Ok(ValidatedEnvironment {
            headers,
            auth: AuthScheme::Credential {
                placement: CredentialPlacement::Bearer,
                secret: SecretValue::new(key),
            },
        })
    }

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        _api_version: Option<&str>,
        lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        let api_base =
            resolved(api_base, lookup, API_BASE_ENV).unwrap_or_else(|| DEFAULT_BASE.into());
        Ok(format!("{}/responses", api_base.trim_end_matches('/')))
    }

    fn transform_responses_api_request(
        &self,
        model: &str,
        input: Value,
        mut params: Map<String, Value>,
    ) -> Result<Value, Error> {
        params.remove("metadata");
        if let Some(tools) = params.get_mut("tools").filter(|tools| json_truthy(tools)) {
            let listed = match tools.take() {
                Value::Array(tools) => tools,
                tool => vec![tool],
            };
            *tools = Value::Array(listed.into_iter().map(xai_tool).collect());
        }
        responses_request(model, input, params, ResponsesHost::Xai)
    }

    fn transform_response_api_response(&self, body: Value) -> Result<ResponsesApiResponse, Error> {
        let mut response = responses_response(body)?;
        if let Some(Value::Object(usage)) = response.extra.get_mut("usage")
            && let Some(cost) = reported_cost(usage.get("cost_in_usd_ticks"))
        {
            usage.insert("cost".into(), json!(cost));
        }
        Ok(response)
    }
}

/// Python's `value or get_secret_str(name)`: an empty value falls through, and nothing is trimmed
fn resolved(
    value: Option<&str>,
    lookup: &dyn Fn(&str) -> Option<String>,
    name: &str,
) -> Option<String> {
    value
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .or_else(|| lookup(name).filter(|value| !value.is_empty()))
}

/// Python's `xai_reported_cost_in_usd`: only a non-negative integer tick count is a cost
fn reported_cost(ticks: Option<&Value>) -> Option<f64> {
    ticks?
        .as_u64()
        .map(|ticks| ticks as f64 / USD_TICKS_PER_DOLLAR)
}

/// xAI takes `code_interpreter` without a container, drops OpenAI's `search_context_size` from
/// `web_search`, and keeps only the `x_search` fields it documents
fn xai_tool(tool: Value) -> Value {
    let Value::Object(fields) = tool else {
        return tool;
    };
    match fields.get("type").and_then(Value::as_str) {
        Some("code_interpreter") => json!({"type": "code_interpreter"}),
        Some("web_search") => web_search_tool(&fields),
        Some("x_search") => {
            let mut xai = Map::from_iter([("type".into(), json!("x_search"))]);
            for name in X_SEARCH_FIELDS {
                if let Some(value) = fields.get(name) {
                    xai.insert(name.into(), value.clone());
                }
            }
            Value::Object(xai)
        }
        _ => Value::Object(fields),
    }
}

/// Domains may come nested under `filters`, the documented shape, or flat on the tool
fn web_search_tool(tool: &Map<String, Value>) -> Value {
    let domains = match tool.get("filters") {
        Some(Value::Object(filters)) => filters,
        _ => tool,
    };
    let filters: Map<String, Value> = WEB_SEARCH_FILTERS
        .iter()
        .filter_map(|name| Some(((*name).into(), domains.get(*name)?.clone())))
        .collect();
    let mut xai = Map::from_iter([("type".into(), json!("web_search"))]);
    if !filters.is_empty() {
        xai.insert("filters".into(), Value::Object(filters));
    }
    if let Some(value) = tool.get("enable_image_understanding") {
        xai.insert("enable_image_understanding".into(), value.clone());
    }
    Value::Object(xai)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn env(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |name| {
            pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| (*value).to_owned())
        }
    }

    #[rstest]
    #[case::default(None, &[], "https://api.x.ai/v1/responses")]
    #[case::explicit(Some("https://proxy.test/v1/"), &[], "https://proxy.test/v1/responses")]
    #[case::secret(None, &[("XAI_API_BASE", "https://secret.test/v1")], "https://secret.test/v1/responses")]
    #[case::openai_base_is_ignored(None, &[("OPENAI_BASE_URL", "https://openai.test/v1")], "https://api.x.ai/v1/responses")]
    fn url_is_the_base_plus_responses(
        #[case] api_base: Option<&str>,
        #[case] secrets: &'static [(&'static str, &'static str)],
        #[case] expected: &str,
    ) {
        assert_eq!(
            XAI_RESPONSES_CONFIG
                .get_complete_url(api_base, None, &env(secrets))
                .unwrap(),
            expected
        );
    }

    #[test]
    fn a_missing_key_is_an_auth_error() {
        assert!(matches!(
            XAI_RESPONSES_CONFIG.validate_environment(vec![], None, &env(&[])),
            Err(Error::Auth(litellm_auth::Error::MissingApiKey { .. }))
        ));
    }

    #[test]
    fn request_drops_metadata_and_keeps_max_output_tokens() {
        let body = XAI_RESPONSES_CONFIG
            .transform_responses_api_request(
                "grok-4",
                json!("hi"),
                Map::from_iter([
                    ("metadata".into(), json!({"user": "u"})),
                    ("max_output_tokens".into(), json!(4)),
                ]),
            )
            .unwrap();
        assert_eq!(
            body,
            json!({"max_output_tokens": 4, "model": "grok-4", "input": "hi"})
        );
    }

    #[rstest]
    #[case::code_interpreter(
        json!({"type": "code_interpreter", "container": {"type": "auto"}}),
        json!({"type": "code_interpreter"})
    )]
    #[case::web_search_nested(
        json!({"type": "web_search", "search_context_size": "high", "filters": {"allowed_domains": ["a.com"], "other": 1}}),
        json!({"type": "web_search", "filters": {"allowed_domains": ["a.com"]}})
    )]
    #[case::web_search_flat(
        json!({"type": "web_search", "excluded_domains": ["b.com"], "enable_image_understanding": true}),
        json!({"type": "web_search", "filters": {"excluded_domains": ["b.com"]}, "enable_image_understanding": true})
    )]
    #[case::x_search(
        json!({"type": "x_search", "from_date": "2025-01-01", "unknown": 1}),
        json!({"type": "x_search", "from_date": "2025-01-01"})
    )]
    #[case::function(
        json!({"type": "function", "name": "f", "parameters": {"type": "object"}}),
        json!({"type": "function", "name": "f", "parameters": {"type": "object"}})
    )]
    fn tools_take_the_xai_shape(#[case] tool: Value, #[case] expected: Value) {
        let body = XAI_RESPONSES_CONFIG
            .transform_responses_api_request(
                "grok-4",
                json!("hi"),
                Map::from_iter([("tools".into(), json!([tool]))]),
            )
            .unwrap();
        assert_eq!(body["tools"], json!([expected]));
    }

    #[rstest]
    #[case::ticks(json!(12_345_000_000u64), Some(1.2345))]
    #[case::negative(json!(-1), None)]
    #[case::float(json!(1.0), None)]
    #[case::string(json!("5"), None)]
    fn usage_cost_comes_from_reported_ticks(#[case] ticks: Value, #[case] expected: Option<f64>) {
        let response = XAI_RESPONSES_CONFIG
            .transform_response_api_response(json!({
                "id": "resp_1",
                "model": "grok-4",
                "output": [],
                "usage": {"input_tokens": 1, "cost_in_usd_ticks": ticks},
            }))
            .unwrap();
        assert_eq!(
            response.extra["usage"].get("cost").and_then(Value::as_f64),
            expected
        );
    }
}
