use litellm_auth::{CredentialPlacement, SecretValue};
use litellm_auth_aws::{
    AwsCredentialSource, bedrock_model_id_and_region,
    constants::{
        AWS_ACCESS_KEY_ID, AWS_BEARER_TOKEN_BEDROCK, AWS_EXTERNAL_ID, AWS_PROFILE_NAME, AWS_REGION,
        AWS_REGION_NAME, AWS_ROLE_NAME, AWS_SECRET_ACCESS_KEY, AWS_SESSION_NAME, AWS_SESSION_TOKEN,
        AWS_STS_ENDPOINT, AWS_WEB_IDENTITY_TOKEN, BEDROCK_SERVICE, DEFAULT_BEDROCK_REGION,
    },
    is_sigv4_computed_header,
};
use litellm_llms_types::{
    formats::messages::{MessagesOptionalParams, MessagesRequest, MessagesResponse, MessagesUsage},
    providers::anthropic::BetaProvider,
    recognized::Recognized,
    serde_compat::Nullable,
};
use serde_json::Map;
use url::Url;

use crate::{
    Error,
    anthropic::{
        beta_headers::BetaPolicy,
        common_utils::DEFAULT_ANTHROPIC_HEADERS,
        messages::{
            handler::shape_anthropic_messages_request,
            transformation::update_headers_with_anthropic_beta,
        },
    },
    base_llm::{
        auth::{AuthScheme, Headers, ValidatedEnvironment},
        messages::{
            context::MessagesTransformContext,
            transformation::{BaseMessagesConfig, MessagesWireRequest, messages_request_body},
        },
    },
    bedrock::messages::{
        connection::BedrockMessagesConnection,
        invoke_transformations::anthropic_claude3_transformation::{
            BedrockMessagesSurface, bedrock_body_field, bedrock_feature_betas,
            transform_bedrock_messages_request,
        },
    },
};

pub const MANTLE_MESSAGES_PATH: &str = "/anthropic/v1/messages";
pub const BEDROCK_MANTLE_API_BASE_ENV: &str = "BEDROCK_MANTLE_API_BASE";

const MANTLE_MODEL_PREFIX: &str = "mantle/";
const MANTLE_HOST_PREFIX: &str = "bedrock-mantle.";
const MANTLE_HOST_SUFFIX: &str = ".api.aws";
const BETA_POLICY: BetaPolicy = BetaPolicy::Filter(BetaProvider::BedrockMantle);
const HEADER_ONLY_BODY_FIELDS: [&str; 2] = ["anthropic_version", "anthropic_beta"];
const API_BASE_SUFFIXES: [&str; 7] = [
    MANTLE_MESSAGES_PATH,
    "/v1/messages",
    "/messages",
    "/anthropic/v1",
    "/openai/v1",
    "/v1",
    "/anthropic",
];

/// Where one LiteLLM provider spelling of the mantle endpoint reads its credentials and
/// endpoint from. The wire format is the same for every spelling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MantleSettings {
    pub bearer_token_envs: &'static [&'static str],
    pub api_base_env: &'static str,
    pub region_envs: &'static [&'static str],
    pub default_region: &'static str,
    pub secret_names: &'static [&'static str],
}

pub const BEDROCK_SETTINGS: MantleSettings = MantleSettings {
    bearer_token_envs: &[AWS_BEARER_TOKEN_BEDROCK],
    api_base_env: BEDROCK_MANTLE_API_BASE_ENV,
    region_envs: &[AWS_REGION_NAME, AWS_REGION],
    default_region: DEFAULT_BEDROCK_REGION,
    secret_names: &[
        AWS_BEARER_TOKEN_BEDROCK,
        BEDROCK_MANTLE_API_BASE_ENV,
        AWS_REGION_NAME,
        AWS_REGION,
        AWS_ACCESS_KEY_ID,
        AWS_SECRET_ACCESS_KEY,
        AWS_SESSION_TOKEN,
        AWS_SESSION_NAME,
        AWS_PROFILE_NAME,
        AWS_ROLE_NAME,
        AWS_WEB_IDENTITY_TOKEN,
        AWS_STS_ENDPOINT,
        AWS_EXTERNAL_ID,
    ],
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AmazonMantleMessagesConfig {
    pub settings: MantleSettings,
}

pub const AMAZON_MANTLE_MESSAGES_CONFIG: AmazonMantleMessagesConfig = AmazonMantleMessagesConfig {
    settings: BEDROCK_SETTINGS,
};

fn non_blank(value: &str) -> Option<String> {
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_string())
}

fn first_env(names: &[&str], env_lookup: &dyn Fn(&str) -> Option<String>) -> Option<String> {
    names
        .iter()
        .find_map(|name| env_lookup(name).as_deref().and_then(non_blank))
}

/// The region in a public `bedrock-mantle.<region>.api.aws` host.
pub fn mantle_host_region(api_base: &str) -> Option<String> {
    let url = Url::parse(api_base.trim()).ok()?;
    if !matches!(url.scheme(), "http" | "https") {
        return None;
    }
    url.host_str()?
        .strip_prefix(MANTLE_HOST_PREFIX)?
        .strip_suffix(MANTLE_HOST_SUFFIX)
        .filter(|region| !region.is_empty() && !region.contains('.'))
        .map(str::to_string)
}

fn mantle_messages_url(api_base: &str) -> String {
    let base = api_base.trim().trim_end_matches('/');
    let host = API_BASE_SUFFIXES
        .iter()
        .find_map(|suffix| base.strip_suffix(suffix))
        .unwrap_or(base);
    format!("{host}{MANTLE_MESSAGES_PATH}")
}

impl AmazonMantleMessagesConfig {
    /// The caller's `api_base`, else the settings' env override.
    pub fn api_base(
        &self,
        api_base: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Option<String> {
        api_base
            .and_then(non_blank)
            .or_else(|| first_env(&[self.settings.api_base_env], env_lookup))
    }

    /// A public mantle host names its own region, which must also scope the SigV4
    /// credential. Any other endpoint uses the configured region.
    pub fn region(
        &self,
        api_base: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> String {
        self.api_base(api_base, env_lookup)
            .as_deref()
            .and_then(mantle_host_region)
            .or_else(|| first_env(self.settings.region_envs, env_lookup))
            .unwrap_or_else(|| self.settings.default_region.to_string())
    }

    pub fn messages_url(
        &self,
        api_base: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> String {
        match self.api_base(api_base, env_lookup) {
            Some(api_base) => mantle_messages_url(&api_base),
            None => format!(
                "https://{MANTLE_HOST_PREFIX}{}{MANTLE_HOST_SUFFIX}{MANTLE_MESSAGES_PATH}",
                self.region(None, env_lookup)
            ),
        }
    }

    pub fn bearer_token(
        &self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Option<String> {
        api_key
            .and_then(non_blank)
            .or_else(|| first_env(self.settings.bearer_token_envs, env_lookup))
    }
}

impl AmazonMantleMessagesConfig {
    pub fn get_complete_url_with_connection(
        &self,
        connection: &BedrockMessagesConnection,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        let region = connection
            .region
            .clone()
            .unwrap_or_else(|| self.region(connection.api_base.as_deref(), env_lookup));
        let base = self.api_base(connection.api_base.as_deref(), env_lookup);
        match base {
            Some(base) if mantle_host_region(&base).is_none() => Ok(mantle_messages_url(&base)),
            _ => Ok(format!(
                "https://{MANTLE_HOST_PREFIX}{region}{MANTLE_HOST_SUFFIX}{MANTLE_MESSAGES_PATH}"
            )),
        }
    }

    pub fn validate_environment_with_connection(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        connection: &BedrockMessagesConnection,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        let headers = connection.headers(headers);
        if let Some(token) = self.bearer_token(api_key, env_lookup) {
            return Ok(ValidatedEnvironment {
                headers,
                auth: AuthScheme::Credential {
                    placement: CredentialPlacement::Bearer,
                    secret: SecretValue::new(token),
                },
            });
        }
        let url = self.get_complete_url_with_connection(connection, env_lookup)?;
        let region = mantle_host_region(&url)
            .or(connection.region.clone())
            .unwrap_or_else(|| self.region(connection.api_base.as_deref(), env_lookup));
        Ok(ValidatedEnvironment {
            headers: headers
                .into_iter()
                .filter(|(name, _)| !is_sigv4_computed_header(name))
                .collect(),
            auth: AuthScheme::AwsSigV4 {
                region,
                service: BEDROCK_SERVICE,
                credentials: Box::new(AwsCredentialSource::from_params(&Map::new(), env_lookup)),
            },
        })
    }
}

impl BaseMessagesConfig for AmazonMantleMessagesConfig {
    fn shape_request(
        &self,
        request: MessagesRequest,
        reasoning_auto_summary: bool,
    ) -> Result<MessagesRequest, Error> {
        shape_anthropic_messages_request(request, reasoning_auto_summary)
    }

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        let (_, region) =
            bedrock_model_id_and_region(model.strip_prefix(MANTLE_MODEL_PREFIX).unwrap_or(model));
        self.get_complete_url_with_connection(
            &BedrockMessagesConnection {
                api_base: api_base.map(str::to_string),
                region,
                ..Default::default()
            },
            env_lookup,
        )
    }

    fn transform_anthropic_messages_request(
        &self,
        request: MessagesRequest,
        context: &MessagesTransformContext,
    ) -> Result<MessagesRequest, Error> {
        let (model, _) = bedrock_model_id_and_region(
            request
                .model
                .strip_prefix(MANTLE_MODEL_PREFIX)
                .unwrap_or(&request.model),
        );
        let request = transform_bedrock_messages_request(
            MessagesRequest { model, ..request },
            context,
            BedrockMessagesSurface::Mantle,
        )?;
        Ok(MessagesRequest {
            params: MessagesOptionalParams {
                stream: request.params.stream.filter(|stream| *stream),
                ..request.params
            },
            ..request
        })
    }

    fn prepare_wire_request(
        &self,
        request: &MessagesRequest,
        headers: Headers,
    ) -> Result<MessagesWireRequest, Error> {
        let derived = request
            .params
            .extra
            .get("anthropic_beta")
            .and_then(serde_json::Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(serde_json::Value::as_str)
            .map(|beta| beta.parse().unwrap_or_else(|never| match never {}))
            .collect();
        let headers = crate::anthropic::common_utils::merge_beta_headers(headers, derived);
        let headers = BETA_POLICY.apply(crate::anthropic::common_utils::merge_beta_headers(
            headers,
            bedrock_feature_betas(request),
        ));
        let serde_json::Value::Object(body) = messages_request_body(request)? else {
            unreachable!("MessagesRequest serializes as an object")
        };
        Ok(MessagesWireRequest {
            headers,
            body: serde_json::Value::Object(
                body.into_iter()
                    .filter(|(key, _)| {
                        !HEADER_ONLY_BODY_FIELDS.contains(&key.as_str())
                            && (bedrock_body_field(key)
                                || matches!(key.as_str(), "model" | "stream"))
                    })
                    .collect(),
            ),
        })
    }

    fn transform_anthropic_messages_response(
        &self,
        _model: &str,
        response: MessagesResponse,
    ) -> Result<MessagesResponse, Error> {
        let usage = match response.usage {
            None => Recognized::Known(with_token_counts(MessagesUsage::default())),
            Some(Recognized::Known(usage)) => Recognized::Known(with_token_counts(usage)),
            Some(unrecognized) => unrecognized,
        };
        Ok(MessagesResponse {
            usage: Some(usage),
            ..response
        })
    }

    fn secret_names(&self) -> &'static [&'static str] {
        self.settings.secret_names
    }

    /// A bearer token is the credential when one resolves. Otherwise the request is signed
    /// with SigV4 for the region the URL's host names.
    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        let (_, region) =
            bedrock_model_id_and_region(model.strip_prefix(MANTLE_MODEL_PREFIX).unwrap_or(model));
        self.validate_environment_with_connection(
            headers,
            api_key,
            &BedrockMessagesConnection {
                region,
                ..Default::default()
            },
            env_lookup,
        )
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        DEFAULT_ANTHROPIC_HEADERS
    }

    fn request_headers(&self, headers: Headers, request: &MessagesRequest) -> Headers {
        BETA_POLICY.apply(update_headers_with_anthropic_beta(headers, request))
    }
}

fn with_token_counts(usage: MessagesUsage) -> MessagesUsage {
    MessagesUsage {
        input_tokens: usage.input_tokens.or(Some(Nullable::Value(0))),
        output_tokens: usage.output_tokens.or(Some(Nullable::Value(0))),
        ..usage
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use litellm_auth_aws::constants::SIGV4_COMPUTED_HEADER_NAMES;
    use litellm_llms_types::providers::anthropic::{AnthropicBeta, BetaSet};
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;
    use crate::{
        anthropic::messages::thinking::ANTHROPIC_MIN_THINKING_BUDGET_TOKENS,
        base_llm::messages::context::MessagesModelCapabilities,
    };

    const CONFIG: AmazonMantleMessagesConfig = AMAZON_MANTLE_MESSAGES_CONFIG;

    fn env_from(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        |name: &str| {
            pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        }
    }

    fn no_env(_: &str) -> Option<String> {
        None
    }

    fn request(fields: Value) -> MessagesRequest {
        let Value::Object(fields) = fields else {
            panic!("case fields are an object")
        };
        serde_json::from_value(Value::Object(
            [
                ("model".to_string(), json!("mantle/anthropic.claude-test")),
                ("max_tokens".to_string(), json!(4096)),
                (
                    "messages".to_string(),
                    json!([{"role": "user", "content": "hi"}]),
                ),
            ]
            .into_iter()
            .chain(fields)
            .collect(),
        ))
        .unwrap()
    }

    fn transformed_with(fields: Value, capabilities: MessagesModelCapabilities) -> Value {
        let context = MessagesTransformContext::with_lookup(capabilities, false, &|_: &str| None);
        let request = CONFIG
            .transform_anthropic_messages_request(request(fields), &context)
            .unwrap();
        CONFIG.prepare_wire_request(&request, vec![]).unwrap().body
    }

    fn transformed(fields: Value) -> Value {
        transformed_with(fields, MessagesModelCapabilities::default())
    }

    fn headers(pairs: &[(&str, &str)]) -> Headers {
        pairs
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect()
    }

    #[rstest]
    #[case::route_prefix("mantle/anthropic.claude-test", "anthropic.claude-test")]
    #[case::only_one_leading_segment("mantle/mantle/x", "mantle/x")]
    #[case::bare_model("anthropic.claude-test", "anthropic.claude-test")]
    #[case::regional_model("us-east-2/anthropic.claude-test", "anthropic.claude-test")]
    #[case::regional_mantle_model(
        "mantle/us-east-2/anthropic.claude-test",
        "anthropic.claude-test"
    )]
    fn the_model_goes_in_the_body_without_one_mantle_segment(
        #[case] model: &str,
        #[case] expected: &str,
    ) {
        assert_eq!(
            transformed(json!({"model": model}))["model"],
            json!(expected)
        );
    }

    #[rstest]
    #[case::streaming(json!({"stream": true}), Some(json!(true)))]
    #[case::not_streaming(json!({"stream": false}), None)]
    #[case::unset(json!({}), None)]
    fn stream_is_in_the_body_only_when_true(
        #[case] fields: Value,
        #[case] expected: Option<Value>,
    ) {
        assert_eq!(transformed(fields).get("stream").cloned(), expected);
    }

    #[rstest::rstest]
    fn anthropic_version_and_beta_never_reach_the_body() {
        let body = transformed(json!({
            "anthropic_version": "bedrock-2023-05-31",
            "anthropic_beta": ["context-1m-2025-08-07"],
            "custom_field": 1
        }));
        assert_eq!(body.get("anthropic_version"), None);
        assert_eq!(body.get("anthropic_beta"), None);
        assert!(body.get("custom_field").is_none());
    }

    #[rstest::rstest]
    fn mid_conversation_system_is_converted_without_changing_prefix() {
        let body = transformed(json!({
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "be terse"}
            ]
        }));
        assert_eq!(
            body["messages"],
            json!([
                {"role":"user", "content":"hi"},
                {"role":"user", "content":[
                    {"type":"text", "text":crate::base_llm::messages::normalization::CONVERTED_SYSTEM_NOTE},
                    {"type":"text", "text":"be terse"}
                ]}
            ])
        );
        assert!(body.get("system").is_none());
    }

    #[rstest]
    #[case::every_supported_edit_is_kept(
        json!({"edits": [{"type": "compact_20260112"}, {"type": "clear_tool_uses_20250919"}, {"type": "clear_thinking_20251015"}]}),
        Some(json!({"edits": [{"type": "compact_20260112"}, {"type": "clear_tool_uses_20250919"}, {"type": "clear_thinking_20251015"}]})),
    )]
    #[case::unknown_edits_are_dropped(
        json!({"edits": [{"type": "clear_tool_uses_20250919"}, {"type": "future_edit_2099"}], "x": 1}),
        Some(json!({"edits": [{"type": "clear_tool_uses_20250919"}], "x": 1})),
    )]
    #[case::nothing_left_drops_the_field(json!({"edits": [{"type": "future_edit_2099"}]}), None)]
    #[case::no_edits_drops_the_field(json!({"x": 1}), None)]
    fn context_management_keeps_only_the_edits_mantle_accepts(
        #[case] context_management: Value,
        #[case] expected: Option<Value>,
    ) {
        assert_eq!(
            transformed(json!({"context_management": context_management}))
                .get("context_management")
                .cloned(),
            expected
        );
    }

    fn reasoning() -> MessagesModelCapabilities {
        MessagesModelCapabilities {
            supports_reasoning: true,
            supports_legacy_thinking: true,
            ..MessagesModelCapabilities::default()
        }
    }

    fn adaptive() -> MessagesModelCapabilities {
        MessagesModelCapabilities {
            supports_reasoning: true,
            supports_adaptive_thinking: true,
            ..MessagesModelCapabilities::default()
        }
    }

    const CLEAR_THINKING: &str = "clear_thinking_20251015";

    #[rstest]
    #[case::legacy_model_gets_the_minimum_budget(
        json!({}),
        reasoning(),
        Some(json!({"type": "enabled", "budget_tokens": ANTHROPIC_MIN_THINKING_BUDGET_TOKENS})),
    )]
    #[case::disabled_thinking_is_replaced(
        json!({"thinking": {"type": "disabled"}}),
        reasoning(),
        Some(json!({"type": "enabled", "budget_tokens": ANTHROPIC_MIN_THINKING_BUDGET_TOKENS})),
    )]
    #[case::adaptive_model_gets_adaptive_thinking(json!({}), adaptive(), Some(json!({"type": "adaptive"})))]
    #[case::caller_thinking_is_kept(
        json!({"thinking": {"type": "enabled", "budget_tokens": 2000}}),
        reasoning(),
        Some(json!({"type": "enabled", "budget_tokens": 2000})),
    )]
    #[case::model_without_thinking_is_untouched(json!({}), MessagesModelCapabilities::default(), None)]
    #[case::no_room_for_the_budget(json!({"max_tokens": ANTHROPIC_MIN_THINKING_BUDGET_TOKENS}), reasoning(), None)]
    fn clear_thinking_turns_thinking_on(
        #[case] fields: Value,
        #[case] capabilities: MessagesModelCapabilities,
        #[case] expected: Option<Value>,
    ) {
        let Value::Object(fields) = fields else {
            panic!("case fields are an object")
        };
        let body = transformed_with(
            Value::Object(
                fields
                    .into_iter()
                    .chain([(
                        "context_management".to_string(),
                        json!({"edits": [{"type": CLEAR_THINKING}]}),
                    )])
                    .collect(),
            ),
            capabilities,
        );
        assert_eq!(body.get("thinking").cloned(), expected);
    }

    #[test]
    fn thinking_is_not_injected_without_clear_thinking() {
        let body = transformed_with(
            json!({"context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]}}),
            reasoning(),
        );
        assert_eq!(body.get("thinking"), None);
    }

    #[test]
    fn betas_are_filtered_against_the_mantle_column() {
        let mapped = |beta: &AnthropicBeta| beta.on(BetaProvider::BedrockMantle);
        let kept = AnthropicBeta::KNOWN
            .into_iter()
            .find(|beta| mapped(beta).is_some())
            .expect("the mantle column accepts some beta");
        let dropped = AnthropicBeta::KNOWN
            .into_iter()
            .find(|beta| mapped(beta).is_none())
            .expect("the mantle column rejects some beta");
        let sent = CONFIG.request_headers(
            headers(&[(
                "anthropic-beta",
                &format!("{kept},{dropped},example-beta-2099-01-01"),
            )]),
            &request(json!({})),
        );
        assert_eq!(
            sent,
            headers(&[("anthropic-beta", mapped(&kept).unwrap().as_str())])
        );
    }

    #[test]
    fn feature_betas_pass_through_the_same_filter() {
        let request = request(json!({
            "speed": "fast",
            "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]}
        }));
        let expected: BetaSet = [
            AnthropicBeta::FastMode20260201,
            AnthropicBeta::ContextManagement20250627,
        ]
        .iter()
        .filter_map(|beta| beta.on(BetaProvider::BedrockMantle))
        .collect();
        let sent = CONFIG.request_headers(Vec::new(), &request);
        let expected_headers = match expected.is_empty() {
            true => Vec::new(),
            false => headers(&[("anthropic-beta", &expected.to_string())]),
        };
        assert_eq!(sent, expected_headers);
    }

    #[test]
    fn the_header_is_dropped_when_no_beta_survives() {
        assert_eq!(
            CONFIG.request_headers(
                headers(&[
                    ("x-other", "1"),
                    ("Anthropic-Beta", "example-beta-2099-01-01")
                ]),
                &request(json!({}))
            ),
            headers(&[("x-other", "1")])
        );
    }

    #[test]
    fn anthropic_version_is_a_default_header() {
        assert!(
            CONFIG
                .default_headers()
                .iter()
                .any(|(name, _)| *name == "anthropic-version")
        );
    }

    #[test]
    fn streams_are_relayed_as_anthropic_sse() {
        assert!(CONFIG.stream_decoder().is_none());
    }

    #[rstest]
    #[case::host("https://vpce.example", "https://vpce.example/anthropic/v1/messages")]
    #[case::trailing_slash("https://vpce.example/", "https://vpce.example/anthropic/v1/messages")]
    #[case::complete(
        "https://vpce.example/anthropic/v1/messages",
        "https://vpce.example/anthropic/v1/messages"
    )]
    #[case::anthropic_base(
        "https://vpce.example/anthropic",
        "https://vpce.example/anthropic/v1/messages"
    )]
    #[case::anthropic_v1(
        "https://vpce.example/anthropic/v1/",
        "https://vpce.example/anthropic/v1/messages"
    )]
    #[case::openai_v1(
        "https://vpce.example/openai/v1",
        "https://vpce.example/anthropic/v1/messages"
    )]
    #[case::v1(
        "https://vpce.example/v1",
        "https://vpce.example/anthropic/v1/messages"
    )]
    #[case::v1_messages(
        "https://vpce.example/v1/messages",
        "https://vpce.example/anthropic/v1/messages"
    )]
    #[case::messages(
        "https://vpce.example/messages",
        "https://vpce.example/anthropic/v1/messages"
    )]
    fn an_override_gets_the_messages_path_exactly_once(#[case] base: &str, #[case] expected: &str) {
        assert_eq!(
            CONFIG.get_complete_url(Some(base), "m", &no_env).unwrap(),
            expected
        );
        let from_env =
            move |name: &str| (name == BEDROCK_MANTLE_API_BASE_ENV).then(|| base.to_string());
        assert_eq!(
            CONFIG.get_complete_url(None, "m", &from_env).unwrap(),
            expected
        );
    }

    #[test]
    fn the_caller_api_base_outranks_the_env_override() {
        let env = env_from(&[(BEDROCK_MANTLE_API_BASE_ENV, "https://env.example")]);
        assert_eq!(
            CONFIG
                .get_complete_url(Some("https://caller.example"), "m", &env)
                .unwrap(),
            "https://caller.example/anthropic/v1/messages"
        );
    }

    #[rstest]
    #[case::region_name(&[(AWS_REGION_NAME, "eu-west-1"), (AWS_REGION, "ap-south-1")], "eu-west-1")]
    #[case::region(&[(AWS_REGION, "ap-south-1")], "ap-south-1")]
    #[case::default(&[], DEFAULT_BEDROCK_REGION)]
    fn without_an_override_the_public_host_uses_the_configured_region(
        #[case] env: &'static [(&'static str, &'static str)],
        #[case] region: &str,
    ) {
        assert_eq!(
            CONFIG.get_complete_url(None, "m", &env_from(env)).unwrap(),
            format!("https://bedrock-mantle.{region}.api.aws/anthropic/v1/messages")
        );
    }

    #[rstest]
    #[case::public_host("https://bedrock-mantle.eu-central-1.api.aws/v1", Some("eu-central-1"))]
    #[case::uppercase_host("HTTPS://Bedrock-Mantle.EU-CENTRAL-1.api.aws", Some("eu-central-1"))]
    #[case::private_endpoint("https://vpce-1.bedrock-mantle.us-east-1.vpce.amazonaws.com", None)]
    #[case::lookalike("https://bedrock-mantle.eu-central-1.api.aws.example", None)]
    #[case::not_a_url("bedrock-mantle.eu-central-1.api.aws", None)]
    fn only_a_public_mantle_host_names_a_region(
        #[case] base: &str,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(mantle_host_region(base).as_deref(), expected);
    }

    fn sigv4_region(env: &dyn Fn(&str) -> Option<String>) -> String {
        match CONFIG
            .validate_environment(Vec::new(), None, "m", env)
            .unwrap()
            .auth
        {
            AuthScheme::AwsSigV4 {
                region, service, ..
            } => {
                assert_eq!(service, BEDROCK_SERVICE);
                region
            }
            other => panic!("expected SigV4, got {other:?}"),
        }
    }

    #[test]
    fn sigv4_scopes_to_the_region_in_the_url_host() {
        let env = env_from(&[
            (
                BEDROCK_MANTLE_API_BASE_ENV,
                "https://bedrock-mantle.eu-central-1.api.aws/openai/v1",
            ),
            (AWS_REGION_NAME, "us-west-2"),
        ]);
        let url = CONFIG.get_complete_url(None, "m", &env).unwrap();
        let region = sigv4_region(&env);
        assert_eq!(mantle_host_region(&url), Some(region.clone()));
        assert_eq!(region, "eu-central-1");
    }

    #[test]
    fn sigv4_uses_the_configured_region_for_a_private_endpoint() {
        let env = env_from(&[
            (BEDROCK_MANTLE_API_BASE_ENV, "https://vpce.example"),
            (AWS_REGION_NAME, "eu-west-1"),
        ]);
        assert_eq!(sigv4_region(&env), "eu-west-1");
    }

    #[test]
    fn the_default_url_and_sigv4_agree_on_the_region() {
        let env = env_from(&[(AWS_REGION, "ap-south-1")]);
        let url = CONFIG.get_complete_url(None, "m", &env).unwrap();
        assert_eq!(mantle_host_region(&url), Some(sigv4_region(&env)));
    }

    #[rstest]
    #[case::explicit_key(Some("key"), &[(AWS_BEARER_TOKEN_BEDROCK, "env-token")], Some("key"))]
    #[case::env_token(None, &[(AWS_BEARER_TOKEN_BEDROCK, "env-token")], Some("env-token"))]
    #[case::blank_key_falls_back_to_env(Some(" "), &[(AWS_BEARER_TOKEN_BEDROCK, "env-token")], Some("env-token"))]
    #[case::blank_env_signs(None, &[(AWS_BEARER_TOKEN_BEDROCK, "")], None)]
    #[case::nothing_signs(None, &[], None)]
    fn a_bearer_token_is_the_credential_else_sigv4(
        #[case] api_key: Option<&str>,
        #[case] env: &'static [(&'static str, &'static str)],
        #[case] expected: Option<&str>,
    ) {
        let auth = CONFIG
            .validate_environment(Vec::new(), api_key, "m", &env_from(env))
            .unwrap()
            .auth;
        match (auth, expected) {
            (
                AuthScheme::Credential {
                    placement: CredentialPlacement::Bearer,
                    secret,
                },
                Some(token),
            ) => assert_eq!(secret.expose(), token),
            (AuthScheme::AwsSigV4 { .. }, None) => {}
            (other, _) => panic!("unexpected auth {other:?}"),
        }
    }

    #[test]
    fn settings_choose_where_credentials_and_endpoint_come_from() {
        let config = AmazonMantleMessagesConfig {
            settings: MantleSettings {
                bearer_token_envs: &["OTHER_KEY"],
                api_base_env: "OTHER_BASE",
                region_envs: &["OTHER_REGION"],
                default_region: "us-east-1",
                secret_names: &["OTHER_KEY", "OTHER_BASE", "OTHER_REGION"],
            },
        };
        let env = env_from(&[
            ("OTHER_KEY", "other-token"),
            ("OTHER_REGION", "eu-west-3"),
            (AWS_BEARER_TOKEN_BEDROCK, "bedrock-token"),
            (BEDROCK_MANTLE_API_BASE_ENV, "https://bedrock.example"),
        ]);
        assert!(matches!(
            config.validate_environment(Vec::new(), None, "m", &env).unwrap().auth,
            AuthScheme::Credential { ref secret, .. } if secret.expose() == "other-token"
        ));
        assert_eq!(
            config.get_complete_url(None, "m", &env).unwrap(),
            "https://bedrock-mantle.eu-west-3.api.aws/anthropic/v1/messages"
        );
        assert_eq!(
            config.get_complete_url(None, "m", &no_env).unwrap(),
            "https://bedrock-mantle.us-east-1.api.aws/anthropic/v1/messages"
        );
        assert_eq!(config.secret_names(), config.settings.secret_names);
    }

    fn response(usage: Option<Value>) -> MessagesResponse {
        let base = json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
            "content": [], "stop_reason": "end_turn", "stop_sequence": null
        });
        let Value::Object(fields) = base else {
            unreachable!()
        };
        serde_json::from_value(Value::Object(
            fields
                .into_iter()
                .chain(usage.map(|usage| ("usage".to_string(), usage)))
                .collect(),
        ))
        .unwrap()
    }

    #[rstest]
    #[case::missing_usage(None, json!({"input_tokens": 0, "output_tokens": 0}))]
    #[case::missing_counts(
        Some(json!({"cache_read_input_tokens": 5})),
        json!({"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 5}),
    )]
    #[case::counts_kept(
        Some(json!({"input_tokens": 3, "output_tokens": 7})),
        json!({"input_tokens": 3, "output_tokens": 7}),
    )]
    fn response_usage_always_carries_token_counts(
        #[case] usage: Option<Value>,
        #[case] expected: Value,
    ) {
        let transformed = CONFIG
            .transform_anthropic_messages_response("m", response(usage))
            .unwrap();
        assert_eq!(
            serde_json::to_value(transformed).unwrap()["usage"],
            expected
        );
    }

    #[test]
    fn secret_names_cover_every_lookup() {
        let requested = RefCell::new(Vec::<String>::new());
        let record = |name: &str| -> Option<String> {
            requested.borrow_mut().push(name.to_string());
            None
        };
        let _ = CONFIG.validate_environment(Vec::new(), None, "m", &record);
        let _ = CONFIG.get_complete_url(None, "m", &record);
        let requested = requested.into_inner();
        assert!(!requested.is_empty());
        let undeclared: Vec<&String> = requested
            .iter()
            .filter(|name| !CONFIG.secret_names().contains(&name.as_str()))
            .collect();
        assert_eq!(undeclared, Vec::<&String>::new());
    }

    #[test]
    fn sigv4_drops_caller_copies_of_signer_computed_headers() {
        let computed: Vec<(&str, &str)> = SIGV4_COMPUTED_HEADER_NAMES
            .iter()
            .map(|name| (*name, "caller"))
            .chain([("Authorization", "Bearer caller")])
            .collect();
        let sent = CONFIG
            .validate_environment(
                headers(&[computed.as_slice(), &[("x-request-id", "abc")]].concat()),
                None,
                "m",
                &env_from(&[]),
            )
            .unwrap();
        assert!(matches!(sent.auth, AuthScheme::AwsSigV4 { .. }));
        assert_eq!(sent.headers, headers(&[("x-request-id", "abc")]));
    }

    #[test]
    fn a_bearer_token_leaves_forwarded_headers_alone() {
        let forwarded = headers(&[("authorization", "Bearer caller"), ("x-request-id", "abc")]);
        let sent = CONFIG
            .validate_environment(forwarded.clone(), Some("key"), "m", &env_from(&[]))
            .unwrap();
        assert_eq!(sent.headers, forwarded);
    }
    #[rstest]
    #[case::supported_mid_conversation(true, false, true)]
    #[case::supported_leading_run(true, true, true)]
    #[case::unsupported_mid_conversation(false, false, false)]
    #[case::unsupported_leading_run(false, true, false)]
    fn test_messages_preserve_mid_conversation_system_by_capability(
        #[case] supports_mid_conversation_system: bool,
        #[case] leading_system: bool,
        #[case] expected_system_role: bool,
    ) {
        use litellm_llms_types::formats::messages::{
            ContentBlock, Message, MessageContent, MessageRole, MessagesOptionalParams,
            SystemPrompt,
        };
        let message = |role, text: &str| Message {
            role,
            content: MessageContent::Text(text.into()),
            extra: Default::default(),
        };
        let messages = leading_system
            .then(|| {
                [
                    message(MessageRole::System, "leading"),
                    message(MessageRole::System, "second leading"),
                ]
            })
            .into_iter()
            .flatten()
            .chain([
                message(MessageRole::User, "first"),
                message(MessageRole::Assistant, "answer"),
                message(MessageRole::System, "later"),
                message(MessageRole::User, "last"),
            ])
            .collect();
        let request = MessagesRequest {
            model: "test-model".into(),
            messages,
            params: MessagesOptionalParams {
                max_tokens: Some(4096),
                ..Default::default()
            },
        };
        let context = MessagesTransformContext::with_lookup(
            MessagesModelCapabilities {
                supports_mid_conversation_system,
                ..Default::default()
            },
            false,
            &|_: &str| None,
        );
        let result = AMAZON_MANTLE_MESSAGES_CONFIG
            .transform_anthropic_messages_request(request, &context)
            .unwrap();
        assert_eq!(result.messages.len(), 4);
        assert_eq!(result.messages[0], message(MessageRole::User, "first"));
        assert_eq!(
            result.messages[1],
            message(MessageRole::Assistant, "answer")
        );
        assert_eq!(result.messages[3], message(MessageRole::User, "last"));
        assert_eq!(
            result.messages[2].role,
            if expected_system_role {
                MessageRole::System
            } else {
                MessageRole::User
            }
        );
        assert_eq!(
            result.messages[2].content,
            if expected_system_role {
                MessageContent::Text("later".into())
            } else {
                MessageContent::Blocks(vec![
                    ContentBlock::text(
                        "Operator note (not from the user): the following was originally a mid-conversation system-role reminder.",
                    ),
                    ContentBlock::text("later"),
                ])
            }
        );
        assert_eq!(
            result.params.system,
            leading_system.then(|| SystemPrompt::Blocks(vec![
                ContentBlock::text("leading"),
                ContentBlock::text("second leading")
            ]))
        );
    }
    #[rstest]
    #[case::region_overrides_public_host(
        Some("https://bedrock-mantle.us-east-1.api.aws/v1"),
        Some("us-east-2"),
        "https://bedrock-mantle.us-east-2.api.aws/anthropic/v1/messages",
        "us-east-2"
    )]
    #[case::host_without_override(
        Some("https://bedrock-mantle.eu-west-1.api.aws/openai/v1/"),
        None,
        "https://bedrock-mantle.eu-west-1.api.aws/anthropic/v1/messages",
        "eu-west-1"
    )]
    #[case::custom_host(
        Some("https://vpce.proxy.invalid/v1"),
        Some("us-west-2"),
        "https://vpce.proxy.invalid/anthropic/v1/messages",
        "us-west-2"
    )]
    #[case::default_host(
        None,
        Some("ap-northeast-1"),
        "https://bedrock-mantle.ap-northeast-1.api.aws/anthropic/v1/messages",
        "ap-northeast-1"
    )]
    fn test_mantle_connection_region_controls_url_and_sigv4(
        #[case] api_base: Option<&str>,
        #[case] region: Option<&str>,
        #[case] expected_url: &str,
        #[case] expected_region: &str,
    ) {
        let connection = BedrockMessagesConnection {
            api_base: api_base.map(str::to_string),
            region: region.map(str::to_string),
            ..Default::default()
        };
        assert_eq!(
            CONFIG
                .get_complete_url_with_connection(&connection, &no_env)
                .unwrap(),
            expected_url
        );
        let environment = CONFIG
            .validate_environment_with_connection(
                headers(&[("Authorization", "stale")]),
                None,
                &connection,
                &no_env,
            )
            .unwrap();
        let AuthScheme::AwsSigV4 { region, .. } = environment.auth else {
            panic!("AWS signing expected")
        };
        assert_eq!(region, expected_region);
        assert!(
            environment
                .headers
                .iter()
                .all(|(name, _)| !name.eq_ignore_ascii_case("authorization"))
        );
    }

    #[rstest]
    #[case::new_header(vec![], Some("project_fixture"), vec![("anthropic-workspace-id", "project_fixture")])]
    #[case::explicit_project_wins(vec![("Anthropic-Workspace-Id", "caller")], Some("project_fixture"), vec![("anthropic-workspace-id", "project_fixture")])]
    #[case::absent_project(vec![("Anthropic-Workspace-Id", "caller")], None, vec![("Anthropic-Workspace-Id", "caller")])]
    fn test_mantle_project_id_becomes_the_workspace_header(
        #[case] incoming: Vec<(&str, &str)>,
        #[case] project: Option<&str>,
        #[case] expected: Vec<(&str, &str)>,
    ) {
        let connection = BedrockMessagesConnection {
            workspace_id: project.map(str::to_string),
            ..Default::default()
        };
        let environment = CONFIG
            .validate_environment_with_connection(
                headers(&incoming),
                Some("bearer_fixture"),
                &connection,
                &no_env,
            )
            .unwrap();
        assert_eq!(environment.headers, headers(&expected));
        assert!(matches!(
            environment.auth,
            AuthScheme::Credential {
                placement: CredentialPlacement::Bearer,
                ..
            }
        ));
    }
    #[rstest]
    #[case::bare("us-east-2/anthropic.test-model")]
    #[case::routed("mantle/us-east-2/anthropic.test-model")]
    fn test_mantle_region_prefix_controls_body_endpoint_and_signature(#[case] model: &str) {
        let expected = "https://bedrock-mantle.us-east-2.api.aws/anthropic/v1/messages";
        assert_eq!(
            CONFIG.get_complete_url(None, model, &no_env).unwrap(),
            expected
        );
        let environment = CONFIG
            .validate_environment(vec![], None, model, &no_env)
            .unwrap();
        let AuthScheme::AwsSigV4 { region, .. } = environment.auth else {
            panic!("AWS signing expected")
        };
        assert_eq!(Some(region), mantle_host_region(expected));
        let request: MessagesRequest = serde_json::from_value(json!({"model": model, "max_tokens": 8, "messages": [{"role":"user","content":"ping"}]})).unwrap();
        let request = CONFIG
            .transform_anthropic_messages_request(request, &MessagesTransformContext::default())
            .unwrap();
        let wire = CONFIG.prepare_wire_request(&request, vec![]).unwrap();
        assert_eq!(wire.body["model"], "anthropic.test-model");
        assert_eq!(
            wire.body["messages"],
            json!([{"role":"user","content":"ping"}])
        );
        assert_eq!(wire.body["max_tokens"], 8);
    }

    #[rstest]
    fn test_mantle_tools_and_choice_reach_wire_unchanged() {
        use litellm_llms_types::formats::messages::{
            CustomTool, MessagesTool, ToolChoice, ToolDefinition,
        };
        use litellm_llms_types::json_schema::{JsonSchema, JsonSchemaObject, JsonSchemaType};
        let tools = vec![Recognized::Known(MessagesTool::Custom(CustomTool {
            definition: ToolDefinition {
                name: Some(Recognized::Known("get_weather".into())),
                description: Some(Recognized::Known("Look up the weather".into())),
                input_schema: Some(Recognized::Known(JsonSchema::Object(Box::new(
                    JsonSchemaObject {
                        schema_type: Some(Recognized::Known(JsonSchemaType::Name("object".into()))),
                        properties: Some(Recognized::Known(
                            [(
                                "city".into(),
                                Recognized::Known(JsonSchema::Object(Box::new(JsonSchemaObject {
                                    schema_type: Some(Recognized::Known(JsonSchemaType::Name(
                                        "string".into(),
                                    ))),
                                    ..Default::default()
                                }))),
                            )]
                            .into_iter()
                            .collect(),
                        )),
                        required: Some(Recognized::Known(vec!["city".into()])),
                        ..Default::default()
                    },
                )))),
                ..Default::default()
            },
        }))];
        let choice: Recognized<ToolChoice> =
            serde_json::from_value(json!({"type":"auto"})).unwrap();
        let input = MessagesRequest {
            params: MessagesOptionalParams {
                tools: Some(tools.clone()),
                tool_choice: Some(choice.clone()),
                ..request(json!({})).params
            },
            ..request(json!({}))
        };
        let transformed = CONFIG
            .transform_anthropic_messages_request(input, &MessagesTransformContext::default())
            .unwrap();
        let wire = CONFIG.prepare_wire_request(&transformed, vec![]).unwrap();
        assert_eq!(wire.body["tools"], serde_json::to_value(tools).unwrap());
        assert_eq!(
            wire.body["tool_choice"],
            serde_json::to_value(choice).unwrap()
        );
    }
    #[rstest]
    #[case::accepted(
        "claude-code-20250219,interleaved-thinking-2025-05-14,context-management-2025-06-27",
        false,
        false,
        false,
        Some("claude-code-20250219,context-management-2025-06-27,interleaved-thinking-2025-05-14")
    )]
    #[case::proxy_values(
        "claude-code-20250219,fast-mode-2026-02-01,interleaved-thinking-2025-05-14",
        false,
        false,
        false,
        Some("claude-code-20250219,interleaved-thinking-2025-05-14")
    )]
    #[case::rejected_values(
        "code-execution-2025-08-25,context-1m-2025-08-07,files-api-2025-04-14",
        false,
        false,
        false,
        Some("context-1m-2025-08-07")
    )]
    #[case::all_rejected("code-execution-2025-08-25", false, false, false, None)]
    #[case::alias(
        "advanced-tool-use-2025-11-20",
        false,
        false,
        false,
        Some("tool-search-tool-2025-10-19")
    )]
    #[case::feature_merge(
        "context-1m-2025-08-07",
        true,
        false,
        false,
        Some("context-1m-2025-08-07,context-management-2025-06-27")
    )]
    #[case::safeguards("", false, true, false, Some("dangerous-tool-use-2026-09-03"))]
    #[case::clear_thinking(
        "",
        true,
        false,
        true,
        Some("context-management-2025-06-27,interleaved-thinking-2025-05-14")
    )]
    fn test_mantle_provider_features_and_caller_betas_share_wire_projection(
        #[case] caller_betas: &str,
        #[case] clear_tools: bool,
        #[case] safeguards: bool,
        #[case] clear_thinking: bool,
        #[case] expected_betas: Option<&str>,
    ) {
        use litellm_llms_types::formats::messages::{
            ContextEdit, ContextManagement, Safeguard, ThinkingConfig,
        };
        let edits: Vec<_> = clear_thinking
            .then(|| {
                Recognized::Known(ContextEdit::ClearThinking {
                    extra: [("keep".into(), json!("all"))].into_iter().collect(),
                })
            })
            .into_iter()
            .chain(
                clear_tools
                    .then(|| Recognized::Known(ContextEdit::ClearToolUses { extra: Map::new() })),
            )
            .collect();
        let safeguards = safeguards.then(|| {
            Recognized::Known(vec![Recognized::Known(Safeguard {
                safeguard_type: "dangerous_tool_use".into(),
                classifier_context: Some(Recognized::Known(
                    [
                        ("v".into(), json!(1)),
                        ("permission_mode".into(), json!("auto")),
                    ]
                    .into_iter()
                    .collect(),
                )),
                extra: Map::new(),
            })])
        });
        let input = request(json!({}));
        let input = MessagesRequest {
            params: MessagesOptionalParams {
                context_management: (!edits.is_empty()).then(|| {
                    Recognized::Known(ContextManagement {
                        edits: Some(edits.clone()),
                        extra: Map::new(),
                    })
                }),
                thinking: clear_thinking.then(|| Recognized::Known(ThinkingConfig::adaptive(None))),
                safeguards: safeguards.clone(),
                ..input.params
            },
            ..input
        };
        let context = MessagesTransformContext::with_lookup(
            MessagesModelCapabilities {
                supports_adaptive_thinking: true,
                ..Default::default()
            },
            false,
            &|_: &str| None,
        );
        let transformed = CONFIG
            .transform_anthropic_messages_request(input, &context)
            .unwrap();
        let wire = CONFIG
            .prepare_wire_request(
                &transformed,
                headers(&[
                    ("anthropic-beta", caller_betas),
                    ("anthropic-version", "caller-version"),
                ]),
            )
            .unwrap();
        assert_eq!(
            wire.headers,
            headers(&[("anthropic-version", "caller-version")])
                .into_iter()
                .chain(expected_betas.map(|betas| ("anthropic-beta".into(), betas.into())))
                .collect::<Headers>()
        );
        assert!(wire.body.get("anthropic_beta").is_none());
        assert!(wire.body.get("anthropic_version").is_none());
        assert_eq!(
            wire.body.get("safeguards"),
            safeguards
                .map(|safeguards| serde_json::to_value(safeguards).unwrap())
                .as_ref()
        );
        assert_eq!(
            transformed.params.context_management,
            (!edits.is_empty()).then(|| Recognized::Known(ContextManagement {
                edits: Some(edits),
                extra: Map::new()
            }))
        );
        assert_eq!(
            transformed.params.thinking,
            clear_thinking.then(|| Recognized::Known(ThinkingConfig::adaptive(None)))
        );
    }
}
