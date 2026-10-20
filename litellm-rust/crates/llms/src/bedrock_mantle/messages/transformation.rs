use std::sync::LazyLock;

use litellm_auth::{AwsParams, CredentialPlacement, SecretValue};
use litellm_auth_aws::AwsCredentialSource;
use litellm_auth_aws::bedrock_model_id_and_region;
use litellm_core_utils::settings::resolve_non_empty;
use litellm_http::request::{with_header, without_headers};
use litellm_llms_types::{
    formats::messages::{MessagesOptionalParams, MessagesRequest},
    providers::anthropic::{AnthropicBeta, BetaProvider, BetaSet, DEFAULT_HEADERS},
};
use litellm_router_types::LitellmParams;
use serde_json::Value;

use crate::{
    Error,
    anthropic::{
        common_utils::{existing_betas, merge_beta_headers},
        messages::{
            handler::shape_anthropic_messages_request,
            transformation::{provider_feature_betas, transform_messages_request},
        },
    },
    base_llm::{
        auth::AuthScheme,
        messages::{
            context::MessagesTransformContext,
            transformation::{BaseMessagesConfig, Headers, ValidatedEnvironment},
        },
    },
    bedrock::common_utils::validated_region,
};

const API_BASE_ENV: &str = "BEDROCK_MANTLE_API_BASE";
const REGION_ENVS: &[&str] = &[
    "BEDROCK_MANTLE_REGION",
    "AWS_REGION_NAME",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
];
const KEY_ENVS: &[&str] = &["BEDROCK_MANTLE_API_KEY", "AWS_BEARER_TOKEN_BEDROCK"];
const MESSAGES_PATH: &str = "/anthropic/v1/messages";
const BASE_SUFFIXES: &[&str] = &[
    MESSAGES_PATH,
    "/v1/messages",
    "/messages",
    "/anthropic/v1",
    "/openai/v1",
    "/v1",
];

pub struct BedrockMantleAnthropicMessagesConfig;
pub const BEDROCK_MANTLE_MESSAGES_CONFIG: BedrockMantleAnthropicMessagesConfig =
    BedrockMantleAnthropicMessagesConfig;

impl BaseMessagesConfig for BedrockMantleAnthropicMessagesConfig {
    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        model: &str,
        _optional_params: &MessagesOptionalParams,
        params: &LitellmParams,
        env: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        let headers: Headers = headers
            .into_iter()
            .filter(|(name, _)| {
                !name.eq_ignore_ascii_case("authorization")
                    && !name.eq_ignore_ascii_case("x-api-key")
            })
            .collect();
        let headers = match params
            .extra
            .get("aws_bedrock_project_id")
            .and_then(Value::as_str)
        {
            Some(project) => with_header(headers, "anthropic-workspace-id", project.to_string()),
            None => headers,
        };
        let auth = match resolve_non_empty(api_key, env, KEY_ENVS) {
            Some(token) => AuthScheme::Credential {
                placement: CredentialPlacement::Bearer,
                secret: SecretValue::new(token),
            },
            None => AuthScheme::AwsSigV4 {
                region: mantle_region(params.api_base.as_deref(), model, params, env)?,
                service: "bedrock",
                credentials: Box::new(AwsCredentialSource::from_params(&params.aws, env)),
            },
        };
        Ok(ValidatedEnvironment { headers, auth })
    }

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        params: &LitellmParams,
        _stream: bool,
        env: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        let region = mantle_region(api_base, model, params, env)?;
        let configured = mantle_base(api_base, params, env)
            .unwrap_or_else(|| format!("https://bedrock-mantle.{region}.api.aws"));
        let trimmed = configured.trim_end_matches('/');
        let stripped = BASE_SUFFIXES
            .iter()
            .find_map(|suffix| trimmed.strip_suffix(suffix))
            .unwrap_or(trimmed);
        let base = if mantle_host_region(stripped).is_some() {
            format!("https://bedrock-mantle.{region}.api.aws")
        } else {
            stripped.to_string()
        };
        Ok(format!("{base}{MESSAGES_PATH}"))
    }

    fn transform_anthropic_messages_request(
        &self,
        request: MessagesRequest,
        context: &MessagesTransformContext,
    ) -> Result<MessagesRequest, Error> {
        transform_messages_request(request, context)
    }

    fn shape_request(
        &self,
        request: MessagesRequest,
        reasoning_auto_summary: bool,
    ) -> Result<MessagesRequest, Error> {
        shape_anthropic_messages_request(request, reasoning_auto_summary)
    }

    fn wire_body(&self, body: Value) -> Value {
        match body {
            Value::Object(fields) => Value::Object(
                fields
                    .into_iter()
                    .filter_map(|(name, value)| match name.as_str() {
                        "anthropic_version" | "anthropic_beta" => None,
                        "model" => Some((
                            name,
                            value
                                .as_str()
                                .map(|model| Value::String(mantle_model(model).0))
                                .unwrap_or(value),
                        )),
                        _ => Some((name, value)),
                    })
                    .collect(),
            ),
            other => other,
        }
    }

    fn secret_names(&self) -> &'static [&'static str] {
        static NAMES: LazyLock<Box<[&'static str]>> = LazyLock::new(|| {
            [
                "BEDROCK_MANTLE_API_KEY",
                "AWS_BEARER_TOKEN_BEDROCK",
                API_BASE_ENV,
            ]
            .into_iter()
            .chain(REGION_ENVS.iter().copied())
            .chain(
                AwsParams::secret_names()
                    .iter()
                    .copied()
                    .filter(|name| !REGION_ENVS.contains(name)),
            )
            .collect()
        });
        &NAMES
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        DEFAULT_HEADERS
    }

    fn request_headers(&self, headers: Headers, request: &MessagesRequest) -> Headers {
        let body_betas: BetaSet = request
            .params
            .extra
            .get("anthropic_beta")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .filter_map(|beta| beta.parse::<AnthropicBeta>().ok())
            .collect();
        let caller: BetaSet = existing_betas(&headers)
            .union(body_betas)
            .into_iter()
            .filter_map(|beta| match beta {
                AnthropicBeta::Other(_) => Some(beta),
                _ => beta.on(BetaProvider::BedrockMantle),
            })
            .collect();
        let safeguards: BetaSet = request
            .params
            .extra
            .contains_key("safeguards")
            .then_some(AnthropicBeta::DangerousToolUse20260903)
            .into_iter()
            .collect();
        merge_beta_headers(
            without_headers(headers, &["anthropic-beta"]),
            caller
                .union(provider_feature_betas(request, BetaProvider::BedrockMantle))
                .union(safeguards),
        )
    }
}

fn mantle_model(model: &str) -> (String, Option<String>) {
    bedrock_model_id_and_region(model.strip_prefix("mantle/").unwrap_or(model))
}

fn mantle_host_region(base: &str) -> Option<String> {
    let url = url::Url::parse(base).ok()?;
    if !matches!(url.scheme(), "https" | "http") {
        return None;
    }
    url.host_str()?
        .strip_prefix("bedrock-mantle.")?
        .strip_suffix(".api.aws")
        .filter(|region| !region.contains('.') && !region.is_empty())
        .map(str::to_string)
}

fn mantle_base(
    base: Option<&str>,
    params: &LitellmParams,
    env: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    let explicit = [
        base,
        params.api_base.as_deref(),
        params.aws.aws_bedrock_runtime_endpoint.as_deref(),
    ]
    .into_iter()
    .flatten()
    .find(|base| !base.trim().is_empty());
    resolve_non_empty(explicit, env, &[API_BASE_ENV])
}

fn mantle_region(
    base: Option<&str>,
    model: &str,
    params: &LitellmParams,
    env: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    let region = mantle_base(base, params, env)
        .and_then(|base| mantle_host_region(&base))
        .or_else(|| resolve_non_empty(params.aws.aws_region_name.as_deref(), &|_| None, &[]))
        .or_else(|| mantle_model(model).1)
        .or_else(|| resolve_non_empty(None, env, REGION_ENVS))
        .unwrap_or_else(|| "us-east-1".into());
    validated_region(region)
}

#[cfg(test)]
mod tests {
    use super::*;
    use litellm_http::request::header_value;
    use rstest::rstest;
    use serde_json::json;

    #[rstest]
    #[case::root("https://private.test", "https://private.test/anthropic/v1/messages")]
    #[case::openai(
        "https://private.test/openai/v1/",
        "https://private.test/anthropic/v1/messages"
    )]
    #[case::complete(
        "https://private.test/anthropic/v1/messages/",
        "https://private.test/anthropic/v1/messages"
    )]
    #[case::tenant(
        "https://private.test/tenant/v1",
        "https://private.test/tenant/anthropic/v1/messages"
    )]
    fn endpoints_are_normalized(#[case] base: &str, #[case] expected: &str) {
        assert_eq!(
            BEDROCK_MANTLE_MESSAGES_CONFIG
                .get_complete_url(
                    Some(base),
                    "claude-test",
                    &LitellmParams::default(),
                    true,
                    &|_| None
                )
                .unwrap(),
            expected
        );
    }

    #[rstest]
    #[case::parameter("us-west-2", "claude-test")]
    #[case::model("us-west-2", "us-west-2/claude-test")]
    fn private_endpoints_keep_their_host_and_signing_region(
        #[case] region: &str,
        #[case] model: &str,
    ) {
        let params = LitellmParams {
            aws: AwsParams {
                aws_region_name: (model == "claude-test").then(|| region.into()),
                aws_bedrock_runtime_endpoint: Some("https://private.test/anthropic/v1".into()),
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(
            BEDROCK_MANTLE_MESSAGES_CONFIG
                .get_complete_url(None, model, &params, false, &|_| None)
                .unwrap(),
            "https://private.test/anthropic/v1/messages"
        );
        let environment = BEDROCK_MANTLE_MESSAGES_CONFIG
            .validate_environment(
                vec![],
                None,
                model,
                &MessagesOptionalParams::default(),
                &params,
                &|_| None,
            )
            .unwrap();
        assert!(
            matches!(environment.auth, AuthScheme::AwsSigV4 { region: actual, service: "bedrock", .. } if actual == region)
        );
    }

    #[rstest]
    #[case::explicit_regional_endpoint(
        "https://bedrock-mantle.us-west-2.api.aws",
        "us-east-1",
        "us-west-2"
    )]
    fn public_regional_host_precedes_configured_region(
        #[case] api_base: &str,
        #[case] configured_region: &str,
        #[case] expected_region: &str,
    ) {
        let params = LitellmParams {
            api_base: Some(api_base.into()),
            aws: AwsParams {
                aws_region_name: Some(configured_region.into()),
                ..Default::default()
            },
            ..Default::default()
        };
        let url = BEDROCK_MANTLE_MESSAGES_CONFIG
            .get_complete_url(None, "claude-test", &params, false, &|_| None)
            .unwrap();
        let environment = BEDROCK_MANTLE_MESSAGES_CONFIG
            .validate_environment(
                vec![],
                None,
                "claude-test",
                &MessagesOptionalParams::default(),
                &params,
                &|_| None,
            )
            .unwrap();
        assert_eq!(
            url,
            "https://bedrock-mantle.us-west-2.api.aws/anthropic/v1/messages"
        );
        assert!(
            matches!(environment.auth, AuthScheme::AwsSigV4 { region, service: "bedrock", .. } if region == expected_region)
        );
    }

    #[rstest]
    #[case::aws_default_region("AWS_DEFAULT_REGION", "eu-west-1")]
    fn aws_default_region_selects_mantle_endpoint(#[case] env_name: &str, #[case] region: &str) {
        let url = BEDROCK_MANTLE_MESSAGES_CONFIG
            .get_complete_url(
                None,
                "claude-test",
                &LitellmParams::default(),
                false,
                &|name| (name == env_name).then(|| region.into()),
            )
            .unwrap();
        assert_eq!(
            url,
            "https://bedrock-mantle.eu-west-1.api.aws/anthropic/v1/messages"
        );
    }

    #[rstest]
    #[case::aws_default_region("AWS_DEFAULT_REGION")]
    fn mantle_secret_names_include_region_fallbacks(#[case] env_name: &str) {
        assert!(
            BEDROCK_MANTLE_MESSAGES_CONFIG
                .secret_names()
                .contains(&env_name)
        );
    }

    #[rstest]
    #[case::explicit(Some("explicit"), "explicit")]
    #[case::environment(None, "mantle")]
    fn bearer_credentials_replace_forwarded_auth(
        #[case] key: Option<&str>,
        #[case] expected: &str,
    ) {
        let environment = BEDROCK_MANTLE_MESSAGES_CONFIG
            .validate_environment(
                vec![
                    ("X-Api-Key".into(), "stale".into()),
                    ("Authorization".into(), "stale".into()),
                ],
                key,
                "claude-test",
                &MessagesOptionalParams::default(),
                &LitellmParams::default(),
                &|name| match name {
                    "BEDROCK_MANTLE_API_KEY" => Some("mantle".into()),
                    "AWS_BEARER_TOKEN_BEDROCK" => Some("aws".into()),
                    _ => None,
                },
            )
            .unwrap();
        assert!(environment.headers.is_empty());
        assert!(
            matches!(environment.auth, AuthScheme::Credential { placement: CredentialPlacement::Bearer, secret } if secret.expose() == expected)
        );
    }

    #[rstest]
    #[case::url_injection("us-east-1.attacker.test/steal?")]
    #[case::uppercase("US-EAST-1")]
    fn invalid_regions_are_rejected_even_with_bearer_auth(#[case] region: &str) {
        let params = LitellmParams {
            aws: AwsParams {
                aws_region_name: Some(region.into()),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(matches!(
            BEDROCK_MANTLE_MESSAGES_CONFIG.get_complete_url(
                None,
                "claude-test",
                &params,
                false,
                &|_| None
            ),
            Err(Error::InvalidRequest(_))
        ));
    }

    #[rstest]
    #[case::arn("arn:aws:bedrock:us-east-1.attacker.test:123:model/claude-test", None)]
    #[case::environment("claude-test", Some("us-east-1/steal"))]
    fn model_arn_and_environment_regions_are_validated(
        #[case] model: &str,
        #[case] region: Option<&str>,
    ) {
        let url = BEDROCK_MANTLE_MESSAGES_CONFIG.get_complete_url(
            None,
            model,
            &LitellmParams::default(),
            false,
            &|name| {
                (name == "BEDROCK_MANTLE_REGION")
                    .then(|| region.map(str::to_string))
                    .flatten()
            },
        );
        assert!(matches!(url, Err(Error::InvalidRequest(_))));
    }

    #[rstest]
    fn caller_betas_are_filtered_renamed_and_merged_with_feature_betas() {
        let request: MessagesRequest = serde_json::from_value(json!({"model": "claude-test", "max_tokens": 16, "messages": [], "safeguards": [{"type": "dangerous_tool_use"}]})).unwrap();
        let headers = BEDROCK_MANTLE_MESSAGES_CONFIG.request_headers(
            vec![(
                "Anthropic-Beta".into(),
                "advanced-tool-use-2025-11-20,code-execution-2025-08-25,custom-beta".into(),
            )],
            &request,
        );
        let actual: std::collections::BTreeSet<&str> = header_value(&headers, "anthropic-beta")
            .unwrap()
            .split(',')
            .collect();
        let tool_search = AnthropicBeta::ToolSearchTool20251019.to_string();
        let safeguards = AnthropicBeta::DangerousToolUse20260903.to_string();
        assert_eq!(
            actual,
            ["custom-beta", tool_search.as_str(), safeguards.as_str()]
                .into_iter()
                .collect()
        );
    }

    #[rstest]
    #[case::body_only(
        json!(["custom-beta", "advanced-tool-use-2025-11-20"]),
        vec![],
        vec![
            AnthropicBeta::Other("custom-beta".into()),
            AnthropicBeta::ToolSearchTool20251019,
        ]
    )]
    #[case::header_and_body(
        json!(["custom-b"]),
        vec![("Anthropic-Beta".into(), "custom-a".into())],
        vec![
            AnthropicBeta::Other("custom-a".into()),
            AnthropicBeta::Other("custom-b".into()),
        ]
    )]
    fn body_betas_are_filtered_renamed_and_merged_with_headers(
        #[case] body_betas: Value,
        #[case] headers: Headers,
        #[case] expected: Vec<AnthropicBeta>,
    ) {
        let request: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-test",
            "max_tokens": 16,
            "messages": [],
            "anthropic_beta": body_betas
        }))
        .unwrap();
        let headers = BEDROCK_MANTLE_MESSAGES_CONFIG.request_headers(headers, &request);
        let actual: BetaSet = header_value(&headers, "anthropic-beta")
            .unwrap()
            .parse()
            .unwrap();
        let expected: BetaSet = expected.into_iter().collect();
        assert_eq!(actual, expected);
    }

    #[rstest]
    fn wire_body_keeps_streaming_and_removes_sdk_fields() {
        assert_eq!(BEDROCK_MANTLE_MESSAGES_CONFIG.wire_body(json!({"model": "mantle/us-west-2/anthropic.claude-test", "stream": true, "anthropic_version": "sdk", "anthropic_beta": ["sdk"], "messages": []})), json!({"model": "anthropic.claude-test", "stream": true, "messages": []}));
    }
}
