use litellm_auth::{CredentialPlacement, SecretValue};
use litellm_auth_aws::AwsCredentialSource;
use litellm_auth_aws::bedrock_model_id_and_region;
use litellm_core_utils::settings::resolve_non_empty;
use litellm_http::request::with_header;
use litellm_llms_types::{formats::messages::MessagesRequest, providers::anthropic::BetaProvider};
use litellm_router_types::LitellmParams;
use serde_json::Value;

use crate::{
    Error,
    anthropic::{
        common_utils::merge_beta_headers,
        messages::{
            handler::shape_anthropic_messages_request,
            transformation::{DEFAULT_HEADERS, provider_feature_betas, transform_messages_request},
        },
    },
    base_llm::{
        auth::AuthScheme,
        messages::{
            context::MessagesTransformContext,
            transformation::{BaseMessagesConfig, Headers, ValidatedEnvironment},
        },
    },
};

const API_BASE_ENV: &str = "BEDROCK_MANTLE_API_BASE";
const REGION_ENVS: &[&str] = &["BEDROCK_MANTLE_REGION", "AWS_REGION_NAME", "AWS_REGION"];
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
                region: mantle_region(params.api_base.as_deref(), model, params, env),
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
        let region = mantle_region(api_base, model, params, env);
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
        &[
            "BEDROCK_MANTLE_API_KEY",
            "AWS_BEARER_TOKEN_BEDROCK",
            API_BASE_ENV,
            "BEDROCK_MANTLE_REGION",
            "AWS_REGION_NAME",
            "AWS_REGION",
        ]
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        DEFAULT_HEADERS
    }

    fn request_headers(&self, headers: Headers, request: &MessagesRequest) -> Headers {
        let existing: Vec<String> = headers
            .iter()
            .filter(|(name, _)| name.eq_ignore_ascii_case("anthropic-beta"))
            .map(|(_, value)| value.clone())
            .collect();
        let derived = merge_beta_headers(
            vec![],
            provider_feature_betas(request, BetaProvider::BedrockMantle),
        );
        let added = derived
            .iter()
            .find(|(name, _)| name == "anthropic-beta")
            .map(|(_, value)| value.clone());
        let betas = existing
            .into_iter()
            .chain(added)
            .collect::<Vec<_>>()
            .join(",");
        if betas.is_empty() {
            return headers;
        }
        with_header(headers, "anthropic-beta", betas)
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
) -> String {
    resolve_non_empty(params.aws.aws_region_name.as_deref(), &|_| None, &[])
        .or_else(|| mantle_model(model).1)
        .or_else(|| mantle_base(base, params, env).and_then(|base| mantle_host_region(&base)))
        .or_else(|| resolve_non_empty(None, env, REGION_ENVS))
        .unwrap_or_else(|| "us-east-1".into())
}
