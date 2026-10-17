use std::sync::LazyLock;

use litellm_auth::{AwsParams, CredentialPlacement, ParamSpec, SecretValue};
use litellm_auth_aws::{AwsCredentialSource, resolve_aws_region};
use litellm_core_utils::settings::resolve_non_empty;
use litellm_http::request::{header_value, with_header, without_headers};
use litellm_llms_types::{
    formats::messages::MessagesRequest,
    providers::anthropic::{DEFAULT_HEADERS, MESSAGES_PATH},
};
use litellm_router_types::LitellmParams;
use serde_json::Value;

use crate::{
    Error, ErrorDetail,
    anthropic::messages::{
        handler::shape_anthropic_messages_request,
        transformation::{transform_messages_request, update_headers_with_anthropic_beta},
    },
    base_llm::{
        auth::{AuthScheme, Headers, ValidatedEnvironment},
        messages::{context::MessagesTransformContext, transformation::BaseMessagesConfig},
    },
    bedrock::common_utils::validated_region,
};

const SERVICE: &str = "aws-external-anthropic";
const BASE_ENVS: &[&str] = &["ANTHROPIC_AWS_BASE_URL", "ANTHROPIC_AWS_API_BASE"];
const WORKSPACE_ENVS: &[&str] = &["ANTHROPIC_AWS_WORKSPACE_ID", "ANTHROPIC_WORKSPACE_ID"];
const WORKSPACE_FIELDS: &[&str] = &[
    "workspace_id",
    "aws_workspace_id",
    "anthropic-workspace-id",
    "anthropic_workspace_id",
];
const WORKSPACE_HEADER: &str = "anthropic-workspace-id";
static WORKSPACE: ParamSpec = ParamSpec {
    setting: "workspace_id",
    wire: WORKSPACE_FIELDS,
    module_global: None,
    env: WORKSPACE_ENVS,
};
const UNSUPPORTED_OVERRIDE: &str = "claude_platform_unsupported_params";

pub struct BedrockClaudePlatformMessagesConfig;
pub const BEDROCK_CLAUDE_PLATFORM_MESSAGES_CONFIG: BedrockClaudePlatformMessagesConfig =
    BedrockClaudePlatformMessagesConfig;

impl BaseMessagesConfig for BedrockClaudePlatformMessagesConfig {
    fn shape_request(
        &self,
        request: MessagesRequest,
        reasoning_auto_summary: bool,
    ) -> Result<MessagesRequest, Error> {
        shape_anthropic_messages_request(request, reasoning_auto_summary)
    }

    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        _model: &str,
        params: &LitellmParams,
        env: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        let workspace = header_value(&headers, WORKSPACE_HEADER)
            .filter(|value| !value.trim().is_empty())
            .or_else(|| {
                WORKSPACE_FIELDS.iter().find_map(|field| {
                    params
                        .extra
                        .get(*field)?
                        .as_str()
                        .filter(|value| !value.trim().is_empty())
                })
            });
        let workspace = resolve_non_empty(workspace, env, WORKSPACE_ENVS)
            .ok_or_else(|| Error::Auth(WORKSPACE.missing("Claude Platform on AWS")))?;
        let headers = with_header(headers, WORKSPACE_HEADER, workspace);
        let key = resolve_non_empty(api_key, env, &["ANTHROPIC_AWS_API_KEY"]);
        let headers = without_headers(headers, &["authorization"]);
        if header_value(&headers, "x-api-key").is_some_and(|value| !value.trim().is_empty()) {
            return Ok(ValidatedEnvironment {
                headers,
                auth: AuthScheme::Forwarded,
            });
        }
        if let Some(key) = key {
            return Ok(ValidatedEnvironment {
                headers: without_headers(headers, &["x-api-key"]),
                auth: AuthScheme::Credential {
                    placement: CredentialPlacement::Header("x-api-key"),
                    secret: SecretValue::new(key),
                },
            });
        }
        let base = resolve_non_empty(params.api_base.as_deref(), env, BASE_ENVS);
        Ok(ValidatedEnvironment {
            headers: without_headers(headers, &["x-api-key"]),
            auth: AuthScheme::AwsSigV4 {
                region: platform_region(base.as_deref(), params, env)?,
                service: SERVICE,
                credentials: Box::new(AwsCredentialSource::from_params(&params.aws, env)),
            },
        })
    }

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        params: &LitellmParams,
        _stream: bool,
        env: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        let base = resolve_non_empty(
            api_base
                .filter(|base| !base.trim().is_empty())
                .or(params.api_base.as_deref()),
            env,
            BASE_ENVS,
        );
        let base = match base {
            Some(base) => base,
            None => format!(
                "https://{SERVICE}.{}.api.aws",
                platform_region(None, params, env)?
            ),
        };
        let base = base.trim_end_matches('/');
        Ok(if base.ends_with(MESSAGES_PATH) {
            base.into()
        } else {
            format!("{base}{MESSAGES_PATH}")
        })
    }

    fn transform_anthropic_messages_request(
        &self,
        request: MessagesRequest,
        context: &MessagesTransformContext,
    ) -> Result<MessagesRequest, Error> {
        let body = serde_json::to_value(request).map_err(invalid_request)?;
        let filtered = serde_json::from_value(filter_body(body, false)).map_err(invalid_request)?;
        transform_messages_request(filtered, context)
    }

    fn secret_names(&self) -> &'static [&'static str] {
        static NAMES: LazyLock<Box<[&'static str]>> = LazyLock::new(|| {
            ["ANTHROPIC_AWS_API_KEY"]
                .into_iter()
                .chain(BASE_ENVS.iter().copied())
                .chain(WORKSPACE_ENVS.iter().copied())
                .chain(["AWS_DEFAULT_REGION"])
                .chain(AwsParams::secret_names().iter().copied())
                .collect()
        });
        &NAMES
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        DEFAULT_HEADERS
    }

    fn request_headers(&self, headers: Headers, request: &MessagesRequest) -> Headers {
        let headers = match WORKSPACE_FIELDS.iter().find_map(|field| {
            request
                .params
                .extra
                .get(*field)?
                .as_str()
                .filter(|value| !value.trim().is_empty())
        }) {
            Some(workspace) => with_header(headers, WORKSPACE_HEADER, workspace.into()),
            None => headers,
        };
        update_headers_with_anthropic_beta(headers, request)
    }

    fn wire_body(&self, body: Value) -> Value {
        filter_body(body, true)
    }
}

fn platform_region(
    base: Option<&str>,
    params: &LitellmParams,
    env: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    let host_region = base.and_then(|base| {
        let url = url::Url::parse(base).ok()?;
        url.host_str()?
            .strip_prefix("aws-external-anthropic.")?
            .strip_suffix(".api.aws")
            .map(str::to_string)
    });
    let region = host_region
        .or_else(|| resolve_aws_region(None, &params.aws, env))
        .or_else(|| resolve_non_empty(None, env, &["AWS_DEFAULT_REGION"]))
        .ok_or_else(|| Error::Auth(AwsParams::REGION.missing("Claude Platform on AWS")))?;
    validated_region(region)
}

fn filter_body(body: Value, strip_connection_fields: bool) -> Value {
    let Value::Object(fields) = body else {
        return body;
    };
    let unsupported: Vec<&str> = fields
        .get(UNSUPPORTED_OVERRIDE)
        .and_then(Value::as_array)
        .map(|values| values.iter().filter_map(Value::as_str).collect())
        .unwrap_or_else(|| vec!["context_management"]);
    let retained = fields
        .iter()
        .filter(|(name, _)| {
            !unsupported.contains(&name.as_str())
                && !(strip_connection_fields
                    && (WORKSPACE_FIELDS.contains(&name.as_str())
                        || name.starts_with("aws_")
                        || name.as_str() == UNSUPPORTED_OVERRIDE))
        })
        .map(|(name, value)| {
            let value = if name == "model" {
                value
                    .as_str()
                    .map(|model| {
                        Value::String(
                            model
                                .strip_prefix("claude_platform/")
                                .unwrap_or(model)
                                .into(),
                        )
                    })
                    .unwrap_or_else(|| value.clone())
            } else {
                value.clone()
            };
            (name.clone(), value)
        })
        .collect();
    Value::Object(retained)
}

fn invalid_request(error: serde_json::Error) -> Error {
    Error::InvalidRequest(ErrorDetail::invalid("Messages request", error))
}

#[cfg(test)]
mod tests {
    use super::*;
    use litellm_llms_types::providers::anthropic::AnthropicBeta;
    use rstest::rstest;
    use serde_json::json;

    #[rstest]
    fn workspace_is_required_before_any_upstream_request() {
        assert_eq!(
            BEDROCK_CLAUDE_PLATFORM_MESSAGES_CONFIG
                .validate_environment(
                    vec![],
                    Some("key"),
                    "claude-test",
                    &LitellmParams::default(),
                    &|_| None
                )
                .unwrap_err(),
            Error::Auth(WORKSPACE.missing("Claude Platform on AWS"))
        );
    }

    #[rstest]
    #[case::valid("forwarded", true)]
    #[case::blank(" ", false)]
    fn forwarded_keys_win_and_blank_keys_fall_back(#[case] key: &str, #[case] forwarded: bool) {
        let environment = BEDROCK_CLAUDE_PLATFORM_MESSAGES_CONFIG
            .validate_environment(
                vec![
                    ("X-Api-Key".into(), key.into()),
                    ("authorization".into(), "stale".into()),
                ],
                Some("explicit"),
                "claude-test",
                &LitellmParams::default(),
                &|name| (name == "ANTHROPIC_AWS_WORKSPACE_ID").then(|| "workspace".into()),
            )
            .unwrap();
        assert_eq!(
            header_value(&environment.headers, WORKSPACE_HEADER),
            Some("workspace")
        );
        assert_eq!(header_value(&environment.headers, "authorization"), None);
        if forwarded {
            assert_eq!(header_value(&environment.headers, "x-api-key"), Some(key));
            assert!(matches!(environment.auth, AuthScheme::Forwarded));
        } else {
            assert_eq!(header_value(&environment.headers, "x-api-key"), None);
            assert!(
                matches!(environment.auth, AuthScheme::Credential { secret, placement: CredentialPlacement::Header("x-api-key") } if secret.expose() == "explicit")
            );
        }
    }

    #[rstest]
    #[case::region("us-west-2", true)]
    #[case::injection("us-east-1.attacker.test/steal?", false)]
    fn generated_endpoint_regions_are_validated(#[case] region: &str, #[case] valid: bool) {
        let result = BEDROCK_CLAUDE_PLATFORM_MESSAGES_CONFIG.get_complete_url(
            None,
            "claude-test",
            &LitellmParams::default(),
            false,
            &|name| (name == "AWS_DEFAULT_REGION").then(|| region.into()),
        );
        if valid {
            assert_eq!(
                result.unwrap(),
                format!("https://{SERVICE}.{region}.api.aws/v1/messages")
            );
        } else {
            assert!(matches!(result, Err(Error::InvalidRequest(_))));
        }
    }

    #[rstest]
    #[case::explicit_regional_endpoint(
        "https://aws-external-anthropic.us-west-2.api.aws",
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
        let env = |name: &str| (name == "ANTHROPIC_AWS_WORKSPACE_ID").then(|| "workspace".into());
        let url = BEDROCK_CLAUDE_PLATFORM_MESSAGES_CONFIG
            .get_complete_url(None, "claude-test", &params, false, &env)
            .unwrap();
        let environment = BEDROCK_CLAUDE_PLATFORM_MESSAGES_CONFIG
            .validate_environment(vec![], None, "claude-test", &params, &env)
            .unwrap();
        assert_eq!(
            url,
            "https://aws-external-anthropic.us-west-2.api.aws/v1/messages"
        );
        assert!(
            matches!(environment.auth, AuthScheme::AwsSigV4 { region, service: SERVICE, .. } if region == expected_region)
        );
    }

    #[rstest]
    #[case::default(json!({}), false)]
    #[case::override_empty(json!({"claude_platform_unsupported_params": []}), true)]
    fn unsupported_fields_are_filtered_before_beta_selection(
        #[case] extra: Value,
        #[case] keep_context: bool,
    ) {
        let fields = json!({"model": "claude_platform/claude-test", "max_tokens": 16, "messages": [], "workspace_id": "workspace", "aws_region_name": "us-west-2", "context_management": {"edits": [{"type": "clear_thinking_20251015"}]}});
        let input = Value::Object(
            fields
                .as_object()
                .unwrap()
                .clone()
                .into_iter()
                .chain(extra.as_object().unwrap().clone())
                .collect(),
        );
        let request = serde_json::from_value(input).unwrap();
        let transformed = BEDROCK_CLAUDE_PLATFORM_MESSAGES_CONFIG
            .transform_anthropic_messages_request(request, &MessagesTransformContext::default())
            .unwrap();
        let headers = BEDROCK_CLAUDE_PLATFORM_MESSAGES_CONFIG.request_headers(vec![], &transformed);
        let expected_beta = AnthropicBeta::ContextManagement20250627.to_string();
        assert_eq!(
            header_value(&headers, "anthropic-beta")
                .is_some_and(|value| value.split(',').any(|beta| beta == expected_beta)),
            keep_context
        );
        let wire = BEDROCK_CLAUDE_PLATFORM_MESSAGES_CONFIG
            .wire_body(serde_json::to_value(transformed).unwrap());
        assert_eq!(wire["model"], "claude-test");
        assert_eq!(wire.get("context_management").is_some(), keep_context);
        assert_eq!(wire.get("workspace_id"), None);
        assert_eq!(wire.get("aws_region_name"), None);
        assert_eq!(wire.get(UNSUPPORTED_OVERRIDE), None);
    }
}
