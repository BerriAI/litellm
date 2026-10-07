use std::collections::{BTreeMap, BTreeSet};

use litellm_auth_aws::constants::{AWS_DEFAULT_REGION, AWS_REGION, AWS_REGION_NAME};
use serde_json::{Map, Value};

use crate::{Error, base_llm::messages::transformation::complete_messages_url};

pub const CLAUDE_PLATFORM_SERVICE_NAME: &str = "aws-external-anthropic";
pub const CLAUDE_PLATFORM_ROUTE_PREFIX: &str = "claude_platform/";
pub const WORKSPACE_HEADER: &str = "anthropic-workspace-id";

pub const ANTHROPIC_AWS_API_KEY_ENV: &str = "ANTHROPIC_AWS_API_KEY";
pub const ANTHROPIC_AWS_WORKSPACE_ID_ENV: &str = "ANTHROPIC_AWS_WORKSPACE_ID";
pub const ANTHROPIC_WORKSPACE_ID_ENV: &str = "ANTHROPIC_WORKSPACE_ID";
pub const ANTHROPIC_AWS_BASE_URL_ENV: &str = "ANTHROPIC_AWS_BASE_URL";
pub const ANTHROPIC_AWS_API_BASE_ENV: &str = "ANTHROPIC_AWS_API_BASE";

pub const UNSUPPORTED_PARAMS_OVERRIDE_KEY: &str = "claude_platform_unsupported_params";
const AWS_PARAM_PREFIX: &str = "aws_";
const AWS_REGION_NAME_PARAM: &str = "aws_region_name";

const WORKSPACE_KEYS: [&str; 4] = [
    "workspace_id",
    "aws_workspace_id",
    "anthropic-workspace-id",
    "anthropic_workspace_id",
];
const DEFAULT_UNSUPPORTED_PARAMS: &[&str] = &["context_management"];

/// The params Python reads from `optional_params` and `litellm_params` that the Messages trait
/// has no channel for, resolved once per call by whoever builds the config.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ClaudePlatformSettings {
    pub workspace_id: Option<String>,
    pub aws_params: BTreeMap<String, String>,
    pub unsupported_params: UnsupportedParams,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum UnsupportedParams {
    #[default]
    Default,
    Replaced(BTreeSet<String>),
    /// A non-list `claude_platform_unsupported_params`. It behaves as `Default`, and the
    /// caller logs the JSON type it found.
    Ignored {
        found: &'static str,
    },
}

impl UnsupportedParams {
    pub fn rejects(&self, key: &str) -> bool {
        match self {
            Self::Replaced(keys) => keys.contains(key),
            Self::Default | Self::Ignored { .. } => DEFAULT_UNSUPPORTED_PARAMS.contains(&key),
        }
    }

    fn from_value(value: &Value) -> Self {
        match value {
            Value::Array(items) => Self::Replaced(items.iter().map(param_name).collect()),
            other => Self::Ignored {
                found: json_type(other),
            },
        }
    }
}

impl ClaudePlatformSettings {
    pub const EMPTY: Self = Self {
        workspace_id: None,
        aws_params: BTreeMap::new(),
        unsupported_params: UnsupportedParams::Default,
    };

    /// Python's precedence: for each workspace key the request wins over the deployment, and an
    /// override list in the request replaces the deployment's.
    pub fn from_params(request: &Map<String, Value>, deployment: &Map<String, Value>) -> Self {
        let workspace_id = WORKSPACE_KEYS
            .iter()
            .flat_map(|key| [request.get(*key), deployment.get(*key)])
            .flatten()
            .find_map(workspace_text);
        let unsupported_params = [request, deployment]
            .into_iter()
            .find_map(|params| {
                params
                    .get(UNSUPPORTED_PARAMS_OVERRIDE_KEY)
                    .filter(|value| !value.is_null())
            })
            .map_or(UnsupportedParams::Default, UnsupportedParams::from_value);
        Self {
            workspace_id,
            aws_params: aws_string_params(deployment)
                .chain(aws_string_params(request))
                .collect(),
            unsupported_params,
        }
    }

    pub fn aws_param_map(&self) -> Map<String, Value> {
        self.aws_params
            .iter()
            .map(|(key, value)| (key.clone(), Value::String(value.clone())))
            .collect()
    }
}

pub fn is_non_request_param(key: &str) -> bool {
    key == UNSUPPORTED_PARAMS_OVERRIDE_KEY
        || WORKSPACE_KEYS.contains(&key)
        || key.starts_with(AWS_PARAM_PREFIX)
}

pub fn strip_claude_platform_route(model: &str) -> &str {
    model
        .strip_prefix(CLAUDE_PLATFORM_ROUTE_PREFIX)
        .unwrap_or(model)
}

pub fn resolve_workspace_id(
    settings: &ClaudePlatformSettings,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    settings
        .workspace_id
        .clone()
        .or_else(|| non_blank_env(env_lookup, ANTHROPIC_AWS_WORKSPACE_ID_ENV))
        .or_else(|| non_blank_env(env_lookup, ANTHROPIC_WORKSPACE_ID_ENV))
        .ok_or_else(|| {
            Error::Auth(litellm_auth::Error::ProviderAuthentication(
                "Missing workspace ID for Claude Platform on AWS. Pass `workspace_id` or \
                 configure the provider workspace setting."
                    .into(),
            ))
        })
}

pub fn resolve_api_key(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    api_key
        .filter(|key| !key.trim().is_empty())
        .map(str::to_string)
        .or_else(|| non_blank_env(env_lookup, ANTHROPIC_AWS_API_KEY_ENV))
}

pub fn resolve_region(
    settings: &ClaudePlatformSettings,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    let region = settings
        .aws_params
        .get(AWS_REGION_NAME_PARAM)
        .filter(|region| !region.is_empty())
        .cloned()
        .or_else(|| non_blank_env(env_lookup, AWS_REGION_NAME))
        .or_else(|| non_blank_env(env_lookup, AWS_REGION))
        .or_else(|| non_blank_env(env_lookup, AWS_DEFAULT_REGION))
        .ok_or_else(|| {
            Error::Auth(litellm_auth::Error::ProviderAuthentication(
                "Missing AWS region for Claude Platform on AWS. Pass `aws_region_name` or set \
                 a standard AWS region environment value."
                    .into(),
            ))
        })?;
    if !is_valid_region(&region) {
        return Err(Error::Auth(litellm_auth::Error::InvalidConfiguration(
            format!(
                "Invalid AWS region format: '{region}'. Region names must contain only \
                 lowercase letters, digits, and hyphens."
            )
            .into(),
        )));
    }
    Ok(region)
}

pub fn complete_claude_platform_url(
    api_base: Option<&str>,
    settings: &ClaudePlatformSettings,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    let configured = api_base
        .map(str::trim)
        .filter(|base| !base.is_empty())
        .map(str::to_string)
        .or_else(|| non_blank_env(env_lookup, ANTHROPIC_AWS_BASE_URL_ENV))
        .or_else(|| non_blank_env(env_lookup, ANTHROPIC_AWS_API_BASE_ENV));
    let api_base = match configured {
        Some(base) => base,
        None => format!(
            "https://{CLAUDE_PLATFORM_SERVICE_NAME}.{}.api.aws",
            resolve_region(settings, env_lookup)?
        ),
    };
    Ok(complete_messages_url(&api_base))
}

fn is_valid_region(region: &str) -> bool {
    !region.is_empty()
        && region
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
}

fn non_blank_env(env_lookup: &dyn Fn(&str) -> Option<String>, name: &str) -> Option<String> {
    env_lookup(name).filter(|value| !value.trim().is_empty())
}

fn aws_string_params(params: &Map<String, Value>) -> impl Iterator<Item = (String, String)> + '_ {
    params
        .iter()
        .filter(|(key, _)| key.starts_with(AWS_PARAM_PREFIX))
        .filter_map(|(key, value)| Some((key.clone(), value.as_str()?.to_string())))
}

fn workspace_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) if !text.is_empty() => Some(text.clone()),
        Value::Number(number) if number.as_f64() != Some(0.0) => Some(number.to_string()),
        _ => None,
    }
}

fn param_name(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

fn json_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn map(value: Value) -> Map<String, Value> {
        match value {
            Value::Object(fields) => fields,
            other => panic!("expected an object, got {other}"),
        }
    }

    fn env(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |name| {
            pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        }
    }

    #[rstest]
    #[case::request_beats_deployment_for_the_same_key(
        json!({"workspace_id": "req"}), json!({"workspace_id": "dep"}), Some("req")
    )]
    #[case::deployment_workspace_id_beats_request_aws_workspace_id(
        json!({"aws_workspace_id": "req-aws"}), json!({"workspace_id": "dep"}), Some("dep")
    )]
    #[case::header_style_key_beats_snake_case_anthropic_key(
        json!({"anthropic_workspace_id": "snake"}), json!({"anthropic-workspace-id": "dash"}), Some("dash")
    )]
    #[case::empty_value_falls_through(
        json!({"workspace_id": ""}), json!({"anthropic_workspace_id": "last"}), Some("last")
    )]
    #[case::number_is_stringified(json!({"workspace_id": 42}), json!({}), Some("42"))]
    #[case::none_configured(json!({}), json!({}), None)]
    fn workspace_id_follows_python_key_precedence(
        #[case] request: Value,
        #[case] deployment: Value,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(
            ClaudePlatformSettings::from_params(&map(request), &map(deployment))
                .workspace_id
                .as_deref(),
            expected
        );
    }

    #[rstest]
    #[case::absent(json!({}), json!({}), UnsupportedParams::Default)]
    #[case::request_list_replaces_deployment_list(
        json!({UNSUPPORTED_PARAMS_OVERRIDE_KEY: ["speed"]}),
        json!({UNSUPPORTED_PARAMS_OVERRIDE_KEY: ["top_k"]}),
        UnsupportedParams::Replaced(BTreeSet::from(["speed".to_string()]))
    )]
    #[case::null_request_value_defers_to_deployment(
        json!({UNSUPPORTED_PARAMS_OVERRIDE_KEY: null}),
        json!({UNSUPPORTED_PARAMS_OVERRIDE_KEY: []}),
        UnsupportedParams::Replaced(BTreeSet::new())
    )]
    #[case::non_list_is_ignored(
        json!({UNSUPPORTED_PARAMS_OVERRIDE_KEY: "speed"}),
        json!({}),
        UnsupportedParams::Ignored { found: "string" }
    )]
    fn unsupported_params_override_is_read_from_request_then_deployment(
        #[case] request: Value,
        #[case] deployment: Value,
        #[case] expected: UnsupportedParams,
    ) {
        assert_eq!(
            ClaudePlatformSettings::from_params(&map(request), &map(deployment)).unsupported_params,
            expected
        );
    }

    #[rstest]
    #[case::default_rejects_context_management(
        UnsupportedParams::Default,
        "context_management",
        true
    )]
    #[case::default_keeps_others(UnsupportedParams::Default, "speed", false)]
    #[case::ignored_behaves_as_default(UnsupportedParams::Ignored { found: "string" }, "context_management", true)]
    #[case::replacement_does_not_extend_the_default(
        UnsupportedParams::Replaced(BTreeSet::from(["speed".to_string()])), "context_management", false
    )]
    #[case::replacement_rejects_its_keys(
        UnsupportedParams::Replaced(BTreeSet::from(["speed".to_string()])), "speed", true
    )]
    fn unsupported_params_replace_the_default_set(
        #[case] unsupported: UnsupportedParams,
        #[case] key: &str,
        #[case] rejected: bool,
    ) {
        assert_eq!(unsupported.rejects(key), rejected);
    }

    #[test]
    fn aws_params_keep_string_aws_keys_with_the_request_winning() {
        let settings = ClaudePlatformSettings::from_params(
            &map(json!({"aws_region_name": "us-east-1", "aws_profile_name": 3, "top_k": "x"})),
            &map(json!({"aws_region_name": "eu-west-1", "aws_role_name": "r"})),
        );
        assert_eq!(
            settings.aws_params,
            BTreeMap::from([
                ("aws_region_name".to_string(), "us-east-1".to_string()),
                ("aws_role_name".to_string(), "r".to_string()),
            ])
        );
    }

    #[rstest]
    #[case::override_key(UNSUPPORTED_PARAMS_OVERRIDE_KEY, true)]
    #[case::workspace_id("workspace_id", true)]
    #[case::header_style_workspace("anthropic-workspace-id", true)]
    #[case::any_aws_key("aws_bedrock_runtime_endpoint", true)]
    #[case::request_param("max_tokens", false)]
    #[case::aws_without_underscore("awsome", false)]
    fn non_request_params_are_workspace_override_and_aws_keys(
        #[case] key: &str,
        #[case] expected: bool,
    ) {
        assert_eq!(is_non_request_param(key), expected);
    }

    #[rstest]
    #[case::prefixed("claude_platform/claude-x", "claude-x")]
    #[case::stripped_once("claude_platform/claude_platform/claude-x", "claude_platform/claude-x")]
    #[case::bare("claude-x", "claude-x")]
    fn the_route_prefix_is_stripped_once(#[case] model: &str, #[case] expected: &str) {
        assert_eq!(strip_claude_platform_route(model), expected);
    }

    #[rstest]
    #[case::settings_win(Some("from-params"), &[(ANTHROPIC_AWS_WORKSPACE_ID_ENV, "env")], "from-params")]
    #[case::aws_env_before_anthropic_env(
        None, &[(ANTHROPIC_WORKSPACE_ID_ENV, "generic"), (ANTHROPIC_AWS_WORKSPACE_ID_ENV, "aws")], "aws"
    )]
    #[case::generic_env_last(None, &[(ANTHROPIC_WORKSPACE_ID_ENV, "generic")], "generic")]
    fn workspace_id_falls_back_to_the_environment(
        #[case] configured: Option<&str>,
        #[case] pairs: &'static [(&'static str, &'static str)],
        #[case] expected: &str,
    ) {
        let settings = ClaudePlatformSettings {
            workspace_id: configured.map(str::to_string),
            ..ClaudePlatformSettings::EMPTY
        };
        assert_eq!(
            resolve_workspace_id(&settings, &env(pairs)).unwrap(),
            expected
        );
    }

    #[rstest]
    #[case::param_first(
        Some("ap-south-1"), &[(AWS_REGION_NAME, "us-east-1")], "https://aws-external-anthropic.ap-south-1.api.aws/v1/messages"
    )]
    #[case::region_name_env(
        None, &[(AWS_REGION, "us-east-2"), (AWS_REGION_NAME, "us-east-1")], "https://aws-external-anthropic.us-east-1.api.aws/v1/messages"
    )]
    #[case::default_region_env_last(
        None, &[(AWS_DEFAULT_REGION, "eu-west-1")], "https://aws-external-anthropic.eu-west-1.api.aws/v1/messages"
    )]
    #[case::base_url_env_skips_region(
        None, &[(ANTHROPIC_AWS_API_BASE_ENV, "https://b.example"), (ANTHROPIC_AWS_BASE_URL_ENV, "https://a.example/")],
        "https://a.example/v1/messages"
    )]
    fn url_comes_from_a_configured_base_or_the_region(
        #[case] region_param: Option<&str>,
        #[case] pairs: &'static [(&'static str, &'static str)],
        #[case] expected: &str,
    ) {
        let settings = ClaudePlatformSettings {
            aws_params: region_param
                .map(|region| (AWS_REGION_NAME_PARAM.to_string(), region.to_string()))
                .into_iter()
                .collect(),
            ..ClaudePlatformSettings::EMPTY
        };
        assert_eq!(
            complete_claude_platform_url(None, &settings, &env(pairs)).unwrap(),
            expected
        );
    }

    #[test]
    fn an_explicit_api_base_needs_no_region() {
        assert_eq!(
            complete_claude_platform_url(
                Some("https://gw.example/v1/messages"),
                &ClaudePlatformSettings::EMPTY,
                &env(&[]),
            )
            .unwrap(),
            "https://gw.example/v1/messages"
        );
    }

    #[rstest]
    #[case::missing(&[], "Missing AWS region")]
    #[case::uppercase(&[(AWS_REGION_NAME, "US-EAST-1")], "Invalid AWS region format")]
    #[case::host_injection(&[(AWS_REGION_NAME, "evil.com/x")], "Invalid AWS region format")]
    fn a_missing_or_malformed_region_is_an_auth_error(
        #[case] pairs: &'static [(&'static str, &'static str)],
        #[case] message: &str,
    ) {
        let error = complete_claude_platform_url(None, &ClaudePlatformSettings::EMPTY, &env(pairs))
            .unwrap_err();
        assert!(matches!(error, Error::Auth(_)));
        assert!(error.to_string().contains(message), "{error}");
    }
}
