//! Bedrock Mantle Responses, matching `litellm/llms/bedrock_mantle/responses/transformation.py`

use std::sync::LazyLock;

use litellm_auth::{AwsParams, CredentialPlacement, SecretValue};
use litellm_auth_aws::{AwsCredentialSource, is_bedrock_region};
use litellm_http::request::{with_header, without_headers};
use litellm_llms_types::formats::responses::ResponsesApiResponse;
use litellm_python_compat::truthy::json_truthy;
use litellm_router_types::LitellmParams;
use serde_json::{Map, Value};

use crate::{
    Error,
    base_llm::{
        auth::{AuthScheme, ValidatedEnvironment},
        responses::{
            additional_tools::hoist_additional_tools,
            codex_compat::{drop_unsupported_tools, normalize_codex_input_items},
            transformation::BaseResponsesApiConfig,
        },
    },
    bedrock::common_utils::validated_region,
    openai::responses::{
        request::ResponsesHost,
        transformation::{responses_request, responses_response},
    },
};

const KEY_ENVS: [&str; 2] = ["BEDROCK_MANTLE_API_KEY", "AWS_BEARER_TOKEN_BEDROCK"];
const API_BASE_ENV: &str = "BEDROCK_MANTLE_API_BASE";
const REGION_ENVS: [&str; 3] = ["BEDROCK_MANTLE_REGION", "AWS_REGION_NAME", "AWS_REGION"];
const DEFAULT_REGION: &str = "us-east-1";
const BASE_SUFFIXES: [&str; 5] = [
    "/openai/v1/responses",
    "/v1/responses",
    "/responses",
    "/openai/v1",
    "/v1",
];
/// Per Bedrock Mantle Responses API validation errors
const SUPPORTED_TOOL_TYPES: [&str; 6] = [
    "function",
    "mcp",
    "custom",
    "namespace",
    "tool_search",
    "web_search",
];
/// The price map's `use_openai_responses_path`, which the host reads for the model
const OPENAI_PATH_PARAM: &str = "use_openai_responses_path";
const PROJECT_PARAM: &str = "aws_bedrock_project_id";

/// One call's Mantle connection, since the model picks the Responses path
pub struct BedrockMantleResponsesApiConfig {
    api_base: Option<String>,
    model_region: Option<String>,
    aws: AwsParams,
    project_id: Option<String>,
    openai_path: bool,
}

impl BedrockMantleResponsesApiConfig {
    pub fn new(api_base: Option<&str>, model_region: Option<&str>, params: &LitellmParams) -> Self {
        Self {
            api_base: api_base.map(str::to_owned),
            model_region: model_region.map(str::to_owned),
            aws: params.aws.clone(),
            project_id: params
                .extra
                .get(PROJECT_PARAM)
                .and_then(Value::as_str)
                .filter(|project| !project.is_empty())
                .map(str::to_owned),
            openai_path: params.extra.get(OPENAI_PATH_PARAM) == Some(&Value::Bool(true)),
        }
    }

    /// Python's `resolve_mantle_region`; the model's region prefix stands in for the default
    /// base `get_llm_provider` builds from it
    fn region(
        &self,
        api_base: Option<&str>,
        lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        if let Some(region) = non_empty(self.aws.aws_region_name.as_deref()) {
            return validated_region(region.to_owned());
        }
        let routed = match configured_base(api_base, lookup) {
            Some(base) => mantle_host_region(base.trim_end_matches('/')).map(str::to_owned),
            None => self.model_region.clone(),
        };
        validated_region(
            routed
                .or_else(|| {
                    REGION_ENVS
                        .iter()
                        .find_map(|name| lookup(name).filter(|value| !value.is_empty()))
                })
                .unwrap_or_else(|| DEFAULT_REGION.into()),
        )
    }
}

/// A leading Bedrock region in a Mantle model id routes the call and is not sent
pub fn split_region_prefix(model: &str) -> (Option<&str>, &str) {
    match model.split_once('/') {
        Some((region, rest)) if is_bedrock_region(region) => (Some(region), rest),
        _ => (None, model),
    }
}

impl BaseResponsesApiConfig for BedrockMantleResponsesApiConfig {
    fn secret_names(
        &self,
        _api_key: Option<&str>,
        _api_base: Option<&str>,
    ) -> &'static [&'static str] {
        static NAMES: LazyLock<Box<[&'static str]>> = LazyLock::new(|| {
            KEY_ENVS
                .into_iter()
                .chain([API_BASE_ENV])
                .chain(REGION_ENVS)
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

    fn validate_environment(
        &self,
        headers: Vec<(String, String)>,
        api_key: Option<&str>,
        lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        let headers = with_header(headers, "content-type", "application/json".into());
        let headers = match &self.project_id {
            Some(project) => with_header(headers, "OpenAI-Project", project.clone()),
            None => headers,
        };
        let bearer = non_empty(api_key).map(str::to_owned).or_else(|| {
            KEY_ENVS
                .iter()
                .find_map(|name| lookup(name).filter(|value| !value.is_empty()))
        });
        Ok(match bearer {
            Some(token) => ValidatedEnvironment {
                headers,
                auth: AuthScheme::Credential {
                    placement: CredentialPlacement::Bearer,
                    secret: SecretValue::new(token),
                },
            },
            None => ValidatedEnvironment {
                headers: without_headers(headers, &["authorization"]),
                auth: AuthScheme::AwsSigV4 {
                    region: self.region(self.api_base.as_deref(), lookup)?,
                    service: "bedrock",
                    credentials: Box::new(AwsCredentialSource::from_params(&self.aws, lookup)),
                },
            },
        })
    }

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        _api_version: Option<&str>,
        lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        let region = self.region(api_base, lookup)?;
        let default = format!("https://bedrock-mantle.{region}.api.aws");
        let configured = configured_base(api_base, lookup).unwrap_or_else(|| default.clone());
        let trimmed = configured.trim_end_matches('/');
        let stripped = BASE_SUFFIXES
            .iter()
            .find_map(|suffix| trimmed.strip_suffix(suffix))
            .unwrap_or(trimmed);
        let base = match mantle_host_region(stripped) {
            Some(_) => default.as_str(),
            None => stripped,
        };
        let path = match self.openai_path {
            true => "/openai/v1/responses",
            false => "/v1/responses",
        };
        Ok(format!("{base}{path}"))
    }

    fn transform_responses_api_request(
        &self,
        model: &str,
        input: Value,
        params: Map<String, Value>,
    ) -> Result<Value, Error> {
        let params = with_supported_tools(params);
        let hoisted = hoist_additional_tools(input);
        let params = match hoisted.hoisted.is_empty() {
            true => params,
            false => with_hoisted_tools(params, hoisted.hoisted),
        };
        responses_request(
            model,
            normalize_codex_input_items(hoisted.input),
            params,
            ResponsesHost::OpenAi,
        )
    }

    fn transform_response_api_response(&self, body: Value) -> Result<ResponsesApiResponse, Error> {
        responses_response(body)
    }
}

fn non_empty(value: Option<&str>) -> Option<&str> {
    value.filter(|value| !value.is_empty())
}

fn configured_base(
    api_base: Option<&str>,
    lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    non_empty(api_base)
        .map(str::to_owned)
        .or_else(|| lookup(API_BASE_ENV).filter(|base| !base.is_empty()))
}

/// Python's `MANTLE_HOST_RE`: `^https?://bedrock-mantle\.([^/.]+)\.api\.aws(?=/|$)`, ignoring case
fn mantle_host_region(base: &str) -> Option<&str> {
    let lowered = base.to_ascii_lowercase();
    let scheme = ["https://", "http://"]
        .into_iter()
        .find(|scheme| lowered.starts_with(scheme))?;
    let host = &base[scheme.len()..];
    let region_start = "bedrock-mantle.".len();
    if !host
        .get(..region_start)?
        .eq_ignore_ascii_case("bedrock-mantle.")
    {
        return None;
    }
    let after = &host[region_start..];
    let region_end = after.find(['.', '/']).filter(|end| *end > 0)?;
    let rest = &after[region_end..];
    let suffix = ".api.aws";
    let matched = rest.get(..suffix.len())?.eq_ignore_ascii_case(suffix)
        && matches!(rest.as_bytes().get(suffix.len()), None | Some(b'/'));
    matched.then(|| &after[..region_end])
}

/// Python's `map_openai_params` tool filter: a single tool is listed, and no supported tool
/// leaves no `tools`
fn with_supported_tools(mut params: Map<String, Value>) -> Map<String, Value> {
    if !params.get("tools").is_some_and(json_truthy) {
        return params;
    }
    let listed = match params.remove("tools") {
        Some(Value::Array(tools)) => tools,
        tool => tool.into_iter().collect(),
    };
    with_tools(
        params,
        drop_unsupported_tools(listed, &SUPPORTED_TOOL_TYPES),
    )
}

fn with_hoisted_tools(mut params: Map<String, Value>, hoisted: Vec<Value>) -> Map<String, Value> {
    let existing = match params.remove("tools") {
        Some(Value::Array(tools)) => tools,
        _ => Vec::new(),
    };
    let combined = existing.into_iter().chain(hoisted).collect();
    with_tools(
        params,
        drop_unsupported_tools(combined, &SUPPORTED_TOOL_TYPES),
    )
}

fn with_tools(mut params: Map<String, Value>, tools: Vec<Value>) -> Map<String, Value> {
    if !tools.is_empty() {
        params.insert("tools".into(), Value::Array(tools));
    }
    params
}

#[cfg(test)]
mod tests {
    use litellm_router_types::LitellmParams;
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn env(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |name| {
            pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| (*value).to_owned())
        }
    }

    fn params(mut fields: Value) -> LitellmParams {
        fields["model"] = json!("m");
        serde_json::from_value(fields).unwrap()
    }

    fn config(
        api_base: Option<&str>,
        model_region: Option<&str>,
        fields: Value,
    ) -> BedrockMantleResponsesApiConfig {
        BedrockMantleResponsesApiConfig::new(api_base, model_region, &params(fields))
    }

    #[rstest]
    #[case::default_region(None, None, json!({}), &[], "https://bedrock-mantle.us-east-1.api.aws/v1/responses")]
    #[case::openai_path(None, None, json!({"use_openai_responses_path": true}), &[], "https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses")]
    #[case::model_prefix_region(None, Some("us-west-2"), json!({}), &[], "https://bedrock-mantle.us-west-2.api.aws/v1/responses")]
    #[case::region_env(None, None, json!({}), &[("AWS_REGION", "eu-west-1")], "https://bedrock-mantle.eu-west-1.api.aws/v1/responses")]
    #[case::aws_region_name_repins_a_mantle_host(
        Some("https://bedrock-mantle.us-east-1.api.aws/openai/v1"),
        None,
        json!({"aws_region_name": "us-west-2", "use_openai_responses_path": true}),
        &[],
        "https://bedrock-mantle.us-west-2.api.aws/openai/v1/responses"
    )]
    #[case::mantle_host_region_beats_the_env(
        Some("https://Bedrock-Mantle.eu-central-1.api.aws/v1/responses/"),
        None,
        json!({}),
        &[("BEDROCK_MANTLE_REGION", "us-west-2")],
        "https://bedrock-mantle.eu-central-1.api.aws/v1/responses"
    )]
    #[case::custom_proxy_keeps_its_path(
        Some("https://proxy.test/tenant/v1"),
        Some("us-west-2"),
        json!({}),
        &[],
        "https://proxy.test/tenant/v1/responses"
    )]
    #[case::base_secret(None, None, json!({}), &[("BEDROCK_MANTLE_API_BASE", "https://proxy.test")], "https://proxy.test/v1/responses")]
    fn url_matches_python(
        #[case] api_base: Option<&str>,
        #[case] model_region: Option<&str>,
        #[case] fields: Value,
        #[case] secrets: &'static [(&'static str, &'static str)],
        #[case] expected: &str,
    ) {
        assert_eq!(
            config(api_base, model_region, fields)
                .get_complete_url(api_base, None, &env(secrets))
                .unwrap(),
            expected
        );
    }

    #[rstest]
    #[case::https("https://bedrock-mantle.us-east-1.api.aws", Some("us-east-1"))]
    #[case::path("http://bedrock-mantle.us-east-1.api.aws/tenant", Some("us-east-1"))]
    #[case::port("https://bedrock-mantle.us-east-1.api.aws:443", None)]
    #[case::other_domain("https://bedrock-mantle.us-east-1.api.aws.test", None)]
    #[case::empty_region("https://bedrock-mantle..api.aws", None)]
    #[case::proxy("https://proxy.test", None)]
    fn mantle_hosts_match_pythons_pattern(#[case] base: &str, #[case] expected: Option<&str>) {
        assert_eq!(mantle_host_region(base), expected);
    }

    #[rstest]
    #[case::regional("us-west-2/openai.gpt-5.5", Some("us-west-2"), "openai.gpt-5.5")]
    #[case::gov(
        "us-gov-west-1/openai.gpt-5.4",
        Some("us-gov-west-1"),
        "openai.gpt-5.4"
    )]
    #[case::bare("openai.gpt-5.5", None, "openai.gpt-5.5")]
    fn region_prefixes_route_and_are_not_sent(
        #[case] model: &str,
        #[case] region: Option<&str>,
        #[case] wire: &str,
    ) {
        assert_eq!(split_region_prefix(model), (region, wire));
    }

    #[rstest]
    #[case::explicit(Some("explicit"), &[("BEDROCK_MANTLE_API_KEY", "env")], "explicit")]
    #[case::mantle_secret(None, &[("BEDROCK_MANTLE_API_KEY", "mantle"), ("AWS_BEARER_TOKEN_BEDROCK", "aws")], "mantle")]
    #[case::bedrock_secret(Some(""), &[("AWS_BEARER_TOKEN_BEDROCK", "aws")], "aws")]
    fn bearer_tokens_skip_sigv4(
        #[case] api_key: Option<&str>,
        #[case] secrets: &'static [(&'static str, &'static str)],
        #[case] expected: &str,
    ) {
        let environment = config(None, None, json!({}))
            .validate_environment(vec![], api_key, &env(secrets))
            .unwrap();
        let AuthScheme::Credential { placement, secret } = environment.auth else {
            panic!("expected a bearer credential");
        };
        assert_eq!(
            (placement, secret.expose()),
            (CredentialPlacement::Bearer, expected)
        );
    }

    #[test]
    fn without_a_bearer_the_call_is_sigv4_signed_for_the_url_region() {
        let environment = config(
            Some("https://bedrock-mantle.eu-west-1.api.aws/openai/v1"),
            Some("us-west-2"),
            json!({"aws_bedrock_project_id": "proj_1"}),
        )
        .validate_environment(
            vec![("Authorization".into(), "Bearer caller".into())],
            None,
            &env(&[]),
        )
        .unwrap();
        assert!(matches!(
            environment.auth,
            AuthScheme::AwsSigV4 { ref region, service: "bedrock", .. } if region == "eu-west-1"
        ));
        assert_eq!(
            environment.headers,
            vec![
                ("content-type".to_owned(), "application/json".to_owned()),
                ("openai-project".to_owned(), "proj_1".to_owned()),
            ]
        );
    }

    #[test]
    fn an_invalid_region_is_a_request_error() {
        assert!(matches!(
            config(None, None, json!({"aws_region_name": "US EAST"})).get_complete_url(
                None,
                None,
                &env(&[])
            ),
            Err(Error::InvalidRequest(_))
        ));
    }

    #[rstest]
    #[case::unsupported_dropped(
        json!([{"type": "function", "name": "f"}, {"type": "file_search"}]),
        json!("hi"),
        Some(json!([{"type": "function", "name": "f"}]))
    )]
    #[case::single_tool_listed(json!({"type": "web_search"}), json!("hi"), Some(json!([{"type": "web_search"}])))]
    #[case::none_supported(json!([{"type": "image_generation"}]), json!("hi"), None)]
    #[case::empty_list_kept(json!([]), json!("hi"), Some(json!([])))]
    #[case::hoisted_after_existing(
        json!([{"type": "function", "name": "f"}]),
        json!([{"type": "additional_tools", "tools": [{"type": "mcp", "server_label": "s"}, {"type": "code_interpreter"}]}, {"role": "user", "content": "hi"}]),
        Some(json!([{"type": "function", "name": "f"}, {"type": "mcp", "server_label": "s"}]))
    )]
    #[case::hoisted_into_empty_tools(
        json!([]),
        json!([{"type": "additional_tools", "tools": [{"type": "code_interpreter"}]}, {"role": "user", "content": "hi"}]),
        None
    )]
    fn tools_are_filtered_to_what_mantle_accepts(
        #[case] tools: Value,
        #[case] input: Value,
        #[case] expected: Option<Value>,
    ) {
        let body = config(None, None, json!({}))
            .transform_responses_api_request(
                "openai.gpt-5.5",
                input,
                Map::from_iter([("tools".into(), tools)]),
            )
            .unwrap();
        assert_eq!(body.get("tools").cloned(), expected);
    }

    #[test]
    fn codex_history_is_rewritten_after_hoisting() {
        let body = config(None, None, json!({}))
            .transform_responses_api_request(
                "openai.gpt-5.5",
                json!([
                    {"type": "additional_tools", "tools": [{"type": "function", "name": "f"}]},
                    {"type": "context_compaction", "encrypted_content": "enc"},
                    {"role": "user", "content": "hi"},
                ]),
                Map::new(),
            )
            .unwrap();
        assert_eq!(
            (&body["input"], &body["tools"]),
            (
                &json!([{"type": "compaction", "encrypted_content": "enc"}, {"role": "user", "content": "hi"}]),
                &json!([{"type": "function", "name": "f"}])
            )
        );
    }
}
