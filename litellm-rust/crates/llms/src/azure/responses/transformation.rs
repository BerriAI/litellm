//! Azure OpenAI Responses, matching `litellm/llms/azure/responses/transformation.py`

use litellm_auth::{CredentialPlacement, SecretValue};
use litellm_llms_types::formats::responses::ResponsesApiResponse;
use serde_json::{Map, Value};
use url::Url;

use crate::{
    Error, ErrorDetail,
    base_llm::{
        auth::{AuthScheme, ValidatedEnvironment},
        responses::transformation::BaseResponsesApiConfig,
    },
    openai::responses::{
        request::ResponsesHost,
        transformation::{responses_request, responses_response},
    },
};

const API_KEY_ENVS: [&str; 2] = ["AZURE_OPENAI_API_KEY", "AZURE_API_KEY"];
const API_BASE_ENV: &str = "AZURE_API_BASE";
const DEFAULT_API_VERSION_ENV: &str = "AZURE_DEFAULT_RESPONSES_API_VERSION";
const DEFAULT_API_VERSION: &str = "preview";
const API_KEY_HEADER: &str = "api-key";
const ROUTE: &str = "/openai/responses";
const MODEL_PREFIXES: [&str; 3] = ["responses/", "o_series/", "azure_ai/"];
/// API versions served from Azure's `/openai/v1` surface
const V1_API_VERSIONS: [&str; 3] = ["preview", "latest", "v1"];

pub struct AzureResponsesApiConfig;
pub const AZURE_RESPONSES_CONFIG: AzureResponsesApiConfig = AzureResponsesApiConfig;

impl BaseResponsesApiConfig for AzureResponsesApiConfig {
    fn secret_names(
        &self,
        api_key: Option<&str>,
        api_base: Option<&str>,
    ) -> &'static [&'static str] {
        match (
            api_key.is_some_and(|key| !key.is_empty()),
            api_base.is_some_and(|base| !base.is_empty()),
        ) {
            (true, true) => &[DEFAULT_API_VERSION_ENV],
            (true, false) => &[API_BASE_ENV, DEFAULT_API_VERSION_ENV],
            (false, true) => &[
                "AZURE_OPENAI_API_KEY",
                "AZURE_API_KEY",
                DEFAULT_API_VERSION_ENV,
            ],
            (false, false) => &[
                "AZURE_OPENAI_API_KEY",
                "AZURE_API_KEY",
                API_BASE_ENV,
                DEFAULT_API_VERSION_ENV,
            ],
        }
    }

    fn validate_environment(
        &self,
        headers: Vec<(String, String)>,
        api_key: Option<&str>,
        lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        if headers.iter().any(|(name, _)| name == API_KEY_HEADER) {
            return Ok(ValidatedEnvironment {
                headers,
                auth: AuthScheme::Forwarded,
            });
        }
        let key = non_empty(api_key)
            .map(str::to_owned)
            .or_else(|| {
                API_KEY_ENVS
                    .iter()
                    .find_map(|name| lookup(name).filter(|key| !key.is_empty()))
            })
            .ok_or(litellm_auth::Error::MissingApiKey {
                provider: "Azure",
                environment_variable: API_KEY_ENVS[0],
            })?;
        Ok(ValidatedEnvironment {
            headers,
            auth: AuthScheme::Credential {
                placement: CredentialPlacement::Header(API_KEY_HEADER),
                secret: SecretValue::new(key),
            },
        })
    }

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        api_version: Option<&str>,
        lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        let api_base = non_empty(api_base)
            .map(str::to_owned)
            .or_else(|| lookup(API_BASE_ENV).filter(|base| !base.is_empty()))
            .ok_or(litellm_auth::Error::MissingApiBase {
                provider: "Azure",
                guidance: "Set `api_base` or the AZURE_API_BASE environment variable",
            })?;
        let api_version = non_empty(api_version).map_or_else(
            || lookup(DEFAULT_API_VERSION_ENV).unwrap_or_else(|| DEFAULT_API_VERSION.into()),
            str::to_owned,
        );
        complete_url(&api_base, &api_version)
    }

    fn transform_responses_api_request(
        &self,
        model: &str,
        input: Value,
        mut params: Map<String, Value>,
    ) -> Result<Value, Error> {
        if let Some(Value::Array(tools)) = params.get_mut("tools") {
            tools.iter_mut().for_each(flatten_function_tool);
        }
        let model = MODEL_PREFIXES
            .iter()
            .fold(model.to_owned(), |model, prefix| model.replace(prefix, ""));
        responses_request(&model, input, params, ResponsesHost::Azure)
    }

    fn transform_response_api_response(&self, body: Value) -> Result<ResponsesApiResponse, Error> {
        responses_response(body)
    }
}

fn non_empty(value: Option<&str>) -> Option<&str> {
    value.filter(|value| !value.is_empty())
}

/// Azure takes Chat Completions-style `{"type": "function", "function": {...}}` tools only with
/// the function's fields lifted to the top level
fn flatten_function_tool(tool: &mut Value) {
    let Some(fields) = tool.as_object_mut() else {
        return;
    };
    let Some(function) = fields.remove("function") else {
        return;
    };
    match function {
        Value::Object(function) => fields.extend(function),
        other => {
            fields.insert("function".into(), other);
        }
    }
}

/// Python's `BaseAzureLLM._get_base_azure_url` for the Responses route
fn complete_url(api_base: &str, api_version: &str) -> Result<String, Error> {
    let mut url = Url::parse(api_base)
        .map_err(|error| Error::InvalidRequest(ErrorDetail::invalid("Azure api_base", error)))?;
    let mut query: Vec<(String, String)> = Vec::new();
    for (name, value) in url.query_pairs() {
        if !query.iter().any(|(seen, _)| *seen == name) {
            query.push((name.into_owned(), value.into_owned()));
        }
    }
    if !api_version.is_empty() && !query.iter().any(|(name, _)| name == "api-version") {
        query.push(("api-version".into(), api_version.into()));
    }
    if !api_base.contains(ROUTE) {
        let path = joined_path(url.path(), ROUTE);
        url.set_path(&path);
    }
    if V1_API_VERSIONS.contains(&api_version) && !url.as_str().contains("/openai/v1") {
        let path = url.path().replace("/openai", "/openai/v1");
        url.set_path(&path);
    }
    url.set_query(None);
    if !query.is_empty() {
        url.query_pairs_mut().extend_pairs(&query);
    }
    Ok(url.into())
}

/// Python's `add_path_to_api_base`: appends `ending` without repeating segments the base already
/// ends with
fn joined_path(base: &str, ending: &str) -> String {
    let base: Vec<&str> = base
        .split('/')
        .filter(|segment| !segment.is_empty())
        .collect();
    let ending: Vec<&str> = ending
        .split('/')
        .filter(|segment| !segment.is_empty())
        .collect();
    let segments = (0..base.len())
        .find(|&index| base[index..] == ending[..(base.len() - index).min(ending.len())])
        .map_or_else(
            || [base.as_slice(), ending.as_slice()].concat(),
            |index| [&base[..index], ending.as_slice()].concat(),
        );
    format!("/{}", segments.join("/"))
}

#[cfg(test)]
mod tests {
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

    #[rstest]
    #[case::resource_default_preview(
        "https://res.openai.azure.com",
        None,
        "https://res.openai.azure.com/openai/v1/responses?api-version=preview"
    )]
    #[case::dated_version(
        "https://res.openai.azure.com",
        Some("2025-03-01-preview"),
        "https://res.openai.azure.com/openai/responses?api-version=2025-03-01-preview"
    )]
    #[case::trailing_openai_segment(
        "https://res.openai.azure.com/openai",
        Some("2025-03-01-preview"),
        "https://res.openai.azure.com/openai/responses?api-version=2025-03-01-preview"
    )]
    #[case::full_route_keeps_its_version(
        "https://res.openai.azure.com/openai/responses?api-version=2024-05-01-preview",
        Some("2025-03-01-preview"),
        "https://res.openai.azure.com/openai/responses?api-version=2024-05-01-preview"
    )]
    #[case::full_route_moves_to_v1(
        "https://res.openai.azure.com/openai/responses",
        Some("latest"),
        "https://res.openai.azure.com/openai/v1/responses?api-version=latest"
    )]
    #[case::v1_base(
        "https://res.openai.azure.com/openai/v1",
        Some("v1"),
        "https://res.openai.azure.com/openai/v1/openai/responses?api-version=v1"
    )]
    #[case::proxy_path(
        "http://127.0.0.1:8080/azure/",
        Some("2025-03-01-preview"),
        "http://127.0.0.1:8080/azure/openai/responses?api-version=2025-03-01-preview"
    )]
    fn urls_match_python(
        #[case] api_base: &str,
        #[case] api_version: Option<&str>,
        #[case] expected: &str,
    ) {
        assert_eq!(
            AZURE_RESPONSES_CONFIG
                .get_complete_url(Some(api_base), api_version, &env(&[]))
                .unwrap(),
            expected
        );
    }

    #[rstest]
    #[case::configured_default(&[(DEFAULT_API_VERSION_ENV, "2025-04-01-preview")], "https://res.openai.azure.com/openai/responses?api-version=2025-04-01-preview")]
    #[case::empty_default_sends_no_version(&[(DEFAULT_API_VERSION_ENV, "")], "https://res.openai.azure.com/openai/responses")]
    fn default_version_comes_from_the_environment(
        #[case] secrets: &'static [(&'static str, &'static str)],
        #[case] expected: &str,
    ) {
        assert_eq!(
            AZURE_RESPONSES_CONFIG
                .get_complete_url(Some("https://res.openai.azure.com"), None, &env(secrets))
                .unwrap(),
            expected
        );
    }

    #[test]
    fn api_base_falls_back_to_the_environment() {
        let lookup = env(&[(API_BASE_ENV, "https://env.openai.azure.com")]);
        assert_eq!(
            AZURE_RESPONSES_CONFIG
                .get_complete_url(None, Some("v1"), &lookup)
                .unwrap(),
            "https://env.openai.azure.com/openai/v1/responses?api-version=v1"
        );
        assert!(
            AZURE_RESPONSES_CONFIG
                .get_complete_url(None, None, &env(&[]))
                .is_err()
        );
    }

    #[rstest]
    #[case::explicit(Some("explicit"), &[("AZURE_OPENAI_API_KEY", "managed")], "explicit")]
    #[case::openai_key_env(None, &[("AZURE_OPENAI_API_KEY", "managed"), ("AZURE_API_KEY", "other")], "managed")]
    #[case::api_key_env(Some(""), &[("AZURE_API_KEY", "other")], "other")]
    fn api_key_goes_in_the_api_key_header(
        #[case] api_key: Option<&str>,
        #[case] secrets: &'static [(&'static str, &'static str)],
        #[case] expected: &str,
    ) {
        let environment = AZURE_RESPONSES_CONFIG
            .validate_environment(Vec::new(), api_key, &env(secrets))
            .unwrap();
        let AuthScheme::Credential { placement, secret } = environment.auth else {
            panic!("expected a credential")
        };
        assert_eq!(
            (placement.header_name(), secret.expose()),
            (API_KEY_HEADER, expected)
        );
    }

    #[test]
    fn a_forwarded_api_key_header_is_kept() {
        let headers = vec![(API_KEY_HEADER.to_owned(), "caller".to_owned())];
        let environment = AZURE_RESPONSES_CONFIG
            .validate_environment(headers.clone(), Some("deployment"), &env(&[]))
            .unwrap();
        assert!(matches!(environment.auth, AuthScheme::Forwarded));
        assert_eq!(environment.headers, headers);
    }

    #[test]
    fn missing_key_is_an_auth_error() {
        assert!(
            AZURE_RESPONSES_CONFIG
                .validate_environment(Vec::new(), None, &env(&[]))
                .is_err()
        );
    }

    #[test]
    fn request_flattens_tools_strips_model_prefixes_and_status() {
        let Value::Object(params) = json!({"tools": [
            {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}},
            {"type": "web_search"}
        ]}) else {
            unreachable!()
        };
        let body = AZURE_RESPONSES_CONFIG
            .transform_responses_api_request(
                "responses/o_series/my-deployment",
                json!([
                    {"type": "message", "role": "user", "content": "hi", "status": "completed"},
                    {"type": "reasoning", "id": "rs_1", "summary": [], "status": "completed"},
                    {"type": "reasoning", "summary": [], "status": "completed"}
                ]),
                params,
            )
            .unwrap();
        assert_eq!(
            body,
            json!({
                "tools": [
                    {"type": "function", "name": "get_weather", "parameters": {"type": "object"}},
                    {"type": "web_search"}
                ],
                "model": "my-deployment",
                "input": [
                    {"type": "message", "role": "user", "content": "hi"},
                    {"type": "reasoning", "id": "rs_1", "summary": []},
                    {"type": "reasoning", "summary": [], "status": "completed"}
                ]
            })
        );
    }

    #[rstest]
    #[case::overlap("/a/openai", "/openai/responses", "/a/openai/responses")]
    #[case::no_overlap("/a", "/openai/responses", "/a/openai/responses")]
    #[case::root("/", "/openai/responses", "/openai/responses")]
    #[case::partial_overlap(
        "/openai/responses/x",
        "/openai/responses",
        "/openai/responses/x/openai/responses"
    )]
    fn joined_path_matches_add_path_to_api_base(
        #[case] base: &str,
        #[case] ending: &str,
        #[case] expected: &str,
    ) {
        assert_eq!(joined_path(base, ending), expected);
    }
}
