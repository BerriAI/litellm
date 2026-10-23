use litellm_auth::{CredentialPlacement, SecretValue, watsonx::WatsonxIamConfig};
use litellm_core_utils::{settings::resolve_non_empty, url_utils::ApiUrl};
use litellm_llms_types::formats::tokenize::TokenizeResponse;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::{
    Error, ErrorDetail,
    base_llm::{
        auth::{AuthScheme, Headers, ValidatedEnvironment},
        tokenize::transformation::{BaseTokenizeConfig, decode_params, validate_token_count},
    },
};

pub const WATSONX_DEFAULT_API_VERSION: &str = "2024-03-13";

const API_BASE_ENV: &[&str] = &["WATSONX_API_BASE", "WATSONX_URL", "WX_URL", "WML_URL"];
const API_KEY_ENV: &[&str] = &["WATSONX_APIKEY", "WATSONX_API_KEY", "WX_API_KEY"];
const PROJECT_ENV: &[&str] = &["WATSONX_PROJECT_ID", "WX_PROJECT_ID", "PROJECT_ID"];
const SPACE_ENV: &[&str] = &[
    "WATSONX_DEPLOYMENT_SPACE_ID",
    "WATSONX_SPACE_ID",
    "WX_SPACE_ID",
    "SPACE_ID",
];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WatsonxTokenizeParams {
    project_id: Option<String>,
    watsonx_project: Option<String>,
    space_id: Option<String>,
    api_version: Option<String>,
    token: Option<String>,
    zen_api_key: Option<String>,
}

#[derive(Serialize)]
struct WatsonxTokenizeRequest<'a> {
    model_id: &'a str,
    input: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    project_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    space_id: Option<String>,
    parameters: WatsonxTokenizeRequestParams,
}

#[derive(Serialize)]
struct WatsonxTokenizeRequestParams {
    return_tokens: bool,
}

#[derive(Deserialize)]
struct WatsonxTokenizeResponse {
    result: WatsonxTokenizeResult,
}

#[derive(Deserialize)]
struct WatsonxTokenizeResult {
    token_count: u64,
    tokens: Vec<String>,
}

pub struct WatsonxTokenizeConfig;

impl BaseTokenizeConfig for WatsonxTokenizeConfig {
    fn get_supported_tokenize_params(&self) -> &'static [&'static str] {
        &[
            "project_id",
            "watsonx_project",
            "space_id",
            "api_version",
            "token",
            "zen_api_key",
        ]
    }

    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        if litellm_http::request::has_header(&headers, "authorization") {
            return Ok(ValidatedEnvironment {
                headers,
                auth: AuthScheme::Forwarded,
            });
        }
        let params: WatsonxTokenizeParams = decode_params(optional_params)?;
        let auth = if let Some(token) =
            resolve_non_empty(params.token.as_deref(), env_lookup, &["WATSONX_TOKEN"])
        {
            AuthScheme::Credential {
                placement: CredentialPlacement::Bearer,
                secret: SecretValue::new(token),
            }
        } else if let Some(key) = resolve_non_empty(
            params.zen_api_key.as_deref(),
            env_lookup,
            &["WATSONX_ZENAPIKEY"],
        ) {
            AuthScheme::Credential {
                placement: CredentialPlacement::Header("Authorization"),
                secret: SecretValue::new(format!("ZenApiKey {key}")),
            }
        } else {
            let key = resolve_non_empty(api_key, env_lookup, API_KEY_ENV).ok_or(
                litellm_auth::Error::MissingApiKey {
                    provider: "watsonx",
                    environment_variable: "WATSONX_API_KEY",
                },
            )?;
            let config = match resolve_non_empty(None, env_lookup, &["WATSONX_IAM_URL"]) {
                Some(endpoint) => WatsonxIamConfig::with_endpoint(SecretValue::new(key), endpoint),
                None => WatsonxIamConfig::new(SecretValue::new(key)),
            };
            AuthScheme::WatsonxIam {
                config: Box::new(config),
            }
        };
        Ok(ValidatedEnvironment { headers, auth })
    }

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        let params: WatsonxTokenizeParams = decode_params(optional_params)?;
        let base = resolve_non_empty(api_base, env_lookup, API_BASE_ENV).ok_or(
            litellm_auth::Error::MissingApiBase {
                provider: "watsonx",
                guidance: "set api_base or WATSONX_API_BASE",
            },
        )?;
        let version = params
            .api_version
            .as_deref()
            .unwrap_or(WATSONX_DEFAULT_API_VERSION);
        ApiUrl::parse(&base)
            .and_then(|url| url.complete_path(&["ml", "v1", "text", "tokenization"]))
            .map(|url| url.replace_query_pair("version", version).into_string())
            .map_err(|source| Error::InvalidRequest(ErrorDetail::invalid("api_base", source)))
    }

    fn transform_tokenize_request(
        &self,
        model: &str,
        prompt: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<Value, Error> {
        let params: WatsonxTokenizeParams = decode_params(optional_params)?;
        let project_id = resolve_non_empty(
            params
                .project_id
                .as_deref()
                .or(params.watsonx_project.as_deref()),
            env_lookup,
            PROJECT_ENV,
        );
        let space_id = if project_id.is_none() {
            resolve_non_empty(params.space_id.as_deref(), env_lookup, SPACE_ENV)
        } else {
            None
        };
        if project_id.is_none() && space_id.is_none() {
            return Err(Error::MissingField("project_id or space_id"));
        }
        serde_json::to_value(WatsonxTokenizeRequest {
            model_id: model,
            input: prompt,
            project_id,
            space_id,
            parameters: WatsonxTokenizeRequestParams {
                return_tokens: true,
            },
        })
        .map_err(|source| Error::InvalidRequest(ErrorDetail::invalid("tokenize request", source)))
    }

    fn transform_tokenize_response(
        &self,
        model: &str,
        response_json: Value,
    ) -> Result<TokenizeResponse, Error> {
        let response: WatsonxTokenizeResponse =
            serde_json::from_value(response_json).map_err(|source| {
                Error::InvalidResponse(ErrorDetail::invalid("tokenize response", source))
            })?;
        validate_token_count(response.result.token_count, response.result.tokens.len())?;
        Ok(TokenizeResponse {
            model: model.to_string(),
            count: response.result.token_count,
            token_ids: None,
            tokens: Some(response.result.tokens),
        })
    }

    fn secret_names(&self) -> Vec<&'static str> {
        API_BASE_ENV
            .iter()
            .chain(API_KEY_ENV)
            .chain(PROJECT_ENV)
            .chain(SPACE_ENV)
            .copied()
            .chain(["WATSONX_TOKEN", "WATSONX_ZENAPIKEY", "WATSONX_IAM_URL"])
            .collect()
    }

    fn secret_param_names(&self) -> &'static [&'static str] {
        &["token", "zen_api_key"]
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    fn params(value: Value) -> Map<String, Value> {
        serde_json::from_value(value).unwrap()
    }

    #[rstest]
    #[case::project(json!({"project_id": "project"}), json!({"project_id": "project"}))]
    #[case::project_alias(json!({"watsonx_project": "project"}), json!({"project_id": "project"}))]
    #[case::space(json!({"space_id": "space"}), json!({"space_id": "space"}))]
    fn request_maps_scope_without_forwarding_credentials(
        #[case] scope: Value,
        #[case] expected: Value,
    ) {
        let mut options = params(scope);
        options.insert("token".into(), json!("private"));
        options.insert("api_version".into(), json!("2025-01-01"));
        let mut request = WatsonxTokenizeConfig
            .transform_tokenize_request("ibm/model", "Hello", &options, &no_env)
            .unwrap();
        assert_eq!(request["model_id"], "ibm/model");
        assert_eq!(request["input"], "Hello");
        assert_eq!(request["parameters"], json!({"return_tokens": true}));
        let body = request.as_object_mut().unwrap();
        body.remove("model_id");
        body.remove("input");
        body.remove("parameters");
        assert_eq!(request, expected);
    }

    #[rstest]
    fn empty_text_is_forwarded_and_project_can_come_from_environment() {
        let body = WatsonxTokenizeConfig
            .transform_tokenize_request("ibm/model", "", &Map::new(), &|name| {
                (name == "WATSONX_PROJECT_ID").then(|| "configured-project".into())
            })
            .unwrap();
        assert_eq!(body["input"], "");
        assert_eq!(body["project_id"], "configured-project");
    }

    #[rstest]
    #[case::missing_scope(json!({}))]
    #[case::unsupported_option(json!({"project_id": "project", "add_special_tokens": true}))]
    #[case::invalid_scope(json!({"project_id": 123}))]
    #[case::iam_endpoint_override(json!({"project_id": "project", "iam_url": "https://untrusted.test"}))]
    fn invalid_request_parameters_fail(#[case] options: Value) {
        assert!(
            WatsonxTokenizeConfig
                .transform_tokenize_request("model", "Hello", &params(options), &no_env)
                .is_err()
        );
    }

    #[rstest]
    #[case::root(
        "https://example.test",
        "https://example.test/ml/v1/text/tokenization?version=custom"
    )]
    #[case::partial(
        "https://example.test/ml/v1",
        "https://example.test/ml/v1/text/tokenization?version=custom"
    )]
    #[case::complete(
        "https://example.test/ml/v1/text/tokenization",
        "https://example.test/ml/v1/text/tokenization?version=custom"
    )]
    #[case::prefix_and_query(
        "https://example.test/proxy/ml/v1?tenant=a",
        "https://example.test/proxy/ml/v1/text/tokenization?tenant=a&version=custom"
    )]
    #[case::existing_version(
        "https://example.test/ml/v1/text/tokenization?version=old&tenant=a",
        "https://example.test/ml/v1/text/tokenization?tenant=a&version=custom"
    )]
    fn endpoint_completion_preserves_base_paths_and_query(
        #[case] base: &str,
        #[case] expected: &str,
    ) {
        let url = WatsonxTokenizeConfig
            .get_complete_url(
                Some(base),
                "model",
                &params(json!({"api_version": "custom"})),
                &no_env,
            )
            .unwrap();
        assert_eq!(url, expected);
    }

    #[rstest]
    fn api_base_and_version_use_documented_defaults() {
        let url = WatsonxTokenizeConfig
            .get_complete_url(None, "model", &Map::new(), &|name| {
                (name == "WATSONX_URL").then(|| "https://configured.test".into())
            })
            .unwrap();
        assert_eq!(
            url,
            format!(
                "https://configured.test/ml/v1/text/tokenization?version={WATSONX_DEFAULT_API_VERSION}"
            )
        );
        assert!(
            WatsonxTokenizeConfig
                .get_complete_url(None, "model", &Map::new(), &no_env)
                .is_err()
        );
    }

    #[rstest]
    #[case::text(vec!["Hello".to_string(), " world".to_string()])]
    #[case::empty(vec![])]
    fn response_preserves_strings_without_fabricating_ids(#[case] tokens: Vec<String>) {
        let response = WatsonxTokenizeConfig
            .transform_tokenize_response(
                "model",
                json!({"result": {"token_count": tokens.len(), "tokens": tokens}}),
            )
            .unwrap();
        assert_eq!(response.model, "model");
        assert_eq!(response.count, tokens.len() as u64);
        assert_eq!(response.tokens, Some(tokens));
        assert_eq!(response.token_ids, None);
    }

    #[rstest]
    #[case::missing_result(json!({}))]
    #[case::missing_tokens(json!({"result": {"token_count": 0}}))]
    #[case::negative_count(json!({"result": {"token_count": -1, "tokens": []}}))]
    #[case::numeric_tokens(json!({"result": {"token_count": 1, "tokens": [1]}}))]
    #[case::inconsistent_count(json!({"result": {"token_count": 2, "tokens": ["Hello"]}}))]
    fn malformed_responses_fail(#[case] body: Value) {
        assert!(
            WatsonxTokenizeConfig
                .transform_tokenize_response("model", body)
                .is_err()
        );
    }

    #[rstest]
    #[case::bearer(json!({"token": "bearer"}), "bearer", CredentialPlacement::Bearer)]
    #[case::zen(json!({"zen_api_key": "zen"}), "ZenApiKey zen", CredentialPlacement::Header("Authorization"))]
    fn static_credentials_take_precedence_over_iam(
        #[case] options: Value,
        #[case] expected: &str,
        #[case] expected_placement: CredentialPlacement,
    ) {
        let environment = WatsonxTokenizeConfig
            .validate_environment(vec![], Some("iam-key"), &params(options), &no_env)
            .unwrap();
        let AuthScheme::Credential { placement, secret } = environment.auth else {
            panic!("expected a static credential");
        };
        assert_eq!(placement, expected_placement);
        assert_eq!(secret.expose(), expected);
    }

    #[rstest]
    fn forwarded_authorization_wins_over_saved_credentials() {
        let headers = vec![("authorization".into(), "Bearer forwarded".into())];
        let environment = WatsonxTokenizeConfig
            .validate_environment(
                headers.clone(),
                Some("key"),
                &params(json!({"token": "token"})),
                &no_env,
            )
            .unwrap();
        assert!(matches!(environment.auth, AuthScheme::Forwarded));
        assert_eq!(environment.headers, headers);
    }

    #[rstest]
    fn bearer_credentials_can_come_from_environment() {
        let environment = WatsonxTokenizeConfig
            .validate_environment(vec![], None, &Map::new(), &|name| {
                (name == "WATSONX_TOKEN").then(|| "environment-token".into())
            })
            .unwrap();
        let AuthScheme::Credential { placement, secret } = environment.auth else {
            panic!("expected a bearer credential");
        };
        assert_eq!(placement, CredentialPlacement::Bearer);
        assert_eq!(secret.expose(), "environment-token");
    }

    #[rstest]
    fn api_key_uses_shared_iam_authentication() {
        let environment = WatsonxTokenizeConfig
            .validate_environment(vec![], Some("configured-key"), &Map::new(), &no_env)
            .unwrap();
        assert!(matches!(environment.auth, AuthScheme::WatsonxIam { .. }));
        assert!(
            WatsonxTokenizeConfig
                .validate_environment(vec![], None, &Map::new(), &no_env)
                .is_err()
        );
    }
}
