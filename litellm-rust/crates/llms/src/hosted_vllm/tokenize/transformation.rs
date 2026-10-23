use litellm_auth::{CredentialPlacement, SecretValue};
use litellm_core_utils::settings::resolve_non_empty;
use litellm_llms_types::formats::tokenize::TokenizeResponse;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use url::Url;

use crate::{
    Error, ErrorDetail,
    base_llm::{
        auth::{AuthScheme, Headers, ValidatedEnvironment},
        tokenize::transformation::{BaseTokenizeConfig, decode_params, validate_token_count},
    },
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HostedVllmTokenizeParams {
    add_special_tokens: Option<bool>,
}

#[derive(Serialize)]
struct HostedVllmTokenizeRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    add_special_tokens: bool,
    return_token_strs: bool,
}

#[derive(Deserialize)]
struct HostedVllmTokenizeResponse {
    count: u64,
    tokens: Vec<u64>,
    token_strs: Option<Vec<String>>,
}

pub struct HostedVllmTokenizeConfig;

impl BaseTokenizeConfig for HostedVllmTokenizeConfig {
    fn get_supported_tokenize_params(&self) -> &'static [&'static str] {
        &["add_special_tokens"]
    }

    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        _optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        if litellm_http::request::has_header(&headers, "authorization") {
            return Ok(ValidatedEnvironment {
                headers,
                auth: AuthScheme::Forwarded,
            });
        }
        let key = resolve_non_empty(api_key, env_lookup, &["HOSTED_VLLM_API_KEY"]);
        let auth = match key {
            Some(key) => AuthScheme::Credential {
                placement: CredentialPlacement::Bearer,
                secret: SecretValue::new(key),
            },
            None => AuthScheme::Forwarded,
        };
        Ok(ValidatedEnvironment { headers, auth })
    }

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        _optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        let base = resolve_non_empty(api_base, env_lookup, &["HOSTED_VLLM_API_BASE"]).ok_or(
            litellm_auth::Error::MissingApiBase {
                provider: "hosted_vllm",
                guidance: "set api_base or HOSTED_VLLM_API_BASE",
            },
        )?;
        tokenize_url(&base)
    }

    fn transform_tokenize_request(
        &self,
        model: &str,
        prompt: &str,
        optional_params: &Map<String, Value>,
        _env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<Value, Error> {
        let params: HostedVllmTokenizeParams = decode_params(optional_params)?;
        serde_json::to_value(HostedVllmTokenizeRequest {
            model,
            prompt,
            add_special_tokens: params.add_special_tokens.unwrap_or(true),
            return_token_strs: true,
        })
        .map_err(|source| Error::InvalidRequest(ErrorDetail::invalid("tokenize request", source)))
    }

    fn transform_tokenize_response(
        &self,
        model: &str,
        response_json: Value,
    ) -> Result<TokenizeResponse, Error> {
        let response: HostedVllmTokenizeResponse =
            serde_json::from_value(response_json).map_err(|source| {
                Error::InvalidResponse(ErrorDetail::invalid("tokenize response", source))
            })?;
        validate_token_count(response.count, response.tokens.len())?;
        if let Some(tokens) = &response.token_strs {
            validate_token_count(response.count, tokens.len())?;
        }
        Ok(TokenizeResponse {
            model: model.to_string(),
            count: response.count,
            token_ids: Some(response.tokens),
            tokens: response.token_strs,
        })
    }

    fn secret_names(&self) -> Vec<&'static str> {
        vec!["HOSTED_VLLM_API_KEY", "HOSTED_VLLM_API_BASE"]
    }
}

fn tokenize_url(base: &str) -> Result<String, Error> {
    let mut url = Url::parse(base.trim())
        .map_err(|source| Error::InvalidRequest(ErrorDetail::invalid("api_base", source)))?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(Error::Unsupported("api_base URL scheme"));
    }
    let path = url.path().trim_end_matches('/');
    let path = path.strip_suffix("/v1").unwrap_or(path);
    let endpoint = if path.ends_with("/tokenize") {
        path.to_string()
    } else {
        format!("{path}/tokenize")
    };
    url.set_path(&endpoint);
    Ok(url.into())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    #[rstest]
    #[case::root("https://example.test", "https://example.test/tokenize")]
    #[case::version("https://example.test/v1/", "https://example.test/tokenize")]
    #[case::prefix(
        "https://example.test/tenant/v1",
        "https://example.test/tenant/tokenize"
    )]
    #[case::complete(
        "https://example.test/tenant/tokenize/",
        "https://example.test/tenant/tokenize"
    )]
    #[case::query(
        "https://example.test/tenant/v1?tenant=a",
        "https://example.test/tenant/tokenize?tenant=a"
    )]
    fn endpoint_preserves_the_deployment_prefix(#[case] base: &str, #[case] expected: &str) {
        assert_eq!(tokenize_url(base).unwrap(), expected);
    }

    #[rstest]
    #[case::invalid("not a url")]
    #[case::unsupported("file:///tokenize")]
    fn invalid_endpoint_is_rejected(#[case] base: &str) {
        assert!(tokenize_url(base).is_err());
    }

    #[rstest]
    #[case::text("Hello")]
    #[case::empty("")]
    fn request_preserves_prompt_and_requests_both_representations(#[case] prompt: &str) {
        let request = HostedVllmTokenizeConfig
            .transform_tokenize_request("served-model", prompt, &Map::new(), &no_env)
            .unwrap();
        assert_eq!(
            request,
            json!({"model": "served-model", "prompt": prompt, "add_special_tokens": true, "return_token_strs": true})
        );
    }

    #[rstest]
    fn caller_can_omit_special_tokens() {
        let params = Map::from_iter([("add_special_tokens".into(), json!(false))]);
        let request = HostedVllmTokenizeConfig
            .transform_tokenize_request("served-model", "Hello", &params, &no_env)
            .unwrap();
        assert_eq!(request["add_special_tokens"], false);
    }

    #[rstest]
    #[case::wrong_type(json!({"add_special_tokens": "false"}))]
    #[case::unknown(json!({"unrecognized": true}))]
    fn invalid_parameters_are_rejected(#[case] params: Value) {
        assert!(
            HostedVllmTokenizeConfig
                .transform_tokenize_request("model", "Hello", params.as_object().unwrap(), &no_env)
                .is_err()
        );
    }

    #[rstest]
    #[case::strings(Some(vec!["Hello".to_string(), " world".to_string()]))]
    #[case::without_strings(None)]
    fn response_keeps_ids_and_strings_separate(#[case] strings: Option<Vec<String>>) {
        let response = HostedVllmTokenizeConfig
            .transform_tokenize_response(
                "served-model",
                json!({"count": 2, "tokens": [10, 20], "token_strs": strings, "max_model_len": 1000}),
            )
            .unwrap();
        assert_eq!(response.model, "served-model");
        assert_eq!(response.count, 2);
        assert_eq!(response.token_ids, Some(vec![10, 20]));
        assert_eq!(response.tokens, strings);
    }

    #[rstest]
    #[case::missing_count(json!({"tokens": []}))]
    #[case::negative_count(json!({"count": -1, "tokens": []}))]
    #[case::negative_id(json!({"count": 1, "tokens": [-1]}))]
    #[case::wrong_id_type(json!({"count": 1, "tokens": ["Hello"]}))]
    #[case::missing_ids(json!({"count": 0}))]
    #[case::wrong_count(json!({"count": 2, "tokens": [10]}))]
    #[case::wrong_strings_length(json!({"count": 1, "tokens": [10], "token_strs": []}))]
    fn malformed_responses_are_rejected(#[case] body: Value) {
        assert!(
            HostedVllmTokenizeConfig
                .transform_tokenize_response("model", body)
                .is_err()
        );
    }

    #[rstest]
    fn empty_token_arrays_remain_available() {
        let response = HostedVllmTokenizeConfig
            .transform_tokenize_response(
                "model",
                json!({"count": 0, "tokens": [], "token_strs": []}),
            )
            .unwrap();
        assert_eq!(response.token_ids, Some(vec![]));
        assert_eq!(response.tokens, Some(vec![]));
    }

    #[rstest]
    #[case::explicit(Some("explicit"), Some("environment"), Some("explicit"))]
    #[case::environment(None, Some("environment"), Some("environment"))]
    #[case::keyless(None, None, None)]
    fn key_resolution_supports_authenticated_and_keyless_deployments(
        #[case] explicit: Option<&str>,
        #[case] environment: Option<&str>,
        #[case] expected: Option<&str>,
    ) {
        let validated = HostedVllmTokenizeConfig
            .validate_environment(vec![], explicit, &Map::new(), &|name| {
                (name == "HOSTED_VLLM_API_KEY")
                    .then_some(environment)
                    .flatten()
                    .map(str::to_string)
            })
            .unwrap();
        match (validated.auth, expected) {
            (AuthScheme::Credential { secret, .. }, Some(expected)) => {
                assert_eq!(secret.expose(), expected)
            }
            (AuthScheme::Forwarded, None) => {}
            _ => panic!("unexpected authentication"),
        }
    }

    #[rstest]
    fn forwarded_authorization_is_preserved() {
        let headers = vec![("AUTHORIZATION".into(), "Bearer forwarded".into())];
        let validated = HostedVllmTokenizeConfig
            .validate_environment(headers.clone(), Some("configured"), &Map::new(), &no_env)
            .unwrap();
        assert!(matches!(validated.auth, AuthScheme::Forwarded));
        assert_eq!(validated.headers, headers);
    }

    #[rstest]
    fn api_base_uses_environment_when_omitted() {
        let url = HostedVllmTokenizeConfig
            .get_complete_url(None, "model", &Map::new(), &|name| {
                (name == "HOSTED_VLLM_API_BASE").then(|| "https://env.test/v1".into())
            })
            .unwrap();
        assert_eq!(url, "https://env.test/tokenize");
        assert!(
            HostedVllmTokenizeConfig
                .get_complete_url(None, "model", &Map::new(), &no_env)
                .is_err()
        );
    }
}
