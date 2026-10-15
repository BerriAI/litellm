//! Hosted vLLM Responses, matching `litellm/llms/hosted_vllm/responses/transformation.py`

use litellm_auth::{CredentialPlacement, SecretValue};
use litellm_llms_types::formats::responses::ResponsesApiResponse;
use serde_json::{Map, Value};

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

const API_KEY_ENV: &str = "HOSTED_VLLM_API_KEY";
const API_BASE_ENV: &str = "HOSTED_VLLM_API_BASE";
/// vLLM does not require authentication by default
const PLACEHOLDER_API_KEY: &str = "fake-api-key";

pub struct HostedVllmResponsesApiConfig;
pub const HOSTED_VLLM_RESPONSES_CONFIG: HostedVllmResponsesApiConfig = HostedVllmResponsesApiConfig;

impl BaseResponsesApiConfig for HostedVllmResponsesApiConfig {
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
            resolved(api_key, lookup, API_KEY_ENV).unwrap_or_else(|| PLACEHOLDER_API_KEY.into());
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
        let api_base = resolved(api_base, lookup, API_BASE_ENV).ok_or(
            litellm_auth::Error::MissingApiBase {
                provider: "Hosted VLLM",
                guidance: "Set `api_base` or the HOSTED_VLLM_API_BASE environment variable",
            },
        )?;
        let api_base = api_base.trim_end_matches('/');
        Ok(match api_base.ends_with("/v1") {
            true => format!("{api_base}/responses"),
            false => format!("{api_base}/v1/responses"),
        })
    }

    fn transform_responses_api_request(
        &self,
        model: &str,
        input: Value,
        params: Map<String, Value>,
    ) -> Result<Value, Error> {
        responses_request(model, input, params, ResponsesHost::Compatible)
    }

    fn transform_response_api_response(&self, body: Value) -> Result<ResponsesApiResponse, Error> {
        responses_response(body)
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
    #[case::bare_host(Some("http://vllm:8000"), &[], "http://vllm:8000/v1/responses")]
    #[case::v1_suffix(Some("http://vllm:8000/v1"), &[], "http://vllm:8000/v1/responses")]
    #[case::trailing_slashes(Some("http://vllm:8000/v1//"), &[], "http://vllm:8000/v1/responses")]
    #[case::secret(None, &[("HOSTED_VLLM_API_BASE", "http://secret:8000")], "http://secret:8000/v1/responses")]
    fn url_appends_the_v1_responses_route(
        #[case] api_base: Option<&str>,
        #[case] secrets: &'static [(&'static str, &'static str)],
        #[case] expected: &str,
    ) {
        assert_eq!(
            HOSTED_VLLM_RESPONSES_CONFIG
                .get_complete_url(api_base, None, &env(secrets))
                .unwrap(),
            expected
        );
    }

    #[test]
    fn url_needs_an_api_base() {
        assert!(matches!(
            HOSTED_VLLM_RESPONSES_CONFIG.get_complete_url(None, None, &env(&[])),
            Err(Error::Auth(litellm_auth::Error::MissingApiBase { .. }))
        ));
    }

    #[rstest]
    #[case::explicit(Some("explicit"), &[("HOSTED_VLLM_API_KEY", "managed")], "explicit")]
    #[case::secret(None, &[("HOSTED_VLLM_API_KEY", "managed")], "managed")]
    #[case::placeholder(None, &[], "fake-api-key")]
    #[case::blank_explicit(Some(""), &[], "fake-api-key")]
    fn bearer_falls_back_to_the_placeholder_key(
        #[case] api_key: Option<&str>,
        #[case] secrets: &'static [(&'static str, &'static str)],
        #[case] expected: &str,
    ) {
        let environment = HOSTED_VLLM_RESPONSES_CONFIG
            .validate_environment(vec![], api_key, &env(secrets))
            .unwrap();
        let AuthScheme::Credential { placement, secret } = environment.auth else {
            panic!("expected a credential");
        };
        assert_eq!(placement, CredentialPlacement::Bearer);
        assert_eq!(secret.expose(), expected);
    }
}
