use litellm_auth::{CredentialPlacement, SecretValue};
use litellm_core_utils::settings::resolve_non_empty;
use litellm_llms_types::formats::responses::ResponsesApiResponse;
use serde_json::{Map, Value};

use crate::{
    Error,
    base_llm::{
        auth::{AuthScheme, ValidatedEnvironment},
        responses::transformation::BaseResponsesApiConfig,
    },
    openai::responses::transformation::{responses_request, responses_response},
};

const API_KEY_ENVS: [&str; 2] = ["OPENROUTER_API_KEY", "OR_API_KEY"];
const API_BASE_ENV: &str = "OPENROUTER_API_BASE";
const DEFAULT_BASE: &str = "https://openrouter.ai/api/v1";
const RESPONSES_PATH: &str = "/responses";

pub struct OpenRouterResponsesApiConfig;
pub const OPENROUTER_RESPONSES_CONFIG: OpenRouterResponsesApiConfig = OpenRouterResponsesApiConfig;

impl BaseResponsesApiConfig for OpenRouterResponsesApiConfig {
    fn secret_names(
        &self,
        api_key: Option<&str>,
        api_base: Option<&str>,
    ) -> &'static [&'static str] {
        match (
            api_key.is_some_and(|key| !key.trim().is_empty()),
            api_base.is_some_and(|base| !base.trim().is_empty()),
        ) {
            (true, true) => &[],
            (true, false) => &[API_BASE_ENV],
            (false, true) => &API_KEY_ENVS,
            (false, false) => &["OPENROUTER_API_KEY", "OR_API_KEY", API_BASE_ENV],
        }
    }

    fn validate_environment(
        &self,
        headers: Vec<(String, String)>,
        api_key: Option<&str>,
        lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        let key = resolve_non_empty(api_key, lookup, &API_KEY_ENVS).ok_or(
            litellm_auth::Error::MissingApiKey {
                provider: "OpenRouter",
                environment_variable: API_KEY_ENVS[0],
            },
        )?;
        Ok(ValidatedEnvironment {
            headers: litellm_http::request::with_default_headers(
                headers,
                &[("content-type", "application/json")],
            ),
            auth: AuthScheme::Credential {
                placement: CredentialPlacement::Bearer,
                secret: SecretValue::new(key),
            },
        })
    }

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        lookup: &dyn Fn(&str) -> Option<String>,
    ) -> String {
        let base = resolve_non_empty(api_base, lookup, &[API_BASE_ENV])
            .unwrap_or_else(|| DEFAULT_BASE.into());
        format!("{}{RESPONSES_PATH}", base.trim_end_matches('/'))
    }

    fn transform_responses_api_request(
        &self,
        model: &str,
        input: Value,
        params: Map<String, Value>,
    ) -> Result<Value, Error> {
        responses_request(model, input, params)
    }

    fn transform_response_api_response(&self, body: Value) -> Result<ResponsesApiResponse, Error> {
        responses_response(body)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn env(values: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |name| {
            values
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| (*value).to_owned())
        }
    }

    #[rstest]
    #[case::default(None, &[], "https://openrouter.ai/api/v1/responses")]
    #[case::explicit(Some("https://proxy.test/api/v1"), &[], "https://proxy.test/api/v1/responses")]
    #[case::trailing_slash(Some("https://proxy.test/api/v1/"), &[], "https://proxy.test/api/v1/responses")]
    #[case::blank_falls_to_secret(Some(" "), &[("OPENROUTER_API_BASE", "https://secret.test/v1")], "https://secret.test/v1/responses")]
    #[case::openai_base_is_ignored(None, &[("OPENAI_BASE_URL", "https://openai.test/v1")], "https://openrouter.ai/api/v1/responses")]
    fn url_is_the_base_plus_responses(
        #[case] api_base: Option<&str>,
        #[case] secrets: &'static [(&'static str, &'static str)],
        #[case] expected: &str,
    ) {
        assert_eq!(
            OPENROUTER_RESPONSES_CONFIG.get_complete_url(api_base, &env(secrets)),
            expected
        );
    }

    #[rstest]
    #[case::explicit(Some("explicit"), &[("OPENROUTER_API_KEY", "managed")], "explicit")]
    #[case::managed(None, &[("OPENROUTER_API_KEY", "managed"), ("OR_API_KEY", "legacy")], "managed")]
    #[case::legacy_name(None, &[("OR_API_KEY", "legacy")], "legacy")]
    #[case::blank_explicit(Some(" "), &[("OR_API_KEY", "legacy")], "legacy")]
    fn bearer_comes_from_the_call_then_either_secret_name(
        #[case] api_key: Option<&str>,
        #[case] secrets: &'static [(&'static str, &'static str)],
        #[case] expected: &str,
    ) {
        let environment = OPENROUTER_RESPONSES_CONFIG
            .validate_environment(vec![], api_key, &env(secrets))
            .unwrap();
        let AuthScheme::Credential { placement, secret } = environment.auth else {
            panic!("expected a credential");
        };
        assert_eq!(placement, CredentialPlacement::Bearer);
        assert_eq!(secret.expose(), expected);
        assert_eq!(
            environment.headers,
            vec![("content-type".to_owned(), "application/json".to_owned())]
        );
    }

    #[rstest]
    #[case::nothing(&[])]
    #[case::openai_key_only(&[("OPENAI_API_KEY", "openai")])]
    fn a_missing_key_names_the_primary_variable(
        #[case] secrets: &'static [(&'static str, &'static str)],
    ) {
        assert!(matches!(
            OPENROUTER_RESPONSES_CONFIG.validate_environment(vec![], None, &env(secrets)),
            Err(Error::Auth(litellm_auth::Error::MissingApiKey {
                provider: "OpenRouter",
                environment_variable: "OPENROUTER_API_KEY",
            }))
        ));
    }

    #[rstest]
    #[case::both_given(Some("key"), Some("base"), &[])]
    #[case::key_given(Some("key"), None, &["OPENROUTER_API_BASE"])]
    #[case::base_given(None, Some("base"), &["OPENROUTER_API_KEY", "OR_API_KEY"])]
    #[case::neither(None, None, &["OPENROUTER_API_KEY", "OR_API_KEY", "OPENROUTER_API_BASE"])]
    fn only_missing_settings_are_looked_up(
        #[case] api_key: Option<&str>,
        #[case] api_base: Option<&str>,
        #[case] expected: &[&str],
    ) {
        assert_eq!(
            OPENROUTER_RESPONSES_CONFIG.secret_names(api_key, api_base),
            expected
        );
    }
}
