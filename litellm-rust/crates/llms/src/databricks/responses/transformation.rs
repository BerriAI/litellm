//! Databricks Responses for its OpenAI GPT serving endpoints, matching
//! `litellm/llms/databricks/responses/transformation.py`

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

const API_KEY_ENV: &str = "DATABRICKS_API_KEY";
const API_BASE_ENV: &str = "DATABRICKS_API_BASE";

pub struct DatabricksResponsesApiConfig;
pub const DATABRICKS_RESPONSES_CONFIG: DatabricksResponsesApiConfig = DatabricksResponsesApiConfig;

impl BaseResponsesApiConfig for DatabricksResponsesApiConfig {
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
            resolved(api_key, lookup, API_KEY_ENV).ok_or(litellm_auth::Error::MissingApiKey {
                provider: "Databricks",
                environment_variable: API_KEY_ENV,
            })?;
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
        _api_version: Option<&str>,
        lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        let api_base = resolved(api_base, lookup, API_BASE_ENV).ok_or(
            litellm_auth::Error::MissingApiBase {
                provider: "Databricks",
                guidance: "Set `api_base` or the DATABRICKS_API_BASE environment variable",
            },
        )?;
        Ok(format!("{}/responses", api_base.trim_end_matches('/')))
    }

    fn transform_responses_api_request(
        &self,
        model: &str,
        input: Value,
        params: Map<String, Value>,
    ) -> Result<Value, Error> {
        let model = model.strip_prefix("databricks/").unwrap_or(model);
        responses_request(model, input, params, ResponsesHost::Compatible)
    }

    fn transform_response_api_response(&self, body: Value) -> Result<ResponsesApiResponse, Error> {
        responses_response(body)
    }
}

/// Python's `value or os.getenv(name)`: an empty value falls through, and nothing is trimmed
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
    #[case::explicit(Some("https://w.cloud.databricks.com/serving-endpoints/"), &[], "https://w.cloud.databricks.com/serving-endpoints/responses")]
    #[case::secret(None, &[("DATABRICKS_API_BASE", "https://s.cloud.databricks.com/serving-endpoints")], "https://s.cloud.databricks.com/serving-endpoints/responses")]
    fn url_is_the_serving_endpoints_base_plus_responses(
        #[case] api_base: Option<&str>,
        #[case] secrets: &'static [(&'static str, &'static str)],
        #[case] expected: &str,
    ) {
        assert_eq!(
            DATABRICKS_RESPONSES_CONFIG
                .get_complete_url(api_base, None, &env(secrets))
                .unwrap(),
            expected
        );
    }

    #[test]
    fn url_needs_an_api_base() {
        assert!(matches!(
            DATABRICKS_RESPONSES_CONFIG.get_complete_url(None, None, &env(&[])),
            Err(Error::Auth(litellm_auth::Error::MissingApiBase { .. }))
        ));
    }

    #[rstest]
    #[case::explicit(Some("dapi-explicit"), &[("DATABRICKS_API_KEY", "dapi-env")], "dapi-explicit")]
    #[case::secret(None, &[("DATABRICKS_API_KEY", "dapi-env")], "dapi-env")]
    fn bearer_is_the_personal_access_token(
        #[case] api_key: Option<&str>,
        #[case] secrets: &'static [(&'static str, &'static str)],
        #[case] expected: &str,
    ) {
        let environment = DATABRICKS_RESPONSES_CONFIG
            .validate_environment(vec![], api_key, &env(secrets))
            .unwrap();
        let AuthScheme::Credential { placement, secret } = environment.auth else {
            panic!("expected a credential");
        };
        assert_eq!(placement, CredentialPlacement::Bearer);
        assert_eq!(secret.expose(), expected);
    }

    #[test]
    fn a_repeated_provider_prefix_is_stripped() {
        let body = DATABRICKS_RESPONSES_CONFIG
            .transform_responses_api_request(
                "databricks/databricks-gpt-5-nano",
                json!("hi"),
                Map::new(),
            )
            .unwrap();
        assert_eq!(body["model"], json!("databricks-gpt-5-nano"));
    }
}
