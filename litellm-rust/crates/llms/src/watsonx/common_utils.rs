use std::collections::BTreeMap;

use litellm_auth::{
    CredentialBoundary, CredentialPlacement, InputSource, SecretValue, Sourced,
    watsonx::WatsonxIamConfig,
};
use litellm_core_utils::settings::resolve_non_empty;
use serde_json::Value;

use crate::{
    Error, ErrorDetail,
    base_llm::auth::{AuthScheme, ValidatedEnvironment},
};

pub const API_KEY_ENV: &[&str] = &["WATSONX_APIKEY", "WATSONX_API_KEY", "WX_API_KEY"];

pub struct WatsonxContext<'a> {
    pub api_key: Option<&'a Sourced<SecretValue>>,
    pub api_base: Option<&'a Sourced<String>>,
    pub extra_headers: &'a BTreeMap<String, Sourced<Value>>,
    pub deployment_params: &'a BTreeMap<String, Sourced<Value>>,
    pub optional_params: &'a BTreeMap<String, Sourced<Value>>,
}

pub fn resolve_environment(
    input: &WatsonxContext<'_>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<ValidatedEnvironment, Error> {
    let headers = input
        .extra_headers
        .iter()
        .map(|(name, value)| {
            let value =
                serde_json::from_value::<String>(value.value().clone()).map_err(|source| {
                    Error::InvalidRequest(ErrorDetail::invalid("extra_headers", source))
                })?;
            Ok((name.clone(), value))
        })
        .collect::<Result<_, Error>>()?;
    let (auth, source) = select_auth(input, env_lookup)?;
    let sources: Vec<InputSource> = input
        .api_key
        .into_iter()
        .map(Sourced::source)
        .chain(
            input
                .extra_headers
                .iter()
                .filter(|(name, _)| name.eq_ignore_ascii_case("authorization"))
                .map(|(_, value)| value.source()),
        )
        .chain(
            input
                .deployment_params
                .iter()
                .chain(input.optional_params)
                .filter(|(name, _)| matches!(name.as_str(), "token" | "zen_api_key"))
                .map(|(_, value)| value.source()),
        )
        .chain([source])
        .collect();
    let boundary = CredentialBoundary::new(&sources)
        .map_err(|error| Error::InvalidRequest(error.to_string().into()))?;
    boundary
        .validate_resources(
            input.api_base.into_iter().map(Sourced::source).chain(
                input
                    .deployment_params
                    .iter()
                    .chain(input.optional_params)
                    .filter(|(name, _)| {
                        matches!(
                            name.as_str(),
                            "project_id" | "watsonx_project" | "space_id" | "api_version"
                        )
                    })
                    .map(|(_, value)| value.source()),
            ),
        )
        .map_err(|error| Error::InvalidRequest(error.to_string().into()))?;
    Ok(ValidatedEnvironment { headers, auth })
}

fn select_auth(
    input: &WatsonxContext<'_>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<(AuthScheme, InputSource), Error> {
    let token = credential_param(input, "token")?;
    let zen = credential_param(input, "zen_api_key")?;
    if let Some((_, authorization)) = input
        .extra_headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("authorization"))
    {
        return Ok((AuthScheme::Forwarded, authorization.source()));
    }
    if let Some(token) = resolve_credential(token.as_ref(), env_lookup, &["WATSONX_TOKEN"]) {
        return Ok((
            AuthScheme::Credential {
                placement: CredentialPlacement::Bearer,
                secret: token.value().clone(),
            },
            token.source(),
        ));
    }
    if let Some(key) = resolve_credential(zen.as_ref(), env_lookup, &["WATSONX_ZENAPIKEY"]) {
        return Ok((
            AuthScheme::Credential {
                placement: CredentialPlacement::Header("Authorization"),
                secret: SecretValue::new(format!("ZenApiKey {}", key.value().expose())),
            },
            key.source(),
        ));
    }
    let key = resolve_credential(input.api_key, env_lookup, API_KEY_ENV).ok_or(
        litellm_auth::Error::MissingApiKey {
            provider: "watsonx",
            environment_variable: "WATSONX_API_KEY",
        },
    )?;
    let source = key.source();
    let config = match resolve_non_empty(None, env_lookup, &["WATSONX_IAM_URL"]) {
        Some(endpoint) => WatsonxIamConfig::with_endpoint(key.into_value(), endpoint),
        None => WatsonxIamConfig::new(key.into_value()),
    };
    Ok((
        AuthScheme::WatsonxIam {
            config: Box::new(config),
        },
        source,
    ))
}

fn credential_param(
    input: &WatsonxContext<'_>,
    name: &'static str,
) -> Result<Option<Sourced<SecretValue>>, Error> {
    let deployment = input
        .deployment_params
        .get(name)
        .map(|value| {
            serde_json::from_value::<Option<String>>(value.value().clone())
                .map(|secret| {
                    secret.map(|secret| Sourced::new(SecretValue::new(secret), value.source()))
                })
                .map_err(|source| Error::InvalidRequest(ErrorDetail::invalid(name, source)))
        })
        .transpose()?;
    let optional = input
        .optional_params
        .get(name)
        .map(|value| {
            serde_json::from_value::<Option<String>>(value.value().clone())
                .map(|secret| {
                    secret.map(|secret| Sourced::new(SecretValue::new(secret), value.source()))
                })
                .map_err(|source| Error::InvalidRequest(ErrorDetail::invalid(name, source)))
        })
        .transpose()?;
    Ok(optional.or(deployment).flatten())
}

fn resolve_credential(
    value: Option<&Sourced<SecretValue>>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
    names: &[&str],
) -> Option<Sourced<SecretValue>> {
    let explicit = value.filter(|value| !value.value().expose().trim().is_empty());
    resolve_non_empty(
        explicit.map(|value| value.value().expose()),
        env_lookup,
        names,
    )
    .map(|secret| {
        Sourced::new(
            SecretValue::new(secret),
            explicit.map_or(InputSource::Environment, Sourced::source),
        )
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case::bearer("token", "bearer", CredentialPlacement::Bearer, "bearer")]
    #[case::zen(
        "zen_api_key",
        "zen",
        CredentialPlacement::Header("Authorization"),
        "ZenApiKey zen"
    )]
    fn static_credentials_precede_iam(
        #[case] name: &str,
        #[case] value: &str,
        #[case] placement: CredentialPlacement,
        #[case] expected: &str,
    ) {
        let key = Sourced::new(SecretValue::new("iam"), InputSource::Request);
        let params = BTreeMap::from([(
            name.into(),
            Sourced::new(json!(value), InputSource::Request),
        )]);
        let empty = BTreeMap::new();
        let input = WatsonxContext {
            api_key: Some(&key),
            api_base: None,
            extra_headers: &empty,
            deployment_params: &empty,
            optional_params: &params,
        };
        let environment = resolve_environment(&input, &|_| None).unwrap();
        let AuthScheme::Credential {
            placement: actual,
            secret,
        } = environment.auth
        else {
            panic!("expected static credential")
        };
        assert_eq!(actual, placement);
        assert_eq!(secret.expose(), expected);
    }

    #[rstest]
    #[case::request(InputSource::Request, true)]
    #[case::deployment(InputSource::Deployment, false)]
    #[case::environment(InputSource::Environment, false)]
    fn resource_provenance_is_checked_for_iam(#[case] source: InputSource, #[case] allowed: bool) {
        let key = Sourced::new(SecretValue::new("iam"), source);
        let base = Sourced::new("https://request.test".into(), InputSource::Request);
        let empty = BTreeMap::new();
        let input = WatsonxContext {
            api_key: Some(&key),
            api_base: Some(&base),
            extra_headers: &empty,
            deployment_params: &empty,
            optional_params: &empty,
        };
        let result = resolve_environment(&input, &|_| None);
        if allowed {
            assert!(matches!(
                result.unwrap().auth,
                AuthScheme::WatsonxIam { .. }
            ));
        } else {
            assert!(matches!(result, Err(Error::InvalidRequest(_))));
        }
    }

    #[rstest]
    #[case::project("project_id")]
    #[case::project_alias("watsonx_project")]
    #[case::space("space_id")]
    #[case::version("api_version")]
    fn environment_credentials_reject_request_resources(#[case] name: &str) {
        let params = BTreeMap::from([(
            name.into(),
            Sourced::new(json!("request"), InputSource::Request),
        )]);
        let empty = BTreeMap::new();
        let input = WatsonxContext {
            api_key: None,
            api_base: None,
            extra_headers: &empty,
            deployment_params: &empty,
            optional_params: &params,
        };
        assert!(matches!(
            resolve_environment(&input, &|name| (name == "WATSONX_TOKEN")
                .then(|| "host-token".into())),
            Err(Error::InvalidRequest(_))
        ));
    }

    #[rstest]
    #[case::blank(serde_json::json!(" "))]
    #[case::null(serde_json::Value::Null)]
    fn request_credential_cannot_fall_back_to_environment(#[case] value: Value) {
        let params = BTreeMap::from([("token".into(), Sourced::new(value, InputSource::Request))]);
        let empty = BTreeMap::new();
        let input = WatsonxContext {
            api_key: None,
            api_base: None,
            extra_headers: &empty,
            deployment_params: &empty,
            optional_params: &params,
        };
        assert!(matches!(
            resolve_environment(&input, &|name| (name == "WATSONX_TOKEN")
                .then(|| "host-token".into())),
            Err(Error::InvalidRequest(_))
        ));
    }
}
