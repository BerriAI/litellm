//! Endpoint and credential resolution every OpenAI API config shares, mirroring the
//! `api_base` / `api_key` fallbacks in `litellm/llms/openai/openai.py`.

use litellm_auth::{CredentialPlacement, SecretValue};

use crate::{
    Error,
    base_llm::auth::{AuthScheme, Headers, ValidatedEnvironment},
};

pub const OPENAI_DEFAULT_API_BASE: &str = "https://api.openai.com/v1";
pub const OPENAI_API_KEY_ENV: &str = "OPENAI_API_KEY";
const OPENAI_API_BASE_ENVS: [&str; 2] = ["OPENAI_BASE_URL", "OPENAI_API_BASE"];

/// The secrets a call has to resolve: only the ones its deployment leaves unset.
pub fn secret_names(api_key: Option<&str>, api_base: Option<&str>) -> &'static [&'static str] {
    match (
        api_key.is_some_and(|key| !key.is_empty()),
        api_base.is_some_and(|base| !base.is_empty()),
    ) {
        (true, true) => &[],
        (true, false) => &["OPENAI_BASE_URL", "OPENAI_API_BASE"],
        (false, true) => &["OPENAI_API_KEY"],
        (false, false) => &["OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE"],
    }
}

/// The deployment's `api_base`, else `OPENAI_BASE_URL` or `OPENAI_API_BASE`, else OpenAI's
/// own endpoint, with `path` appended.
pub fn complete_url(
    api_base: Option<&str>,
    path: &str,
    lookup: &dyn Fn(&str) -> Option<String>,
) -> String {
    let base = api_base
        .filter(|base| !base.is_empty())
        .map(str::to_owned)
        .or_else(|| OPENAI_API_BASE_ENVS.into_iter().find_map(lookup))
        .unwrap_or_else(|| OPENAI_DEFAULT_API_BASE.into());
    format!("{}{path}", base.trim_end_matches('/'))
}

/// The deployment's key, else `OPENAI_API_KEY`, sent as a bearer token. The configured key
/// replaces any `authorization` the caller forwarded.
pub fn bearer_environment(
    headers: Headers,
    api_key: Option<&str>,
    lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<ValidatedEnvironment, Error> {
    let key = api_key
        .filter(|key| !key.is_empty())
        .map(str::to_owned)
        .or_else(|| lookup(OPENAI_API_KEY_ENV))
        .filter(|key| !key.is_empty())
        .ok_or(litellm_auth::Error::MissingApiKey {
            provider: "OpenAI",
            environment_variable: OPENAI_API_KEY_ENV,
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

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn env(name: &str) -> Option<String> {
        match name {
            "OPENAI_API_BASE" => Some("http://env-base/v1/".into()),
            "OPENAI_API_KEY" => Some("sk-env".into()),
            _ => None,
        }
    }

    fn no_env(_: &str) -> Option<String> {
        None
    }

    #[rstest]
    #[case::default(None, no_env, "https://api.openai.com/v1/chat/completions")]
    #[case::empty_deployment_base(Some(""), no_env, "https://api.openai.com/v1/chat/completions")]
    #[case::environment(None, env, "http://env-base/v1/chat/completions")]
    #[case::deployment_wins(
        Some("http://mock/d/1/v1/"),
        env,
        "http://mock/d/1/v1/chat/completions"
    )]
    fn the_url_prefers_the_deployment_then_the_environment(
        #[case] api_base: Option<&str>,
        #[case] lookup: fn(&str) -> Option<String>,
        #[case] expected: &str,
    ) {
        assert_eq!(
            complete_url(api_base, "/chat/completions", &lookup),
            expected
        );
    }

    #[rstest]
    #[case::deployment(Some("sk-deployment"), no_env, "sk-deployment")]
    #[case::deployment_wins(Some("sk-deployment"), env, "sk-deployment")]
    #[case::environment(None, env, "sk-env")]
    #[case::empty_deployment_key(Some(""), env, "sk-env")]
    fn the_key_prefers_the_deployment_then_the_environment(
        #[case] api_key: Option<&str>,
        #[case] lookup: fn(&str) -> Option<String>,
        #[case] expected: &str,
    ) {
        let validated = bearer_environment(Vec::new(), api_key, &lookup).unwrap();
        assert!(matches!(
            validated.auth,
            AuthScheme::Credential { placement: CredentialPlacement::Bearer, secret }
                if secret.expose() == expected
        ));
    }

    #[rstest]
    fn a_missing_key_names_the_environment_variable() {
        assert_eq!(
            bearer_environment(Vec::new(), None, &no_env).unwrap_err(),
            Error::Auth(litellm_auth::Error::MissingApiKey {
                provider: "OpenAI",
                environment_variable: "OPENAI_API_KEY",
            })
        );
    }
}
