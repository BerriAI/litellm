use crate::error::{AuthError, Error};

pub const MISTRAL_API_KEY_ENV: &str = "MISTRAL_API_KEY";

pub fn resolve_api_key(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    api_key
        .map(str::trim)
        .filter(|key| !key.is_empty())
        .map(str::to_string)
        .or_else(|| env_lookup(MISTRAL_API_KEY_ENV).filter(|key| !key.trim().is_empty()))
        .ok_or(Error::Auth(AuthError::MissingApiKey {
            provider: "Mistral",
        }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_api_key_prefers_param_then_env() {
        let no_env = |_: &str| None;
        assert_eq!(
            resolve_api_key(Some("sk-param"), &no_env).unwrap(),
            "sk-param"
        );

        let with_env = |key: &str| (key == MISTRAL_API_KEY_ENV).then(|| "sk-env".to_string());
        assert_eq!(resolve_api_key(None, &with_env).unwrap(), "sk-env");
        assert_eq!(resolve_api_key(Some("  "), &with_env).unwrap(), "sk-env");
    }

    #[test]
    fn resolve_api_key_errors_when_absent() {
        let err = resolve_api_key(None, &|_| None).expect_err("missing key should error");
        assert_eq!(
            err,
            Error::Auth(AuthError::MissingApiKey {
                provider: "Mistral",
            })
        );
    }
}
