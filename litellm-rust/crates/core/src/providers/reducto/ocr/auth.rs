use crate::error::{AuthError, Error};

pub const REDUCTO_API_KEY_ENV: &str = "REDUCTO_API_KEY";

const MISSING_KEY_MESSAGE: &str = "Missing REDUCTO_API_KEY - set it in the environment or pass api_key to litellm.ocr()/litellm.aocr()";

pub fn resolve_api_key(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    api_key
        .map(str::trim)
        .filter(|key| !key.is_empty())
        .map(str::to_string)
        .or_else(|| {
            env_lookup(REDUCTO_API_KEY_ENV)
                .map(|key| key.trim().to_string())
                .filter(|key| !key.is_empty())
        })
        .ok_or_else(|| Error::Auth(AuthError::Message(MISSING_KEY_MESSAGE.to_string())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_api_key_prefers_explicit_value() {
        let key = resolve_api_key(Some("passed-key"), &|_| Some("env-key".to_string()))
            .expect("explicit key resolves");

        assert_eq!(key, "passed-key");
    }

    #[test]
    fn resolve_api_key_uses_environment_fallback() {
        let env_lookup =
            |name: &str| (name == REDUCTO_API_KEY_ENV).then(|| " env-key ".to_string());

        assert_eq!(resolve_api_key(None, &env_lookup).unwrap(), "env-key");
    }
}
