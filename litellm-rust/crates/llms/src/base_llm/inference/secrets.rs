use std::sync::Arc;

use futures_util::future::BoxFuture;
use litellm_core_utils::settings::{Lookup, ProcessEnvironment};
use litellm_secrets::Error;

pub type Secrets = Arc<dyn Lookup + Send + Sync>;

pub trait SecretSource: Send + Sync {
    fn resolve<'a>(&'a self, names: &'a [&'static str]) -> BoxFuture<'a, Result<Secrets, Error>>;
}

pub struct EnvironmentSecrets;

impl SecretSource for EnvironmentSecrets {
    fn resolve<'a>(&'a self, _names: &'a [&'static str]) -> BoxFuture<'a, Result<Secrets, Error>> {
        Box::pin(async { Ok(Arc::new(ProcessEnvironment) as Secrets) })
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::{EnvironmentSecrets, SecretSource};

    #[rstest]
    #[case::lowercase_true("LITELLM_ENVIRONMENT_SECRETS_TRUE", "true", None)]
    #[case::padded_false("LITELLM_ENVIRONMENT_SECRETS_FALSE", " FALSE ", None)]
    #[case::text("LITELLM_ENVIRONMENT_SECRETS_TEXT", "secret", Some("secret"))]
    #[tokio::test]
    async fn boolean_environment_values_are_absent_like_get_secret_str(
        #[case] name: &'static str,
        #[case] value: &str,
        #[case] expected: Option<&str>,
    ) {
        unsafe { std::env::set_var(name, value) };
        let secret = EnvironmentSecrets.resolve(&[name]).await.unwrap().get(name);
        unsafe { std::env::remove_var(name) };
        assert_eq!(secret.as_deref(), expected);
    }
}
