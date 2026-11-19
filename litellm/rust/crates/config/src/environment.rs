use litellm_secret_value::SecretValue;
use serde::Deserialize;

/// The environment variables the proxy reads outside its config file, as written. Each field is
/// its variable's name in lower case.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct Environment {
    pub litellm_license: Option<SecretValue>,
    pub non_llm_connection_timeout: Option<u64>,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn from_variables(variables: &[(&str, &str)]) -> envy::Result<Environment> {
        envy::from_iter(
            variables
                .iter()
                .map(|(key, value)| (key.to_string(), value.to_string())),
        )
    }

    #[rstest]
    fn the_proxy_environment_is_read_by_variable_name() {
        let environment = from_variables(&[
            ("LITELLM_LICENSE", "license-key"),
            ("NON_LLM_CONNECTION_TIMEOUT", "40"),
        ])
        .unwrap();

        assert_eq!(
            environment,
            Environment {
                litellm_license: Some("license-key".into()),
                non_llm_connection_timeout: Some(40),
            }
        );
        assert!(!format!("{environment:?}").contains("license-key"));
    }

    #[rstest]
    fn an_empty_environment_leaves_every_proxy_setting_unset() {
        assert_eq!(from_variables(&[]).unwrap(), Environment::default());
    }

    #[rstest]
    fn a_timeout_that_is_not_a_number_fails_startup() {
        assert!(from_variables(&[("NON_LLM_CONNECTION_TIMEOUT", "soon")]).is_err());
    }
}
