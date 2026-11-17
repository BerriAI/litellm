use litellm_python_compat::number;
use litellm_secret_value::SecretValue;
use serde::{Deserialize, Deserializer};

/// The environment variables the proxy reads outside its config file, as written. Each field is
/// its variable's name in lower case.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct Environment {
    pub litellm_license: Option<SecretValue>,
    #[serde(deserialize_with = "int_or_unset")]
    pub non_llm_connection_timeout: Option<u64>,
}

/// `get_env_int_or_none`, which reads a blank or unparsable value as unset. A negative one is
/// unset here too, where Python keeps it.
fn int_or_unset<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<u64>, D::Error> {
    let raw = String::deserialize(deserializer)?;
    Ok(number::int(&raw).and_then(|value| u64::try_from(value).ok()))
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
    #[case::blank("", None)]
    #[case::whitespace("  ", None)]
    #[case::words("soon", None)]
    #[case::fraction("15.5", None)]
    #[case::negative("-1", None)]
    #[case::padded(" 20 ", Some(20))]
    #[case::grouped_digits("1_000", Some(1000))]
    fn a_timeout_is_read_the_way_python_reads_an_integer(
        #[case] raw: &str,
        #[case] expected: Option<u64>,
    ) {
        let environment = from_variables(&[("NON_LLM_CONNECTION_TIMEOUT", raw)]).unwrap();

        assert_eq!(environment.non_llm_connection_timeout, expected);
    }
}
