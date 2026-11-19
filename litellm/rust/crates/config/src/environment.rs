use litellm_python_compat::lenient::UintOrUnset;
use serde::Deserialize;
use serde_with::serde_as;

/// The environment variables the proxy reads outside its config file, as written. Each field is
/// its variable's name in lower case.
#[serde_as]
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct Environment {
    #[serde_as(as = "UintOrUnset")]
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
        let environment = from_variables(&[("NON_LLM_CONNECTION_TIMEOUT", "40")]).unwrap();

        assert_eq!(
            environment,
            Environment {
                non_llm_connection_timeout: Some(40),
            }
        );
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
