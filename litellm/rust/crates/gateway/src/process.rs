use std::{error::Error, ffi::OsString};

use litellm_config::Environment;
use litellm_tracing::Level;
use serde::{Deserialize, de::DeserializeOwned};

/// Everything the process takes from its environment, read once.
pub(super) struct Process {
    pub environment: Environment,
    pub log_level: Level,
    pub mode: Mode,
}

pub(super) enum Mode {
    /// The Python supervisor set `LITELLM_FORWARD_UPSTREAM` and started the gateway in front of
    /// its proxy. `HOST` and `PORT` are Python's here and are never read.
    InFrontOfPython(litellm_gateway_forward::Settings),
    Standalone(Standalone),
}

#[derive(Deserialize)]
#[serde(default)]
pub(super) struct Standalone {
    pub host: String,
    pub port: u16,
    pub litellm_config: String,
}

impl Default for Standalone {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".into(),
            port: 4000,
            litellm_config: "config.yaml".into(),
        }
    }
}

impl Process {
    pub fn from_env(
        variables: impl IntoIterator<Item = (OsString, OsString)>,
    ) -> Result<Self, Box<dyn Error>> {
        let variables: Vec<(OsString, OsString)> = variables.into_iter().collect();
        let environment: Environment = from_vars(&variables)?;
        let log: LogSettings = from_vars(&variables)?;
        let mode = match litellm_gateway_forward::Settings::from_vars(variables.iter().cloned())? {
            Some(forward) => Mode::InFrontOfPython(forward),
            None => Mode::Standalone(from_vars(&variables)?),
        };
        Ok(Self {
            environment,
            log_level: log.log_level(),
            mode,
        })
    }
}

#[derive(Deserialize)]
struct LogSettings {
    rust_log: Option<String>,
}

impl LogSettings {
    fn log_level(&self) -> Level {
        self.rust_log
            .as_deref()
            .and_then(|value| value.parse().ok())
            .unwrap_or(Level::INFO)
    }
}

fn from_vars<T: DeserializeOwned>(variables: &[(OsString, OsString)]) -> envy::Result<T> {
    envy::from_iter(
        variables.iter().filter_map(|(key, value)| {
            Some((key.to_str()?.to_owned(), value.to_str()?.to_owned()))
        }),
    )
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn os_vars(variables: &[(&str, &str)]) -> Vec<(OsString, OsString)> {
        variables
            .iter()
            .map(|(key, value)| (OsString::from(key), OsString::from(value)))
            .collect()
    }

    fn parse<T: DeserializeOwned>(variables: &[(&str, &str)]) -> envy::Result<T> {
        from_vars(&os_vars(variables))
    }

    #[rstest]
    fn defaults_allow_startup_without_configuration() {
        let settings: Standalone = parse(&[]).unwrap();

        assert_eq!(settings.host, "0.0.0.0");
        assert_eq!(settings.port, 4000);
        assert_eq!(settings.litellm_config, "config.yaml");
    }

    #[rstest]
    fn environment_overrides_startup_settings() {
        let settings: Standalone = parse(&[
            ("HOST", "127.0.0.1"),
            ("PORT", "8080"),
            ("LITELLM_CONFIG", "/tmp/custom.yaml"),
        ])
        .unwrap();

        assert_eq!(settings.host, "127.0.0.1");
        assert_eq!(settings.port, 8080);
        assert_eq!(settings.litellm_config, "/tmp/custom.yaml");
    }

    #[rstest]
    #[case::unset(&[], Level::INFO)]
    #[case::level(&[("RUST_LOG", "debug")], Level::DEBUG)]
    #[case::empty(&[("RUST_LOG", "")], Level::INFO)]
    #[case::filter(&[("RUST_LOG", "litellm=debug")], Level::INFO)]
    #[case::invalid(&[("RUST_LOG", "invalid")], Level::INFO)]
    fn the_log_level_falls_back_to_info(
        #[case] variables: &[(&str, &str)],
        #[case] expected: Level,
    ) {
        let settings: LogSettings = parse(variables).unwrap();

        assert_eq!(settings.log_level(), expected);
    }

    fn process(variables: &[(&str, &str)]) -> Result<Process, Box<dyn Error>> {
        Process::from_env(os_vars(variables))
    }

    #[rstest]
    fn the_forward_upstream_puts_the_gateway_in_front_of_python_without_reading_its_port() {
        let process = process(&[
            ("LITELLM_FORWARD_UPSTREAM", "http://127.0.0.1:53211"),
            ("LITELLM_FORWARD_HOST", "0.0.0.0"),
            ("LITELLM_FORWARD_PORT", "4000"),
            ("LITELLM_FORWARD_UPSTREAM_KEEPALIVE", "2"),
            ("LITELLM_FORWARD_SHUTDOWN_DRAIN", "40"),
            ("LITELLM_FORWARD_CONFIG", ""),
            ("PORT", "pythons-own"),
            ("RUST_LOG", "debug"),
        ])
        .unwrap();

        assert!(matches!(process.mode, Mode::InFrontOfPython(_)));
        assert_eq!(process.log_level, Level::DEBUG);
    }

    #[rstest]
    #[case::words("soon", None)]
    #[case::blank("", None)]
    #[case::number("40", Some(40))]
    fn in_front_of_python_a_timeout_python_tolerates_does_not_fail_startup(
        #[case] raw: &str,
        #[case] expected: Option<u64>,
    ) {
        let process = process(&[
            ("LITELLM_FORWARD_UPSTREAM", "http://127.0.0.1:53211"),
            ("LITELLM_FORWARD_HOST", "0.0.0.0"),
            ("LITELLM_FORWARD_PORT", "4000"),
            ("LITELLM_FORWARD_UPSTREAM_KEEPALIVE", "2"),
            ("LITELLM_FORWARD_SHUTDOWN_DRAIN", "40"),
            ("LITELLM_FORWARD_CONFIG", ""),
            ("NON_LLM_CONNECTION_TIMEOUT", raw),
        ])
        .unwrap();

        assert!(matches!(process.mode, Mode::InFrontOfPython(_)));
        assert_eq!(process.environment.non_llm_connection_timeout, expected);
    }

    #[rstest]
    #[case::invalid_port(&[("PORT", "http")])]
    fn invalid_standalone_settings_fail_startup(#[case] variables: &[(&str, &str)]) {
        assert!(process(variables).is_err());
    }

    #[rstest]
    #[case::empty("")]
    #[case::negative("-1")]
    #[case::overflow("65536")]
    #[case::invalid("http")]
    fn invalid_ports_fail_startup(#[case] value: &str) {
        assert!(parse::<Standalone>(&[("PORT", value)]).is_err());
    }

    #[cfg(unix)]
    #[rstest]
    #[case::unrelated_variable("UNRELATED")]
    #[case::host_fallback("HOST")]
    fn non_unicode_environment_values_are_ignored(#[case] key: &str) {
        use std::os::unix::ffi::OsStringExt;

        let settings: Standalone = from_vars(&[
            (OsString::from(key), OsString::from_vec(vec![0xff])),
            (OsString::from_vec(vec![0xff]), OsString::from("value")),
            (OsString::from("PORT"), OsString::from("8080")),
        ])
        .unwrap();

        assert_eq!(settings.host, "0.0.0.0");
        assert_eq!(settings.port, 8080);
    }
}
