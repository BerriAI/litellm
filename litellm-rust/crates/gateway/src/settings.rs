use std::ffi::OsString;

use litellm_auth_types::SecretValue;
use litellm_tracing::Level;
use serde::{Deserialize, de::DeserializeOwned};

#[derive(Deserialize)]
#[serde(default)]
pub(super) struct Settings {
    pub host: String,
    pub port: u16,
    pub litellm_config: String,
    rust_log: Option<String>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".into(),
            port: 4000,
            litellm_config: "config.yaml".into(),
            rust_log: None,
        }
    }
}

impl Settings {
    pub fn log_level(&self) -> Level {
        self.rust_log
            .as_deref()
            .and_then(|value| value.parse().ok())
            .unwrap_or(Level::INFO)
    }
}

#[derive(Deserialize)]
pub(super) struct UiSettings {
    #[serde(default = "default_username")]
    pub ui_username: String,
    pub ui_password: SecretValue,
    #[serde(default = "default_secure_cookies")]
    pub litellm_ui_secure_cookies: bool,
}

fn default_username() -> String {
    "admin".into()
}

fn default_secure_cookies() -> bool {
    true
}

pub(super) fn from_iter<T: DeserializeOwned>(
    variables: impl IntoIterator<Item = (OsString, OsString)>,
) -> envy::Result<T> {
    envy::from_iter(
        variables
            .into_iter()
            .filter_map(|(key, value)| Some((key.into_string().ok()?, value.into_string().ok()?))),
    )
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn parse<T: DeserializeOwned>(variables: &[(&str, &str)]) -> envy::Result<T> {
        from_iter(
            variables
                .iter()
                .map(|(key, value)| (OsString::from(key), OsString::from(value))),
        )
    }

    #[rstest]
    fn defaults_allow_startup_without_ui_credentials() {
        let settings: Settings = parse(&[]).unwrap();

        assert_eq!(settings.host, "0.0.0.0");
        assert_eq!(settings.port, 4000);
        assert_eq!(settings.litellm_config, "config.yaml");
        assert_eq!(settings.log_level(), Level::INFO);
    }

    #[rstest]
    fn environment_overrides_startup_settings() {
        let settings: Settings = parse(&[
            ("HOST", "127.0.0.1"),
            ("PORT", "8080"),
            ("LITELLM_CONFIG", "/tmp/custom.yaml"),
            ("RUST_LOG", "debug"),
            ("LITELLM_UI_SECURE_COOKIES", "invalid-but-ui-disabled"),
        ])
        .unwrap();

        assert_eq!(settings.host, "127.0.0.1");
        assert_eq!(settings.port, 8080);
        assert_eq!(settings.litellm_config, "/tmp/custom.yaml");
        assert_eq!(settings.log_level(), Level::DEBUG);
    }

    #[rstest]
    #[case::empty("")]
    #[case::filter("litellm=debug")]
    #[case::invalid("invalid")]
    fn unrecognized_log_levels_fall_back_to_info(#[case] value: &str) {
        let settings: Settings = parse(&[("RUST_LOG", value)]).unwrap();

        assert_eq!(settings.log_level(), Level::INFO);
    }

    #[rstest]
    #[case::empty("")]
    #[case::negative("-1")]
    #[case::overflow("65536")]
    #[case::invalid("http")]
    fn invalid_ports_fail_startup(#[case] value: &str) {
        assert!(parse::<Settings>(&[("PORT", value)]).is_err());
    }

    #[rstest]
    fn ui_defaults_keep_secure_cookies_and_preserve_password() {
        let settings: UiSettings = parse(&[("UI_PASSWORD", " secret,with spaces ")]).unwrap();

        assert_eq!(settings.ui_username, "admin");
        assert_eq!(settings.ui_password.expose(), " secret,with spaces ");
        assert!(settings.litellm_ui_secure_cookies);
    }

    #[rstest]
    #[case::secure("true", true)]
    #[case::insecure("false", false)]
    fn ui_environment_overrides(#[case] value: &str, #[case] expected: bool) {
        let settings: UiSettings = parse(&[
            ("UI_USERNAME", "operator"),
            ("UI_PASSWORD", "password"),
            ("LITELLM_UI_SECURE_COOKIES", value),
        ])
        .unwrap();

        assert_eq!(settings.ui_username, "operator");
        assert_eq!(settings.litellm_ui_secure_cookies, expected);
    }

    #[rstest]
    #[case::missing_password(&[])]
    #[case::invalid_cookie_flag(&[("UI_PASSWORD", "password"), ("LITELLM_UI_SECURE_COOKIES", "invalid")])]
    fn invalid_ui_settings_fail(#[case] variables: &[(&str, &str)]) {
        assert!(parse::<UiSettings>(variables).is_err());
    }

    #[cfg(unix)]
    #[rstest]
    #[case::unrelated_variable("UNRELATED")]
    #[case::host_fallback("HOST")]
    fn non_unicode_environment_values_are_ignored(#[case] key: &str) {
        use std::os::unix::ffi::OsStringExt;

        let settings: Settings = from_iter([
            (OsString::from(key), OsString::from_vec(vec![0xff])),
            (OsString::from_vec(vec![0xff]), OsString::from("value")),
            (OsString::from("PORT"), OsString::from("8080")),
        ])
        .unwrap();

        assert_eq!(settings.host, "0.0.0.0");
        assert_eq!(settings.port, 8080);
    }
}
