use litellm_auth_types::SecretValue;
use litellm_layered::{Invalid, Layered, read};
use litellm_tracing::Level;

#[derive(Layered)]
#[expect(dead_code, reason = "only BootKeysPartial is built")]
pub(super) struct BootKeys {
    #[layer(env = "HOST", read = read::text)]
    host: String,
    #[layer(env = "PORT", read = read::required::<u16>)]
    port: u16,
    #[layer(env = "LITELLM_CONFIG", read = read::text)]
    litellm_config: String,
    #[layer(env = "RUST_LOG", read = read::text)]
    rust_log: String,
    #[layer(env = "LITELLM_UI_PATH", read = read::text)]
    ui_path: String,
}

pub(super) struct Settings {
    pub host: String,
    pub port: u16,
    pub litellm_config: String,
    pub ui_path: Option<String>,
    rust_log: Option<String>,
}

impl From<BootKeysPartial> for Settings {
    fn from(keys: BootKeysPartial) -> Self {
        Self {
            host: keys.host.unwrap_or_else(|| "0.0.0.0".into()),
            port: keys.port.unwrap_or(4000),
            litellm_config: keys.litellm_config.unwrap_or_else(|| "config.yaml".into()),
            ui_path: keys.ui_path,
            rust_log: keys.rust_log,
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

#[derive(Layered)]
#[expect(dead_code, reason = "only UiKeysPartial is built")]
pub(super) struct UiKeys {
    #[layer(env = "UI_USERNAME", read = read::text)]
    ui_username: String,
    #[layer(env = "UI_PASSWORD", read = secret, secret)]
    ui_password: SecretValue,
    #[layer(env = "LITELLM_UI_SECURE_COOKIES", read = read::required::<bool>)]
    litellm_ui_secure_cookies: bool,
}

fn secret(value: &str) -> Result<Option<SecretValue>, Invalid> {
    Ok(Some(SecretValue::new(value)))
}

pub(super) struct UiSettings {
    pub ui_username: String,
    pub ui_password: SecretValue,
    pub litellm_ui_secure_cookies: bool,
}

#[derive(Debug, thiserror::Error)]
#[error("UI_PASSWORD must be set when LITELLM_UI_PATH serves the UI")]
pub(super) struct MissingUiPassword;

impl TryFrom<UiKeysPartial> for UiSettings {
    type Error = MissingUiPassword;

    fn try_from(keys: UiKeysPartial) -> Result<Self, Self::Error> {
        Ok(Self {
            ui_username: keys.ui_username.unwrap_or_else(|| "admin".into()),
            ui_password: keys.ui_password.ok_or(MissingUiPassword)?,
            litellm_ui_secure_cookies: keys.litellm_ui_secure_cookies.unwrap_or(true),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;

    use litellm_layered::Env;
    use rstest::rstest;

    use super::*;

    fn boot(variables: &[(&str, &str)]) -> Result<Settings, litellm_layered::Error> {
        BootKeysPartial::from_env(&Env::of(variables)).map(Settings::from)
    }

    fn ui(variables: &[(&str, &str)]) -> Result<UiSettings, Box<dyn std::error::Error>> {
        Ok(UiSettings::try_from(UiKeysPartial::from_env(&Env::of(
            variables,
        ))?)?)
    }

    #[rstest]
    fn defaults_allow_startup_without_ui_credentials() {
        let settings = boot(&[]).unwrap();

        assert_eq!(settings.host, "0.0.0.0");
        assert_eq!(settings.port, 4000);
        assert_eq!(settings.litellm_config, "config.yaml");
        assert_eq!(settings.ui_path, None);
        assert_eq!(settings.log_level(), Level::INFO);
    }

    #[rstest]
    fn environment_overrides_startup_settings() {
        let settings = boot(&[
            ("HOST", "127.0.0.1"),
            ("PORT", "8080"),
            ("LITELLM_CONFIG", "/tmp/custom.yaml"),
            ("RUST_LOG", "debug"),
            ("LITELLM_UI_PATH", "/srv/ui"),
            ("LITELLM_UI_SECURE_COOKIES", "invalid-but-ui-disabled"),
        ])
        .unwrap();

        assert_eq!(settings.host, "127.0.0.1");
        assert_eq!(settings.port, 8080);
        assert_eq!(settings.litellm_config, "/tmp/custom.yaml");
        assert_eq!(settings.ui_path.as_deref(), Some("/srv/ui"));
        assert_eq!(settings.log_level(), Level::DEBUG);
    }

    #[rstest]
    #[case::empty("")]
    #[case::filter("litellm=debug")]
    #[case::invalid("invalid")]
    fn unrecognized_log_levels_fall_back_to_info(#[case] value: &str) {
        assert_eq!(
            boot(&[("RUST_LOG", value)]).unwrap().log_level(),
            Level::INFO
        );
    }

    #[rstest]
    #[case::negative("-1")]
    #[case::overflow("65536")]
    #[case::invalid("abc")]
    fn invalid_ports_fail_startup_naming_the_variable(#[case] value: &str) {
        let error = boot(&[("PORT", value)]).err().unwrap();
        assert_eq!(error.name, "PORT");
        assert!(error.to_string().contains("PORT"));
    }

    #[rstest]
    fn blank_startup_settings_fall_back_to_defaults() {
        let settings = boot(&[("HOST", ""), ("PORT", " "), ("LITELLM_UI_PATH", "")]).unwrap();

        assert_eq!(settings.host, "0.0.0.0");
        assert_eq!(settings.port, 4000);
        assert_eq!(settings.ui_path, None);
    }

    #[rstest]
    fn ui_defaults_keep_secure_cookies_and_preserve_password() {
        let settings = ui(&[("UI_PASSWORD", " secret,with spaces ")]).unwrap();

        assert_eq!(settings.ui_username, "admin");
        assert_eq!(settings.ui_password.expose(), " secret,with spaces ");
        assert!(settings.litellm_ui_secure_cookies);
    }

    #[rstest]
    #[case::secure("true", true)]
    #[case::insecure("false", false)]
    fn ui_environment_overrides(#[case] value: &str, #[case] expected: bool) {
        let settings = ui(&[
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
    #[case::blank_password(&[("UI_PASSWORD", " ")])]
    #[case::invalid_cookie_flag(&[("UI_PASSWORD", "password"), ("LITELLM_UI_SECURE_COOKIES", "invalid")])]
    fn invalid_ui_settings_fail(#[case] variables: &[(&str, &str)]) {
        assert!(ui(variables).is_err());
    }

    #[cfg(unix)]
    #[rstest]
    #[case::unrelated_variable("UNRELATED")]
    #[case::host_fallback("HOST")]
    fn non_unicode_environment_values_are_ignored(#[case] key: &str) {
        use std::os::unix::ffi::OsStringExt;

        let env = Env::from_vars([
            (OsString::from(key), OsString::from_vec(vec![0xff])),
            (OsString::from_vec(vec![0xff]), OsString::from("value")),
            (OsString::from("PORT"), OsString::from("8080")),
        ]);
        let settings = Settings::from(BootKeysPartial::from_env(&env).unwrap());

        assert_eq!(settings.host, "0.0.0.0");
        assert_eq!(settings.port, 8080);
    }
}
