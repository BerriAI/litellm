use litellm_core_utils::settings::{Selected, SettingSource, select_non_blank};
use litellm_llms_types::providers::anthropic::API_BASE;

use super::common_utils::{ANTHROPIC_API_BASE_ENV, ANTHROPIC_BASE_URL_ENV, messages_url};
use crate::base_llm::connection::{
    InvalidApiBase, RuntimeDefaults, complete_url, path_segments, validated_api_base,
};

/// The Anthropic host every Anthropic operation sends to, chosen once from the explicit
/// `api_base`, the host's runtime default, `ANTHROPIC_API_BASE`, `ANTHROPIC_BASE_URL`, then
/// the public API.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnthropicConnection {
    api_base: Selected<String>,
}

impl AnthropicConnection {
    pub fn resolve(
        api_base: Option<&str>,
        runtime_defaults: &RuntimeDefaults,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<Self, InvalidApiBase> {
        let api_base = select_non_blank(
            api_base,
            runtime_defaults.api_base.as_deref(),
            env_lookup,
            &[ANTHROPIC_API_BASE_ENV, ANTHROPIC_BASE_URL_ENV],
        )
        .unwrap_or_else(|| Selected::new(API_BASE.to_string(), SettingSource::ProviderDefault));
        Ok(Self {
            api_base: validated_api_base(api_base)?,
        })
    }

    pub fn api_base(&self) -> &str {
        &self.api_base.value
    }

    pub fn api_base_source(&self) -> SettingSource {
        self.api_base.source
    }

    pub fn messages_url(&self) -> String {
        messages_url(self.api_base())
    }

    pub fn endpoint(&self, path: &str) -> Result<String, InvalidApiBase> {
        complete_url(
            self.api_base().trim_end_matches('/'),
            &path_segments(path).collect::<Vec<_>>(),
        )
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn env(values: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |name| {
            values
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        }
    }

    fn runtime(api_base: Option<&str>) -> RuntimeDefaults {
        RuntimeDefaults {
            api_base: api_base.map(Into::into),
            ..RuntimeDefaults::default()
        }
    }

    #[rstest]
    #[case::explicit_beats_both_aliases(
        Some("https://explicit.test"),
        None,
        &[("ANTHROPIC_API_BASE", "https://a.test"), ("ANTHROPIC_BASE_URL", "https://b.test")],
        "https://explicit.test",
        SettingSource::Explicit
    )]
    #[case::runtime_default_beats_the_environment(
        None,
        Some("https://runtime.test"),
        &[("ANTHROPIC_API_BASE", "https://a.test")],
        "https://runtime.test",
        SettingSource::RuntimeDefault
    )]
    #[case::api_base_alias_before_base_url_alias(
        None,
        None,
        &[("ANTHROPIC_BASE_URL", "https://b.test"), ("ANTHROPIC_API_BASE", "https://a.test")],
        "https://a.test",
        SettingSource::Environment("ANTHROPIC_API_BASE")
    )]
    #[case::base_url_alias_alone(
        None,
        None,
        &[("ANTHROPIC_BASE_URL", "https://b.test")],
        "https://b.test",
        SettingSource::Environment("ANTHROPIC_BASE_URL")
    )]
    #[case::blank_overrides_fall_through(
        Some(" "),
        Some(""),
        &[],
        "https://api.anthropic.com",
        SettingSource::ProviderDefault
    )]
    fn the_api_base_follows_the_layer_precedence(
        #[case] explicit: Option<&str>,
        #[case] runtime_default: Option<&str>,
        #[case] vars: &'static [(&'static str, &'static str)],
        #[case] expected: &str,
        #[case] source: SettingSource,
    ) {
        let connection =
            AnthropicConnection::resolve(explicit, &runtime(runtime_default), &env(vars)).unwrap();
        assert_eq!(
            (connection.api_base(), connection.api_base_source()),
            (expected, source)
        );
    }

    #[rstest]
    #[case::explicit("not a url", None)]
    #[case::runtime_default(" ", Some("relative/path"))]
    fn a_malformed_override_is_rejected(
        #[case] explicit: &str,
        #[case] runtime_default: Option<&str>,
    ) {
        assert_eq!(
            AnthropicConnection::resolve(Some(explicit), &runtime(runtime_default), &env(&[])),
            Err(InvalidApiBase)
        );
    }
}
