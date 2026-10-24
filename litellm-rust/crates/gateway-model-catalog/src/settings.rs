use litellm_core_utils::settings::{Layer, Lookup};
use litellm_model_catalog::DEFAULT_REGISTRY_URL;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CostMapKeysPartial {
    pub local_only: Option<bool>,
    pub offline: Option<bool>,
    pub url: Option<String>,
}

impl Layer for CostMapKeysPartial {
    fn or(self, lower: Self) -> Self {
        Self {
            local_only: self.local_only.or(lower.local_only),
            offline: self.offline.or(lower.offline),
            url: self.url.or(lower.url),
        }
    }
}

impl CostMapKeysPartial {
    pub fn from_env(lookup: &impl Lookup) -> Self {
        Self {
            local_only: lookup
                .get("LITELLM_LOCAL_MODEL_COST_MAP")
                .map(|value| value.to_lowercase() == "true"),
            offline: lookup
                .get("LITELLM_OFFLINE_MODE")
                .map(|value| value.trim().to_lowercase() == "true"),
            url: lookup.truthy("LITELLM_MODEL_COST_MAP_URL"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CostMapSettings {
    local_only: bool,
    offline: bool,
    url: String,
}

impl CostMapSettings {
    pub fn from_partial(keys: CostMapKeysPartial) -> Self {
        Self {
            local_only: keys.local_only.unwrap_or_default(),
            offline: keys.offline.unwrap_or_default(),
            url: keys.url.unwrap_or_else(|| DEFAULT_REGISTRY_URL.to_owned()),
        }
    }

    pub fn remote_url(&self) -> Option<&str> {
        (!self.local_only && !self.offline).then_some(self.url.as_str())
    }

    pub fn is_env_forced(&self) -> bool {
        self.local_only || self.offline
    }
}

impl Default for CostMapSettings {
    fn default() -> Self {
        Self::from_partial(CostMapKeysPartial::default())
    }
}

#[cfg(test)]
mod tests {
    use litellm_core_utils::settings::{Lookup, merge};
    use rstest::rstest;

    use super::*;

    struct Values<'a>(&'a [(&'a str, &'a str)]);

    impl Lookup for Values<'_> {
        fn get(&self, name: &str) -> Option<String> {
            self.0
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| (*value).to_owned())
        }
    }

    #[rstest]
    #[case::uppercase_local("TRUE", true, true)]
    #[case::titlecase_local("True", true, true)]
    #[case::lowercase_local("true", true, true)]
    #[case::numeric_local("1", false, false)]
    #[case::yes_local("yes", false, false)]
    #[case::false_local("False", false, false)]
    #[case::empty_local("", false, false)]
    #[case::padded_local(" true ", false, true)]
    fn parses_local_and_offline_flags(
        #[case] value: &'static str,
        #[case] expected_local: bool,
        #[case] expected_offline: bool,
    ) {
        let parsed = CostMapKeysPartial::from_env(&Values(&[
            ("LITELLM_LOCAL_MODEL_COST_MAP", value),
            ("LITELLM_OFFLINE_MODE", value),
        ]));
        assert_eq!(parsed.local_only, Some(expected_local));
        assert_eq!(parsed.offline, Some(expected_offline));
    }

    #[rstest]
    #[case::environment_over_config(Some("environment"), Some("config"), Some("environment"))]
    #[case::empty_environment_falls_through(Some(""), Some("config"), Some("config"))]
    #[case::missing_environment_uses_config(None, Some("config"), Some("config"))]
    fn combines_environment_and_config_url(
        #[case] environment_url: Option<&'static str>,
        #[case] config_url: Option<&'static str>,
        #[case] expected: Option<&'static str>,
    ) {
        let values = environment_url
            .map(|value| [("LITELLM_MODEL_COST_MAP_URL", value)])
            .unwrap_or_default();
        let environment = CostMapKeysPartial::from_env(&Values(&values));
        let config = CostMapKeysPartial {
            local_only: None,
            offline: None,
            url: config_url.map(str::to_owned),
        };
        assert_eq!(merge([environment, config]).url.as_deref(), expected);
    }

    #[rstest]
    fn defaults_to_shared_registry_url() {
        let settings = CostMapSettings::default();
        assert_eq!(settings.remote_url(), Some(DEFAULT_REGISTRY_URL));
    }

    #[rstest]
    #[case::local_only(CostMapKeysPartial { local_only: Some(true), offline: None, url: None })]
    #[case::offline(CostMapKeysPartial { local_only: None, offline: Some(true), url: None })]
    fn either_flag_disables_remote_revalidation(#[case] partial: CostMapKeysPartial) {
        assert_eq!(CostMapSettings::from_partial(partial).remote_url(), None);
    }
}
