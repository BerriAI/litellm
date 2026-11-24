//! The environment as a layer under config.yaml. A key the config leaves unset and a variable
//! backs is read from that variable here, once, so nothing past this crate asks which of the two
//! set it.
//!
//! One rule for every such key: the config's own value first, then `LITELLM_<KEY>`, then the
//! default. A flag is read by the one flag rule, `lax_bool`, whichever of the two wrote it, and
//! leaves this crate as a `bool`. Two keys are read as the proxy reads them instead.
//! `reasoning_auto_summary` is on when the config's value is truthy or its variable is `true` in
//! any case, even under a config that sets it to `false`. `request_timeout` is read from
//! `REQUEST_TIMEOUT`, the name the proxy reads it by, as `float()` reads its text.
//!
//! The proxy differs in two places, and neither is kept. It reads some of these variables only
//! when it starts, before the config's `environment_variables` are exported, where the variable
//! here is read from whatever `environment` is given. It reads each flag variable by a rule of its
//! own, one of which takes any word as on, where a word that is no flag leaves the key unset here.

use std::collections::BTreeMap;

use litellm_env::Lookup;
use litellm_python_compat::{
    flags::{lax_bool, lower_is_true},
    number::float,
};
use litellm_secret_value::SecretValue;

use crate::{Config, LiteLlmSettings, Spelled};

const REASONING_AUTO_SUMMARY: &str = "litellm_settings.reasoning_auto_summary";
const REASONING_AUTO_SUMMARY_VARIABLE: &str = "LITELLM_REASONING_AUTO_SUMMARY";

/// What the environment did to each config key the config left unset, with the variable it read.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Origins(BTreeMap<&'static str, Origin>);

#[derive(Clone, Debug, PartialEq, Eq)]
enum Origin {
    Set(String),
    Unread(String),
}

impl Origins {
    /// The variable that set the config key `key`, as `litellm_settings.drop_params` names one,
    /// when the environment set it.
    pub fn variable(&self, key: &str) -> Option<&str> {
        match self.0.get(key)? {
            Origin::Set(variable) => Some(variable),
            Origin::Unread(_) => None,
        }
    }

    /// The variable of the config key `key` when it holds text the key is not read from, which
    /// leaves the key unset.
    pub fn unread(&self, key: &str) -> Option<&str> {
        match self.0.get(key)? {
            Origin::Unread(variable) => Some(variable),
            Origin::Set(_) => None,
        }
    }
}

/// What a key holds when its variable sets it, `None` for text the key is not read from.
trait Written: Sized {
    fn written(text: String) -> Option<Self>;
}

impl Written for bool {
    fn written(text: String) -> Option<Self> {
        lax_bool(&text.as_str().into())
    }
}

impl Written for String {
    fn written(text: String) -> Option<Self> {
        Some(text)
    }
}

impl Written for SecretValue {
    fn written(text: String) -> Option<Self> {
        Some(Self::new(text))
    }
}

impl Written for Spelled<f64> {
    fn written(text: String) -> Option<Self> {
        float(&text).map(Self::Value)
    }
}

/// The config's `own` value of `key` when it set one, else what `variable` holds, with what the
/// variable did to the key. A blank variable is unset.
fn filled<T: Written>(
    own: Option<T>,
    environment: &dyn Lookup,
    key: &'static str,
    variable: String,
) -> (Option<T>, Option<(&'static str, Origin)>) {
    if own.is_some() {
        return (own, None);
    }
    let Some(text) = environment
        .get(&variable)
        .filter(|text| !text.trim().is_empty())
    else {
        return (None, None);
    };
    match T::written(text) {
        Some(value) => (Some(value), Some((key, Origin::Set(variable)))),
        None => (None, Some((key, Origin::Unread(variable)))),
    }
}

/// `is_reasoning_auto_summary_enabled`: on when the config's `own` value is, else when the
/// variable is `true` in any case, with the variable when it turned the key on.
fn reasoning_auto_summary(
    own: bool,
    environment: &dyn Lookup,
) -> (bool, Option<(&'static str, Origin)>) {
    if own {
        return (true, None);
    }
    let on = environment
        .get(REASONING_AUTO_SUMMARY_VARIABLE)
        .is_some_and(|text| lower_is_true(&text));
    let origin = Origin::Set(REASONING_AUTO_SUMMARY_VARIABLE.to_owned());
    (on, on.then_some((REASONING_AUTO_SUMMARY, origin)))
}

/// `$settings` with each of the fields it leaves unset read from `LITELLM_<FIELD>`, and what the
/// variable did to each key it was read for.
macro_rules! layered {
    ($section:ident, $settings:expr, $environment:expr, $key:literal, [$($field:ident),+ $(,)?]) => {{
        let settings = $settings;
        $(
            let $field = filled(
                settings.$field,
                $environment,
                concat!($key, ".", stringify!($field)),
                concat!("LITELLM_", stringify!($field)).to_uppercase(),
            );
        )+
        let origins = [$($field.1),+];
        ($section { $($field: $field.0,)+ ..settings }, origins)
    }};
}

impl Config {
    /// This config with `environment` under it.
    pub fn with_environment(self, environment: &dyn Lookup) -> Self {
        let (litellm_settings, origins) = layered!(
            LiteLlmSettings,
            self.litellm_settings,
            environment,
            "litellm_settings",
            [
                drop_params,
                modify_params,
                enable_anthropic_prompt_caching,
                use_chat_completions_url_for_anthropic_messages,
                route_all_chat_openai_to_responses,
                gemini_live_defer_setup,
                anthropic_prompt_caching_ttl,
                model_cost_map_url,
                blog_posts_url,
                anthropic_beta_headers_url,
                autorouter_presets_url,
            ]
        );
        let (reasoning_auto_summary, summarized) =
            reasoning_auto_summary(litellm_settings.reasoning_auto_summary, environment);
        let (request_timeout, timed) = filled(
            litellm_settings.request_timeout,
            environment,
            "litellm_settings.request_timeout",
            "REQUEST_TIMEOUT".to_owned(),
        );
        Self {
            litellm_settings: LiteLlmSettings {
                reasoning_auto_summary,
                request_timeout,
                ..litellm_settings
            },
            origins: Origins(
                self.origins
                    .0
                    .into_iter()
                    .chain(origins.into_iter().flatten())
                    .chain(summarized)
                    .chain(timed)
                    .collect(),
            ),
            ..self
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    const KEY: &str = "litellm_settings.drop_params";

    fn layered(yaml: &str, written: Option<&str>) -> Config {
        let lookup = |name: &str| {
            (name == "LITELLM_DROP_PARAMS")
                .then_some(written)
                .flatten()
                .map(str::to_owned)
        };
        Config::from_yaml(yaml).unwrap().with_environment(&lookup)
    }

    #[rstest]
    #[case::the_variable_when_the_config_leaves_the_key_unset(
        "model_list: []\n",
        Some("true"),
        Some(true),
        Some("LITELLM_DROP_PARAMS")
    )]
    #[case::the_config_when_both_set_it(
        "litellm_settings:\n  drop_params: false\n",
        Some("true"),
        Some(false),
        None
    )]
    #[case::neither("model_list: []\n", None, None, None)]
    #[case::a_blank_variable("model_list: []\n", Some("  "), None, None)]
    fn a_key_is_read_from_the_config_first_and_its_variable_second(
        #[case] yaml: &str,
        #[case] written: Option<&str>,
        #[case] value: Option<bool>,
        #[case] variable: Option<&str>,
    ) {
        let config = layered(yaml, written);

        assert_eq!(
            (
                config.litellm_settings.drop_params,
                config.origins.variable(KEY)
            ),
            (value, variable)
        );
    }

    #[rstest]
    fn a_variable_is_named_after_its_key() {
        let lookup = |name: &str| {
            (name == "LITELLM_USE_CHAT_COMPLETIONS_URL_FOR_ANTHROPIC_MESSAGES")
                .then(|| "true".to_owned())
        };

        let config = Config::default().with_environment(&lookup);

        assert_eq!(
            config
                .litellm_settings
                .use_chat_completions_url_for_anthropic_messages,
            Some(true)
        );
    }

    #[rstest]
    #[case::off("false", Some(false), Some("LITELLM_DROP_PARAMS"), None)]
    #[case::another_word_for_on("yes", Some(true), Some("LITELLM_DROP_PARAMS"), None)]
    #[case::a_word_that_is_no_flag("banana", None, None, Some("LITELLM_DROP_PARAMS"))]
    fn a_flag_variable_is_read_by_the_one_flag_rule(
        #[case] written: &str,
        #[case] value: Option<bool>,
        #[case] variable: Option<&str>,
        #[case] unread: Option<&str>,
    ) {
        let config = layered("model_list: []\n", Some(written));

        assert_eq!(
            (
                config.litellm_settings.drop_params,
                config.origins.variable(KEY),
                config.origins.unread(KEY)
            ),
            (value, variable, unread)
        );
    }

    fn read_with(yaml: &str, variable: &str, written: Option<&str>) -> Config {
        let lookup = |name: &str| {
            (name == variable)
                .then_some(written)
                .flatten()
                .map(str::to_owned)
        };
        Config::from_yaml(yaml).unwrap().with_environment(&lookup)
    }

    #[rstest]
    #[case::unset("model_list: []\n", None, false)]
    #[case::the_config("litellm_settings:\n  reasoning_auto_summary: true\n", None, true)]
    #[case::config_text_that_is_truthy(
        "litellm_settings:\n  reasoning_auto_summary: \"false\"\n",
        None,
        true
    )]
    #[case::the_variable_under_a_config_set_to_false(
        "litellm_settings:\n  reasoning_auto_summary: false\n",
        Some("TRUE"),
        true
    )]
    #[case::a_variable_word_other_than_true("model_list: []\n", Some("yes"), false)]
    #[case::a_padded_variable("model_list: []\n", Some(" true"), false)]
    fn reasoning_auto_summary_is_on_when_the_config_is_truthy_or_its_variable_is_true(
        #[case] yaml: &str,
        #[case] written: Option<&str>,
        #[case] expected: bool,
    ) {
        let config = read_with(yaml, "LITELLM_REASONING_AUTO_SUMMARY", written);

        assert_eq!(config.litellm_settings.reasoning_auto_summary, expected);
    }

    #[rstest]
    #[case::the_variable_when_the_config_leaves_it_unset(
        "model_list: []\n",
        Some(" 2.5 "),
        Some(2.5),
        Some("REQUEST_TIMEOUT")
    )]
    #[case::the_config_when_both_set_it(
        "litellm_settings:\n  request_timeout: 30\n",
        Some("5"),
        Some(30.0),
        None
    )]
    #[case::text_float_refuses("model_list: []\n", Some("soon"), None, None)]
    fn a_request_timeout_is_read_from_the_config_first_and_request_timeout_second(
        #[case] yaml: &str,
        #[case] written: Option<&str>,
        #[case] seconds: Option<f64>,
        #[case] variable: Option<&str>,
    ) {
        let config = read_with(yaml, "REQUEST_TIMEOUT", written);

        assert_eq!(
            (
                config.litellm_settings.request_timeout,
                config.origins.variable("litellm_settings.request_timeout")
            ),
            (seconds.map(Spelled::Value), variable)
        );
    }

    #[rstest]
    fn a_text_key_holds_its_variables_text() {
        let lookup =
            |name: &str| (name == "LITELLM_ANTHROPIC_PROMPT_CACHING_TTL").then(|| "1h".to_owned());

        let config = Config::default().with_environment(&lookup);

        assert_eq!(
            (
                config
                    .litellm_settings
                    .anthropic_prompt_caching_ttl
                    .as_deref(),
                config
                    .origins
                    .variable("litellm_settings.anthropic_prompt_caching_ttl")
            ),
            (Some("1h"), Some("LITELLM_ANTHROPIC_PROMPT_CACHING_TTL"))
        );
    }

    #[rstest]
    fn a_second_environment_keeps_what_the_first_one_set() {
        let first = |name: &str| (name == "LITELLM_DROP_PARAMS").then(|| "true".to_owned());
        let second = |name: &str| (name == "LITELLM_MODIFY_PARAMS").then(|| "true".to_owned());

        let config = Config::default()
            .with_environment(&first)
            .with_environment(&second);

        assert_eq!(
            (
                config.origins.variable(KEY),
                config.origins.variable("litellm_settings.modify_params")
            ),
            (Some("LITELLM_DROP_PARAMS"), Some("LITELLM_MODIFY_PARAMS"))
        );
    }

    #[rstest]
    #[case::the_config(
        "litellm_settings:\n  model_cost_map_url: https://user:credential-value@a.example/map.json\n",
        None
    )]
    #[case::its_variable(
        "model_list: []\n",
        Some("https://user:credential-value@a.example/map.json")
    )]
    fn a_url_key_is_hidden_whichever_of_the_two_set_it(
        #[case] yaml: &str,
        #[case] written: Option<&str>,
    ) {
        let lookup = |name: &str| {
            (name == "LITELLM_MODEL_COST_MAP_URL")
                .then_some(written)
                .flatten()
                .map(str::to_owned)
        };

        let config = Config::from_yaml(yaml).unwrap().with_environment(&lookup);
        let debug = format!("{:?}", config.litellm_settings);

        assert!(!debug.contains("credential-value"), "{debug}");
        assert!(debug.contains("[REDACTED]"), "{debug}");
    }
}
