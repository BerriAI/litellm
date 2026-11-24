//! The environment as a layer under config.yaml. A key the config leaves unset and a variable
//! backs is read from that variable here, once, so nothing past this crate asks which of the two
//! set it.
//!
//! One rule for every such key: the config's own value first, then `LITELLM_<KEY>`, then the
//! default. A flag is read by the one flag rule, `lax_bool`, whichever of the two wrote it, and
//! leaves this crate as a `bool`.
//!
//! The proxy differs in three places, and none of them is kept. It reads some of these variables
//! only when it starts, before the config's `environment_variables` are exported, where the
//! variable here is read from whatever `environment` is given. It reads each flag variable by a
//! rule of its own, one of which takes any word as on, where a word that is no flag leaves the key
//! unset here. And it reads `reasoning_auto_summary` as on when its variable is, even under a
//! config that sets it to `false`.

use std::collections::BTreeMap;

use litellm_env::Lookup;
use litellm_python_compat::flags::lax_bool;
use litellm_secret_value::SecretValue;

use crate::{Config, LiteLlmSettings};

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
                reasoning_auto_summary,
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
        Self {
            litellm_settings,
            origins: Origins(
                self.origins
                    .0
                    .into_iter()
                    .chain(origins.into_iter().flatten())
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
