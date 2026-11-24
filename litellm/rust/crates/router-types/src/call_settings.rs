use litellm_python_compat::lenient::{IsOn, ParamNames, Present, TextEntries};
use serde::Deserialize;
use serde_with::serde_as;

/// What a call does with a param its deployment does not support: `drop_params`,
/// `allowed_openai_params` and `additional_drop_params`, as a deployment's `litellm_params` or a
/// request spells them.
///
/// `None` is a key that is not there. A key that is there is `Some`, a `null` included, because a
/// call reads `{**litellm_params, **kwargs}`: whatever the request sends under a key replaces the
/// deployment's value, and a `null` reads as off or as no names.
#[serde_as]
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
pub struct CallSettings {
    /// Every unsupported param is dropped (`normalize_drop_params(value) is True`).
    #[serde_as(as = "Present<IsOn>")]
    #[serde(default)]
    pub drop_params: Option<bool>,
    /// The params passed on even where unsupported (`allowed_openai_param_names`).
    #[serde_as(as = "Present<ParamNames>")]
    #[serde(default)]
    pub allowed_openai_params: Option<Box<[String]>>,
    /// The params dropped before the call: a list's text entries.
    #[serde_as(as = "Present<TextEntries>")]
    #[serde(default)]
    pub additional_drop_params: Option<Box<[String]>>,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;

    fn names(names: &[&str]) -> Option<Box<[String]>> {
        Some(names.iter().map(|name| (*name).to_owned()).collect())
    }

    #[rstest]
    #[case::nothing(json!({}), CallSettings::default())]
    #[case::other_keys(json!({"model": "openai/a", "tpm": 5}), CallSettings::default())]
    #[case::flag(
        json!({"drop_params": true}),
        CallSettings { drop_params: Some(true), ..CallSettings::default() }
    )]
    #[case::flag_word(
        json!({"drop_params": "yes"}),
        CallSettings { drop_params: Some(true), ..CallSettings::default() }
    )]
    #[case::allowlist_text(
        json!({"allowed_openai_params": "tools,response_format"}),
        CallSettings {
            allowed_openai_params: names(&["tools", "response_format"]),
            ..CallSettings::default()
        }
    )]
    #[case::drop_list(
        json!({"additional_drop_params": ["response_format", 5]}),
        CallSettings {
            additional_drop_params: names(&["response_format"]),
            ..CallSettings::default()
        }
    )]
    #[case::nulls(
        json!({"drop_params": null, "allowed_openai_params": null, "additional_drop_params": null}),
        CallSettings {
            drop_params: Some(false),
            allowed_openai_params: names(&[]),
            additional_drop_params: names(&[]),
        }
    )]
    fn each_key_is_read_only_when_it_is_there(#[case] spelled: Value, #[case] read: CallSettings) {
        assert_eq!(
            serde_json::from_value::<CallSettings>(spelled).unwrap(),
            read
        );
    }
}
