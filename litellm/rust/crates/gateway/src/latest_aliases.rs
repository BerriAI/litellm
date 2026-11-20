//! The cost map's alias rows, such as `claude-sonnet-latest`, as the proxy's
//! `litellm_core_utils/model_latest_aliases.py` reads them: a limit set on the model an alias
//! stands for also counts a call that asks for the alias.

use std::collections::BTreeMap;

use litellm_model_catalog::Catalog;

/// The model an alias row stands for, and the provider the alias is asked through.
struct Target {
    model: String,
    provider: Option<String>,
}

/// Every alias row of a cost map.
///
/// Accepted difference: an alias written under a prefix is asked through that prefix. Python
/// only takes the prefix when its `provider_list` names it and otherwise asks the row for its
/// provider, which reads the same for every alias the cost map holds.
#[derive(Default)]
pub struct LatestAliases(BTreeMap<String, Target>);

impl LatestAliases {
    pub fn of(catalog: &Catalog) -> Self {
        let provider_of = |alias: &str, model: &str| match alias.split_once('/') {
            Some((prefix, _)) => Some(prefix.to_owned()),
            None => catalog
                .get(model)
                .and_then(|row| row.get("litellm_provider")?.as_str())
                .map(str::to_owned),
        };
        Self(
            catalog
                .aliases()
                .iter()
                .map(|(alias, model)| {
                    let target = Target {
                        model: model.clone(),
                        provider: provider_of(alias, model),
                    };
                    (alias.clone(), target)
                })
                .collect(),
        )
    }

    /// `_alias_target`: the model the alias `key` stands for when it is asked through `provider`,
    /// without that provider's prefix.
    fn target(&self, key: &str, provider: &str) -> Option<&str> {
        let target = self
            .0
            .get(key)
            .filter(|target| target.provider.as_deref() == Some(provider))?;
        let prefixed = target
            .model
            .strip_prefix(provider)
            .and_then(|rest| rest.strip_prefix('/'));
        Some(prefixed.unwrap_or(&target.model))
    }

    /// `expand_model_latest_alias`: `model`, and after it the model it stands for when the cost
    /// map serves it as an alias.
    pub fn expand(&self, model: &str) -> Vec<String> {
        let stands_for = match model.split_once('/') {
            None => self.0.get(model).map(|target| target.model.clone()),
            Some((provider, name)) => self
                .target(model, provider)
                .or_else(|| self.target(name, provider))
                .filter(|target| *target != name)
                .map(|target| format!("{provider}/{target}")),
        };
        std::iter::once(model.to_owned())
            .chain(stands_for)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};

    use super::*;

    #[fixture]
    fn aliases() -> LatestAliases {
        let catalog = Catalog::parse(
            br#"{
                "sonnet-5": {"litellm_provider": "anthropic"},
                "sonnet-latest": {"alias_of": "sonnet-5"},
                "azure_ai/sonnet-5": {"litellm_provider": "azure_ai"},
                "azure_ai/sonnet-latest": {"alias_of": "azure_ai/sonnet-5"},
                "global.sonnet-5": {"litellm_provider": "bedrock_converse"},
                "bedrock/sonnet-latest": {"alias_of": "global.sonnet-5"}
            }"#,
        )
        .unwrap()
        .value;
        LatestAliases::of(&catalog)
    }

    #[rstest]
    #[case::an_alias("sonnet-latest", &["sonnet-latest", "sonnet-5"])]
    #[case::the_model_itself("sonnet-5", &["sonnet-5"])]
    #[case::a_name_the_cost_map_does_not_hold("claude", &["claude"])]
    #[case::an_alias_written_under_its_provider(
        "azure_ai/sonnet-latest",
        &["azure_ai/sonnet-latest", "azure_ai/sonnet-5"]
    )]
    #[case::a_bare_alias_asked_through_its_models_provider(
        "anthropic/sonnet-latest",
        &["anthropic/sonnet-latest", "anthropic/sonnet-5"]
    )]
    #[case::a_bare_alias_asked_through_another_provider(
        "openai/sonnet-latest",
        &["openai/sonnet-latest"]
    )]
    #[case::an_alias_of_a_model_its_provider_hosts(
        "bedrock/sonnet-latest",
        &["bedrock/sonnet-latest", "bedrock/global.sonnet-5"]
    )]
    #[case::a_model_under_its_provider("azure_ai/sonnet-5", &["azure_ai/sonnet-5"])]
    fn a_name_expands_to_itself_and_the_model_it_stands_for(
        aliases: LatestAliases,
        #[case] model: &str,
        #[case] expected: &[&str],
    ) {
        assert_eq!(aliases.expand(model), expected);
    }
}
