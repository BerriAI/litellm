//! The model groups a config names and where the router can send a request for each: its
//! aliases and its fallbacks.

use std::collections::{BTreeMap, BTreeSet};

use litellm_config::RouterSettings;
use serde_json::Value;

/// The model groups the config names and where the router can send a request for each.
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct Groups {
    names: BTreeSet<Box<str>>,
    aliases: BTreeMap<Box<str>, Box<str>>,
    fallbacks: BTreeMap<Box<str>, BTreeSet<Box<str>>>,
    /// The fallbacks of every group: `default_fallbacks` and the `*` entries.
    shared: BTreeSet<Box<str>>,
}

const EVERY_GROUP: &str = "*";

fn fallback_entries(lists: &[&Option<Vec<Value>>]) -> Option<Vec<(Box<str>, Box<str>)>> {
    lists
        .iter()
        .filter_map(|list| list.as_ref())
        .flatten()
        .map(|entry| serde_json::from_value::<BTreeMap<Box<str>, Vec<Box<str>>>>(entry.clone()))
        .collect::<Result<Vec<_>, _>>()
        .ok()
        .map(|entries| {
            entries
                .into_iter()
                .flatten()
                .flat_map(|(group, targets)| {
                    targets
                        .into_iter()
                        .map(move |target| (group.clone(), target))
                })
                .collect()
        })
}

fn alias_target(target: &Value) -> Option<Box<str>> {
    match target {
        Value::String(group) => Some(group.as_str().into()),
        Value::Object(alias) => alias.get("model")?.as_str().map(Box::from),
        _ => None,
    }
}

impl Groups {
    /// `None` when a fallback list or an alias is written in a shape this does not read.
    pub(super) fn read<'a>(
        names: impl Iterator<Item = &'a str>,
        settings: &RouterSettings,
    ) -> Option<Self> {
        let entries = fallback_entries(&[
            &settings.fallbacks,
            &settings.context_window_fallbacks,
            &settings.content_policy_fallbacks,
        ])?;
        let aliases = settings
            .model_group_alias
            .iter()
            .flatten()
            .map(|(alias, target)| Some((alias.as_str().into(), alias_target(target)?)))
            .collect::<Option<_>>()?;
        let (shared, own): (Vec<_>, Vec<_>) = entries
            .into_iter()
            .partition(|(group, _)| &**group == EVERY_GROUP);
        Some(Self {
            names: names.map(Box::from).collect(),
            aliases,
            fallbacks: own.into_iter().fold(
                BTreeMap::<Box<str>, BTreeSet<Box<str>>>::new(),
                |mut fallbacks, (group, target)| {
                    fallbacks.entry(group).or_default().insert(target);
                    fallbacks
                },
            ),
            shared: shared
                .into_iter()
                .map(|(_, target)| target)
                .chain(
                    settings
                        .default_fallbacks
                        .iter()
                        .flatten()
                        .map(|target| target.as_str().into()),
                )
                .collect(),
        })
    }

    /// The group `model` names, itself or through an alias.
    pub(super) fn group<'a>(&'a self, model: &'a str) -> &'a str {
        self.aliases.get(model).map_or(model, |group| &**group)
    }

    pub(super) fn holds(&self, group: &str) -> bool {
        self.names.contains(group)
    }

    fn targets<'a>(&'a self, group: &str) -> impl Iterator<Item = &'a str> {
        self.fallbacks
            .get(group)
            .into_iter()
            .flatten()
            .chain(&self.shared)
            .map(|target| self.group(target))
    }

    /// `group` and every group it can fall back to, through any number of fallbacks, in the order
    /// they are reached.
    pub(super) fn reached<'a>(&'a self, group: &'a str) -> Vec<&'a str> {
        let mut reached = vec![group];
        let mut next = 0;
        while let Some(&at) = reached.get(next) {
            let found: Vec<&str> = self.targets(at).collect();
            for target in found {
                if !reached.contains(&target) {
                    reached.push(target);
                }
            }
            next += 1;
        }
        reached
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::a_fallback_target_that_is_not_a_name("  fallbacks:\n    - claude: [{model: backup}]\n")]
    #[case::a_fallback_entry_that_is_not_a_mapping("  fallbacks: [claude]\n")]
    #[case::an_alias_without_a_model("  model_group_alias:\n    sonnet: {hidden: true}\n")]
    fn groups_written_in_another_shape_are_not_read(#[case] router_settings: &str) {
        let settings =
            litellm_config::Config::from_yaml(&format!("router_settings:\n{router_settings}"))
                .unwrap()
                .router_settings;

        assert_eq!(Groups::read(["claude"].into_iter(), &settings), None);
    }
}
