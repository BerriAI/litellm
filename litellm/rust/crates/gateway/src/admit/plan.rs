//! What the gateway leaves to the Python proxy, decided once from the config it runs with.
//!
//! A [`Finding`] is one setting the native app does not honor as Python does, and its [`Scope`]
//! says whose requests that leaves to Python. A [`Plan`] holds every finding and answers, for one
//! request, whether any of them reaches it. Nothing here does IO.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
};

use litellm_config::RouterSettings;
use serde_json::Value;

use super::{Refused, request::Trait};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Scope {
    Everything,
    /// The requests for this model group, and for each group that can fall back to it.
    Group(Box<str>),
    /// The requests that carry this trait.
    Asking(Trait),
}

impl fmt::Display for Scope {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Everything => formatter.write_str("every request"),
            Self::Group(group) => write!(formatter, "requests for model group {group}"),
            Self::Asking(carried) => {
                write!(formatter, "requests with {}", <&str>::from(*carried))
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Finding {
    /// Named as the config spells it.
    pub setting: Box<str>,
    pub leaves: Scope,
}

/// Why one request is left to Python, and the setting or trait behind it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Left<'a> {
    pub why: Refused,
    pub over: &'a str,
}

/// Where the router can send a request for each model group.
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct Routes {
    groups: BTreeSet<Box<str>>,
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

impl Routes {
    /// `None` when a fallback list or an alias is written in a shape this does not read.
    pub(super) fn read<'a>(
        groups: impl Iterator<Item = &'a str>,
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
            groups: groups.map(Box::from).collect(),
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

    fn group<'a>(&'a self, model: &'a str) -> &'a str {
        self.aliases.get(model).map_or(model, |group| &**group)
    }

    fn targets<'a>(&'a self, group: &str) -> impl Iterator<Item = &'a str> {
        self.fallbacks
            .get(group)
            .into_iter()
            .flatten()
            .chain(&self.shared)
            .map(|target| self.group(target))
    }

    /// `left` grown by each group that can fall back to a group in it, or to one the config does
    /// not hold, until no group is added.
    fn reached(&self, left: BTreeMap<Box<str>, Box<str>>) -> BTreeMap<Box<str>, Box<str>> {
        let added: BTreeMap<Box<str>, Box<str>> = self
            .groups
            .iter()
            .filter(|group| !left.contains_key(*group))
            .filter_map(|group| {
                let target = self
                    .targets(group)
                    .find(|target| left.contains_key(*target) || !self.groups.contains(*target))?;
                Some((
                    group.clone(),
                    format!("router_settings fallback to {target}").into(),
                ))
            })
            .collect();
        match added.is_empty() {
            true => left,
            false => self.reached(left.into_iter().chain(added).collect()),
        }
    }
}

/// The requests the native app answers as Python would, by the config both run with.
#[derive(Debug)]
pub(super) struct Plan {
    findings: Box<[Finding]>,
    routes: Routes,
    /// Each model group left to Python, with the setting that leaves it.
    left: BTreeMap<Box<str>, Box<str>>,
}

impl Plan {
    pub(super) fn new(findings: Vec<Finding>, routes: Routes) -> Self {
        let own = findings
            .iter()
            .rev()
            .filter_map(|finding| match &finding.leaves {
                Scope::Group(group) => Some((group.clone(), finding.setting.clone())),
                Scope::Everything | Scope::Asking(_) => None,
            })
            .collect();
        Self {
            left: routes.reached(own),
            routes,
            findings: findings.into(),
        }
    }

    pub(super) fn findings(&self) -> &[Finding] {
        &self.findings
    }

    pub(super) fn everything(&self) -> Result<(), Left<'_>> {
        self.findings
            .iter()
            .find(|finding| finding.leaves == Scope::Everything)
            .map_or(Ok(()), |finding| {
                Err(Left {
                    why: Refused::Settings,
                    over: &finding.setting,
                })
            })
    }

    /// A trait the native app reads as Python does is refused only when a finding names it.
    pub(super) fn carrying(&self, carried: &[Trait]) -> Result<(), Left<'_>> {
        carried
            .iter()
            .find_map(|carried| {
                let named = self
                    .findings
                    .iter()
                    .find(|finding| finding.leaves == Scope::Asking(*carried));
                match (named, carried.is_read()) {
                    (Some(finding), _) => Some((carried, &*finding.setting)),
                    (None, false) => Some((carried, <&str>::from(*carried))),
                    (None, true) => None,
                }
            })
            .map_or(Ok(()), |(carried, over)| {
                Err(Left {
                    why: carried.part(),
                    over,
                })
            })
    }

    pub(super) fn model<'a>(&'a self, model: &'a str) -> Result<(), Left<'a>> {
        let group = self.routes.group(model);
        let left = |over| {
            Err(Left {
                why: Refused::Model,
                over,
            })
        };
        match (self.left.get(group), self.routes.groups.contains(group)) {
            (Some(setting), _) => left(setting),
            (None, false) => left(model),
            (None, true) => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn settings(yaml: &str) -> RouterSettings {
        litellm_config::Config::from_yaml(&format!("router_settings:\n{yaml}"))
            .unwrap()
            .router_settings
    }

    fn plan(left: &[&str], router_settings: &str) -> Plan {
        let findings = left
            .iter()
            .map(|group| Finding {
                setting: "model_list.litellm_params.tags".into(),
                leaves: Scope::Group((*group).into()),
            })
            .collect();
        let routes = Routes::read(
            ["claude", "backup", "last", "other"].into_iter(),
            &settings(router_settings),
        )
        .unwrap();
        Plan::new(findings, routes)
    }

    fn left_over(plan: &Plan, model: &str) -> Option<String> {
        plan.model(model).err().map(|left| left.over.to_owned())
    }

    #[rstest]
    fn a_group_with_a_finding_is_left_to_python_and_no_other_group_is() {
        let plan = plan(&["claude"], "  num_retries: 1\n");

        assert_eq!(
            (left_over(&plan, "claude"), left_over(&plan, "other")),
            (Some("model_list.litellm_params.tags".to_owned()), None)
        );
    }

    #[rstest]
    #[case::a_fallback("  fallbacks:\n    - claude: [backup]\n", "claude", "backup")]
    #[case::a_fallback_of_a_fallback(
        "  fallbacks:\n    - claude: [last]\n    - last: [backup]\n",
        "claude",
        "last"
    )]
    #[case::a_context_window_fallback(
        "  context_window_fallbacks:\n    - claude: [backup]\n",
        "claude",
        "backup"
    )]
    #[case::a_content_policy_fallback(
        "  content_policy_fallbacks:\n    - claude: [backup]\n",
        "claude",
        "backup"
    )]
    #[case::a_default_fallback("  default_fallbacks: [backup]\n", "other", "backup")]
    #[case::a_fallback_of_every_group("  fallbacks:\n    - \"*\": [backup]\n", "other", "backup")]
    #[case::a_fallback_to_an_alias(
        "  fallbacks:\n    - claude: [spare]\n  model_group_alias:\n    spare: backup\n",
        "claude",
        "backup"
    )]
    #[case::a_fallback_to_a_group_the_config_does_not_hold(
        "  fallbacks:\n    - claude: [nowhere]\n",
        "claude",
        "nowhere"
    )]
    fn a_group_that_can_fall_back_to_one_left_to_python_is_left_too(
        #[case] router_settings: &str,
        #[case] model: &str,
        #[case] target: &str,
    ) {
        let plan = plan(&["backup"], router_settings);

        assert_eq!(
            left_over(&plan, model),
            Some(format!("router_settings fallback to {target}"))
        );
    }

    #[rstest]
    fn a_fallback_to_a_native_group_leaves_its_group_native() {
        let plan = plan(&["backup"], "  fallbacks:\n    - claude: [other]\n");

        assert_eq!(left_over(&plan, "claude"), None);
    }

    #[rstest]
    #[case::a_name("  model_group_alias:\n    sonnet: claude\n", "sonnet", None)]
    #[case::a_hidden_alias(
        "  model_group_alias:\n    sonnet: {model: claude, hidden: true}\n",
        "sonnet",
        None
    )]
    #[case::of_a_group_left_to_python(
        "  model_group_alias:\n    sonnet: backup\n",
        "sonnet",
        Some("model_list.litellm_params.tags")
    )]
    fn an_alias_is_judged_as_the_group_it_names(
        #[case] router_settings: &str,
        #[case] model: &str,
        #[case] over: Option<&str>,
    ) {
        let plan = plan(&["backup"], router_settings);

        assert_eq!(left_over(&plan, model).as_deref(), over);
    }

    #[rstest]
    #[case::another_name("gpt")]
    #[case::a_pattern("anthropic/claude-haiku-5-5")]
    fn a_model_the_config_does_not_name_is_left_to_python(#[case] model: &str) {
        assert_eq!(
            plan(&[], "  num_retries: 1\n").model(model),
            Err(Left {
                why: Refused::Model,
                over: model
            })
        );
    }

    #[rstest]
    #[case::a_fallback_target_that_is_not_a_name("  fallbacks:\n    - claude: [{model: backup}]\n")]
    #[case::a_fallback_entry_that_is_not_a_mapping("  fallbacks: [claude]\n")]
    #[case::an_alias_without_a_model("  model_group_alias:\n    sonnet: {hidden: true}\n")]
    fn routes_written_in_another_shape_are_not_read(#[case] router_settings: &str) {
        assert_eq!(
            Routes::read(["claude"].into_iter(), &settings(router_settings)),
            None
        );
    }

    fn finding(setting: &str, leaves: Scope) -> Finding {
        Finding {
            setting: setting.into(),
            leaves,
        }
    }

    #[rstest]
    fn a_setting_that_leaves_everything_is_named_first() {
        let plan = Plan::new(
            vec![
                finding("model_list.blocked", Scope::Group("claude".into())),
                finding("litellm_settings.cache", Scope::Everything),
                finding("general_settings.custom_auth", Scope::Everything),
            ],
            Routes::default(),
        );

        assert_eq!(
            plan.everything(),
            Err(Left {
                why: Refused::Settings,
                over: "litellm_settings.cache"
            })
        );
    }

    #[rstest]
    #[case::read_as_python_does(&[], &[Trait::CodingAgent], None)]
    #[case::named_by_a_finding(
        &[Trait::CodingAgent],
        &[Trait::CodingAgent],
        Some((Refused::Headers, "litellm_settings.drop_params"))
    )]
    #[case::never_read(&[], &[Trait::UnreadKey], Some((Refused::Body, "unread_key")))]
    #[case::the_first_one_refused(
        &[],
        &[Trait::CodingAgent, Trait::ProxyHeader, Trait::Thread],
        Some((Refused::Headers, "proxy_header"))
    )]
    fn a_request_is_left_to_python_over_a_trait_the_native_app_does_not_read(
        #[case] named: &[Trait],
        #[case] carried: &[Trait],
        #[case] left: Option<(Refused, &str)>,
    ) {
        let findings = named
            .iter()
            .map(|named| finding("litellm_settings.drop_params", Scope::Asking(*named)))
            .collect();
        let plan = Plan::new(findings, Routes::default());

        assert_eq!(
            plan.carrying(carried)
                .err()
                .map(|left| (left.why, left.over)),
            left
        );
    }
}
