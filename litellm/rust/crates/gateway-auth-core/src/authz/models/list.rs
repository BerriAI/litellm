//! The one reading of a model list's sentinels, so every layer reads a list the same way.

use crate::authz::ModelAsk;

/// The entries that reach every model.
const EVERY_MODEL: [&str; 2] = ["*", "all-proxy-models"];
/// The entry that hands the decision to the team's list.
const TEAM_MODELS: &str = "all-team-models";
/// The entry that reaches no model, which the proxy gives a user who may only call through a team.
const NO_MODELS: &str = "no-default-models";

/// A model list as its row holds it, read once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelList<'a> {
    /// An empty list, or one that holds `*` or `all-proxy-models`.
    Every,
    Named(Names<'a>),
    /// The list says nothing of its own and the team's decides: a list that holds
    /// `all-team-models`, or a team member with no `allowed_models`. A key with no team to take
    /// from reaches every model. Only a layer the proxy reads under a team defers this way, and
    /// on any other the list reaches nothing.
    Inherit,
    /// A list that holds `no-default-models`.
    Nothing,
}

/// The names a list grants: its own entries, and the models its access groups add.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Names<'a> {
    models: &'a [String],
    granted: &'a [String],
}

impl<'a> Names<'a> {
    fn iter(&self) -> impl Iterator<Item = &'a String> + use<'a> {
        self.models.iter().chain(self.granted)
    }

    fn hold(&self, entry: &str) -> bool {
        self.iter().any(|name| name == entry)
    }

    /// A list reaches a model it names, by any of the ask's candidates or by an access group of
    /// their deployments.
    pub(super) fn reach(&self, ask: &ModelAsk<'_>) -> bool {
        self.iter()
            .any(|name| ask.candidates.contains(&name.as_str()) || ask.access_groups.contains(name))
    }
}

impl<'a> ModelList<'a> {
    pub fn read(models: &'a [String]) -> Self {
        Self::with_access_groups(models, &[])
    }

    /// `models` with `granted`, the model names the row's access groups add. A list reaches a
    /// model through either, the same for every layer.
    pub fn with_access_groups(models: &'a [String], granted: &'a [String]) -> Self {
        let names = Names { models, granted };
        if names.hold(NO_MODELS) {
            Self::Nothing
        } else if names.hold(TEAM_MODELS) {
            Self::Inherit
        } else if models.is_empty() || EVERY_MODEL.iter().any(|every| names.hold(every)) {
            Self::Every
        } else {
            Self::Named(names)
        }
    }

    /// A team member's `allowed_models`, where an empty list leaves the team's to decide.
    pub fn member(allowed: &'a [String]) -> Self {
        match allowed {
            [] => Self::Inherit,
            allowed => Self::read(allowed),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::authz::models::owned;

    #[derive(Debug, PartialEq, Eq)]
    enum Read {
        Every,
        Named,
        Inherit,
        Nothing,
    }

    fn kind(list: ModelList<'_>) -> Read {
        match list {
            ModelList::Every => Read::Every,
            ModelList::Named(_) => Read::Named,
            ModelList::Inherit => Read::Inherit,
            ModelList::Nothing => Read::Nothing,
        }
    }

    #[rstest]
    #[case::an_empty_list(&[], &[], Read::Every)]
    #[case::a_star(&["gpt", "*"], &[], Read::Every)]
    #[case::every_proxy_model(&["all-proxy-models"], &[], Read::Every)]
    #[case::names(&["gpt", "claude"], &[], Read::Named)]
    #[case::a_pattern_is_only_a_name(&["anthropic/*"], &[], Read::Named)]
    #[case::the_teams_models(&["all-team-models"], &[], Read::Inherit)]
    #[case::the_teams_models_beside_a_star(&["*", "all-team-models"], &[], Read::Inherit)]
    #[case::no_default_models(&["no-default-models", "gpt"], &[], Read::Nothing)]
    #[case::no_default_models_beside_the_teams(
        &["all-team-models", "no-default-models"],
        &[],
        Read::Nothing
    )]
    #[case::an_empty_list_with_access_groups(&[], &["gpt"], Read::Every)]
    #[case::an_access_group_that_grants_every_model(&["gpt"], &["*"], Read::Every)]
    fn a_list_is_read_by_its_sentinels(
        #[case] models: &[&str],
        #[case] granted: &[&str],
        #[case] read: Read,
    ) {
        let (models, granted) = (owned(models), owned(granted));

        assert_eq!(kind(ModelList::with_access_groups(&models, &granted)), read);
    }

    #[rstest]
    #[case::no_allowed_models(&[], Read::Inherit)]
    #[case::allowed_models(&["gpt"], Read::Named)]
    #[case::every_model(&["*"], Read::Every)]
    fn a_member_with_no_allowed_models_leaves_the_team_to_decide(
        #[case] allowed: &[&str],
        #[case] read: Read,
    ) {
        let allowed = owned(allowed);

        assert_eq!(kind(ModelList::member(&allowed)), read);
    }

    #[rstest]
    #[case::the_name_asked_for(&["claude"], &[], &["claude"], &[], true)]
    #[case::another_name(&["claude"], &[], &["gpt"], &[], false)]
    #[case::a_name_that_only_starts_the_same(&["claude"], &[], &["claude-down"], &[], false)]
    #[case::a_later_candidate(&["claude"], &[], &["sonnet", "claude"], &[], true)]
    #[case::an_access_group_of_the_model(&["fast"], &[], &["claude"], &["haiku", "fast"], true)]
    #[case::an_access_group_of_another_model(&["haiku"], &[], &["gpt"], &["fast"], false)]
    #[case::a_name_its_access_groups_add(&["gpt"], &["claude"], &["claude"], &[], true)]
    fn a_named_list_reaches_a_model_by_a_candidate_or_an_access_group(
        #[case] models: &[&str],
        #[case] granted: &[&str],
        #[case] candidates: &[&str],
        #[case] access_groups: &[&str],
        #[case] reached: bool,
    ) {
        let (models, granted) = (owned(models), owned(granted));
        let access_groups = owned(access_groups);
        let ask = ModelAsk {
            name: candidates[0],
            candidates,
            access_groups: &access_groups,
            team_alias: false,
        };
        let ModelList::Named(names) = ModelList::with_access_groups(&models, &granted) else {
            panic!("expected a named list");
        };

        assert_eq!(names.reach(&ask), reached);
    }
}
