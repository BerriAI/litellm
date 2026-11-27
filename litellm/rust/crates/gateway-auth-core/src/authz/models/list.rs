//! A model list read once, entry by entry through [`ModelEntry`], so no layer and no host
//! compares an entry with a special spelling itself.

use litellm_router_types::{ModelEntry, SpecialModel};

use crate::authz::ModelAsk;

/// A model list as its row holds it, read once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelList<'a> {
    /// An empty list, or one that holds `*` or `all-proxy-models`.
    Every,
    Named(&'a [String]),
    /// A list with an entry the rule does not read yet: `all-team-models`, `no-default-models`,
    /// or a pattern such as `anthropic/*` the proxy matches against the model's provider. A host
    /// declines a caller with one at authentication, and it reaches no model.
    Unread,
}

impl<'a> ModelList<'a> {
    pub fn read(models: &'a [String]) -> Self {
        let entries = || models.iter().map(|model| ModelEntry::of(model));
        if !entries().all(is_read) {
            Self::Unread
        } else if models.is_empty()
            || entries().any(|entry| entry == ModelEntry::Special(SpecialModel::Every))
        {
            Self::Every
        } else {
            Self::Named(models)
        }
    }

    /// Whether the list reaches `ask`: every model, or one it names by a candidate or by an
    /// access group of its deployments.
    pub(super) fn reaches(&self, ask: &ModelAsk<'_>) -> bool {
        match self {
            Self::Every => true,
            Self::Named(models) => models.iter().any(|model| {
                ask.candidates.contains(&model.as_str()) || ask.access_groups.contains(model)
            }),
            Self::Unread => false,
        }
    }
}

/// Whether the rule reads `entry` yet. The others need the caller's team or user, or the
/// model's provider.
fn is_read(entry: ModelEntry<'_>) -> bool {
    match entry {
        ModelEntry::Special(SpecialModel::Every) | ModelEntry::Name(_) => true,
        ModelEntry::Special(SpecialModel::TeamModels | SpecialModel::NoDefaultModels)
        | ModelEntry::Pattern(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::authz::models::owned;

    #[rstest]
    #[case::an_empty_list(&[], ModelList::Every)]
    #[case::a_star(&["gpt", "*"], ModelList::Every)]
    #[case::every_proxy_model(&["all-proxy-models"], ModelList::Every)]
    #[case::the_teams_models(&["all-team-models"], ModelList::Unread)]
    #[case::the_teams_models_beside_a_star(&["*", "all-team-models"], ModelList::Unread)]
    #[case::no_default_models(&["gpt", "no-default-models"], ModelList::Unread)]
    #[case::a_pattern(&["anthropic/*"], ModelList::Unread)]
    fn a_list_is_read_by_its_special_values(#[case] models: &[&str], #[case] read: ModelList<'_>) {
        assert_eq!(ModelList::read(&owned(models)), read);
    }
}
