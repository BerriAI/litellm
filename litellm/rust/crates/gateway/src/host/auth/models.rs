//! The model lists the native host read for a caller, as the auth core's model rule and the
//! router read them.

use litellm_gateway_auth::Authenticated;
use litellm_gateway_auth_core::{
    Credential, Team,
    authz::{Layer, ModelList},
};

use super::NativeRows;

/// The model lists read for `identity`: the key's, the team's and the team member's.
pub(crate) fn layers(identity: &Authenticated<NativeRows>) -> impl Iterator<Item = Layer<'_>> {
    let key = match &identity.credential {
        Credential::MasterKey => None,
        Credential::Key { key } => Some(Layer::Key(ModelList::read(key.models()))),
    };
    let team = identity
        .affiliation
        .holder
        .team()
        .and_then(team_list)
        .map(Layer::Team);
    let member = identity
        .affiliation
        .holder
        .membership()
        .map(|row| Layer::TeamMember(ModelList::read(&row.budget_extras.allowed_models)));
    key.into_iter().chain(team).chain(member)
}

/// The team's `models` column, read into a `ModelList`. A key's copy of its team is never served
/// natively, so it reaches no model.
fn team_list(team: &Team<NativeRows>) -> Option<ModelList<'_>> {
    match team {
        Team::Row(record) => Some(record.model_list()),
        Team::KeyCopy(_) => Some(ModelList::Unread),
        Team::Unstored(_) => None,
    }
}

/// The raw entries of the key's and then the team's `models`, special values included. The router
/// narrows a caller that reaches a model only through an access group by these, like a key's
/// `models` plus its `team_models` in `can_key_call_model`.
pub(crate) fn allowed_models(identity: &Authenticated<NativeRows>) -> Vec<String> {
    let key = match &identity.credential {
        Credential::MasterKey => &[][..],
        Credential::Key { key } => key.models(),
    };
    let team = match identity.affiliation.holder.team() {
        Some(Team::Row(record)) => &record.models[..],
        Some(Team::KeyCopy(_) | Team::Unstored(_)) | None => &[],
    };
    [key, team].concat()
}

#[cfg(test)]
mod tests {
    use litellm_gateway_auth_core::{TeamCopy, record::TeamRecord};
    use rstest::rstest;

    use super::*;

    fn row(models: &[&str]) -> Team<NativeRows> {
        Team::Row(TeamRecord {
            team_id: "team-1".to_owned(),
            team_alias: None,
            models: models.iter().map(|model| (*model).to_owned()).collect(),
            ..TeamRecord::default()
        })
    }

    fn key_copy() -> Team<NativeRows> {
        Team::KeyCopy(TeamCopy {
            team_id: "team-1".to_owned(),
            object_permission: None,
        })
    }

    #[rstest]
    #[case::a_team_with_no_list(row(&[]), Some(true))]
    #[case::a_team_that_names_its_models(row(&["claude"]), Some(false))]
    #[case::a_keys_copy_of_a_team(key_copy(), Some(false))]
    #[case::a_team_no_row_stands_behind(Team::Unstored("team-1".to_owned()), None)]
    fn only_a_teams_own_row_reaches_a_model(
        #[case] team: Team<NativeRows>,
        #[case] reaches_every_model: Option<bool>,
    ) {
        assert_eq!(
            team_list(&team).map(|list| list == ModelList::Every),
            reaches_every_model
        );
    }
}
