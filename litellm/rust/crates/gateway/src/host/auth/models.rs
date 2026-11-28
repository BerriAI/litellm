//! The model lists the native host read for a caller, as the auth core's model rule and the
//! router read them.

use litellm_gateway_auth::Authenticated;
use litellm_gateway_auth_core::{
    Credential, Team,
    authz::{Layer, ModelList},
};

use super::NativeRows;

/// The model lists the native host read for `identity`: a key's own and its team's, and none for
/// the master key. A key that names an owner or anything else that carries a list was declined
/// when it was authenticated. A team the caller names with no row behind it has no list.
pub(crate) fn layers(identity: &Authenticated<NativeRows>) -> impl Iterator<Item = Layer<'_>> {
    let key = match &identity.credential {
        Credential::MasterKey | Credential::Jwt { .. } => None,
        Credential::Key { key, .. } => Some(Layer::Key(ModelList::read(key.models()))),
    };
    let team = identity
        .affiliation
        .holder
        .team()
        .and_then(team_list)
        .map(Layer::Team);
    key.into_iter().chain(team)
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
        Credential::MasterKey | Credential::Jwt { .. } => &[][..],
        Credential::Key { key, .. } => key.models(),
    };
    let team = match identity.affiliation.holder.team() {
        Some(Team::Row(record)) => &record.models[..],
        Some(Team::KeyCopy(_) | Team::Unstored(_)) | None => &[],
    };
    [key, team].concat()
}

#[cfg(test)]
mod tests {
    use litellm_config::LitellmRole;
    use litellm_gateway_auth_core::{
        Affiliation, Caller, Holder, JwtActor, JwtStamp, TeamCopy, Via, record::TeamRecord,
    };
    use rstest::rstest;

    use super::{super::Key, *};

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

    fn in_team(
        credential: Credential<NativeRows>,
        team: Option<Team<NativeRows>>,
    ) -> Authenticated<NativeRows> {
        Authenticated::from(Caller {
            credential,
            affiliation: Affiliation {
                holder: Holder::new(None, team.map(|team| (team, None))),
                org: None,
                project: None,
            },
            end_user: None,
        })
    }

    fn jwt_in(team: Option<Team<NativeRows>>) -> Authenticated<NativeRows> {
        let credential = Credential::Jwt {
            jwt: JwtStamp {
                claims: serde_json::Map::new(),
                user_email: None,
            },
            actor: JwtActor::User {
                role: LitellmRole::InternalUser,
            },
        };
        in_team(credential, team)
    }

    fn owned(models: &[&str]) -> Vec<String> {
        models.iter().map(|model| (*model).to_owned()).collect()
    }

    #[rstest]
    #[case::a_team_that_names_its_models(Some(row(&["claude"])), Some(&["claude"][..]))]
    #[case::a_team_with_no_list(Some(row(&[])), Some(&[][..]))]
    #[case::a_team_no_row_stands_behind(Some(Team::Unstored("team-1".to_owned())), None)]
    #[case::no_team(None, None)]
    fn a_jwt_is_checked_against_its_teams_list(
        #[case] team: Option<Team<NativeRows>>,
        #[case] listed: Option<&[&str]>,
    ) {
        let identity = jwt_in(team);
        let listed = listed.map(owned);

        assert_eq!(
            (
                layers(&identity).collect::<Vec<_>>(),
                allowed_models(&identity)
            ),
            (
                listed
                    .iter()
                    .map(|models| Layer::Team(ModelList::read(models)))
                    .collect(),
                listed.clone().unwrap_or_default()
            )
        );
    }

    #[test]
    fn a_keys_list_is_read_before_its_teams() {
        let credential = Credential::Key {
            key: Key::listing(&["claude"]),
            via: Via::OwnToken,
        };
        let identity = in_team(credential, Some(row(&["haiku"])));
        let (key, team) = (owned(&["claude"]), owned(&["haiku"]));

        assert_eq!(
            (
                layers(&identity).collect::<Vec<_>>(),
                allowed_models(&identity)
            ),
            (
                vec![
                    Layer::Key(ModelList::read(&key)),
                    Layer::Team(ModelList::read(&team))
                ],
                owned(&["claude", "haiku"])
            )
        );
    }
}
