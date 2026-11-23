//! Whether the model lists a host read for a caller reach the model it asks for. The host
//! supplies each list it read as a [`Layer`] and the [`ModelAsk`] it resolved the name to, and
//! decides nothing itself.

mod aliases;
mod list;

pub use aliases::{Aliases, Candidates, candidates};
pub use list::{ModelList, Names};

use super::{AuthzDenial, DeniedList, ModelAsk};

/// One model list the host read for a caller, named by the row that holds it. A request has to
/// be reached by every layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layer<'a> {
    Key(ModelList<'a>),
    Team(ModelList<'a>),
    /// The caller's membership of its team.
    Member(ModelList<'a>),
    User(ModelList<'a>),
    Project(ModelList<'a>),
    Org(ModelList<'a>),
    EndUser(ModelList<'a>),
}

impl<'a> Layer<'a> {
    pub fn list(&self) -> ModelList<'a> {
        match *self {
            Self::Key(list)
            | Self::Team(list)
            | Self::Member(list)
            | Self::User(list)
            | Self::Project(list)
            | Self::Org(list)
            | Self::EndUser(list) => list,
        }
    }

    /// The team grants a name by aliasing it, to itself and to its keys. No other layer is the
    /// team's to answer for.
    fn takes_team_aliases(&self) -> bool {
        match self {
            Self::Key(_) | Self::Team(_) => true,
            Self::Member(_)
            | Self::User(_)
            | Self::Project(_)
            | Self::Org(_)
            | Self::EndUser(_) => false,
        }
    }

    /// A list hands the decision to the team's only on a layer the proxy reads under a team:
    /// the key's, the team's own, a member's and a project's. On a user's, an organization's
    /// or an end user's list the proxy reads `all-team-models` as a model's name, which
    /// reaches nothing.
    fn defers_to_the_team(&self) -> bool {
        match self {
            Self::Key(_) | Self::Team(_) | Self::Member(_) | Self::Project(_) => true,
            Self::User(_) | Self::Org(_) | Self::EndUser(_) => false,
        }
    }

    fn reaches(&self, ask: &ModelAsk<'_>) -> bool {
        if self.takes_team_aliases() && ask.team_alias {
            return true;
        }
        match self.list() {
            ModelList::Every => true,
            ModelList::Inherit => self.defers_to_the_team(),
            ModelList::Named(names) => names.reach(ask),
            ModelList::Nothing => false,
        }
    }

    fn denial(&self, ask: &ModelAsk<'_>) -> AuthzDenial {
        let list = match self {
            Self::Key(_) => DeniedList::Key,
            Self::Team(_) => DeniedList::Team,
            Self::Member(_) => DeniedList::Member,
            Self::User(ModelList::Nothing) => DeniedList::UserWithoutDefaults,
            Self::User(_) => DeniedList::User,
            Self::Project(_) => DeniedList::Project,
            Self::Org(_) => DeniedList::Org,
            Self::EndUser(_) => {
                return AuthzDenial::CustomerModelAccess {
                    model: ask.name.to_owned(),
                };
            }
        };
        AuthzDenial::ModelAccess {
            list,
            model: ask.name.to_owned(),
        }
    }
}

/// Whether every one of `layers` reaches `ask`, with the denial of the first that does not. No
/// layer at all, as the master key has, reaches every model.
pub fn authorize<'a>(
    layers: impl IntoIterator<Item = Layer<'a>>,
    ask: &ModelAsk<'_>,
) -> Result<(), AuthzDenial> {
    layers.into_iter().try_for_each(|layer| match layer {
        layer if layer.reaches(ask) => Ok(()),
        layer => Err(layer.denial(ask)),
    })
}

#[cfg(test)]
pub(crate) fn owned(names: &[&str]) -> Vec<String> {
    names.iter().map(|name| (*name).to_owned()).collect()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    const GPT: ModelAsk<'static> = ModelAsk {
        name: "gpt",
        candidates: &["gpt"],
        access_groups: &[],
        team_alias: false,
    };

    fn refused(list: DeniedList) -> AuthzDenial {
        AuthzDenial::ModelAccess {
            list,
            model: "gpt".to_owned(),
        }
    }

    #[rstest]
    #[case::key(Layer::Key, refused(DeniedList::Key))]
    #[case::team(Layer::Team, refused(DeniedList::Team))]
    #[case::member(Layer::Member, refused(DeniedList::Member))]
    #[case::user(Layer::User, refused(DeniedList::User))]
    #[case::project(Layer::Project, refused(DeniedList::Project))]
    #[case::org(Layer::Org, refused(DeniedList::Org))]
    #[case::end_user(Layer::EndUser, AuthzDenial::CustomerModelAccess { model: "gpt".to_owned() })]
    fn a_layer_that_names_other_models_refuses_as_the_proxy_does_for_it(
        #[case] layer: fn(ModelList<'static>) -> Layer<'static>,
        #[case] denial: AuthzDenial,
    ) {
        let (claude, gpt) = (owned(&["claude"]).leak(), owned(&["gpt"]).leak());

        assert_eq!(
            (
                authorize([layer(ModelList::read(claude))], &GPT),
                authorize([layer(ModelList::read(gpt))], &GPT)
            ),
            (Err(denial), Ok(()))
        );
    }

    #[rstest]
    #[case::key(Layer::Key, true)]
    #[case::team(Layer::Team, true)]
    #[case::member(Layer::Member, true)]
    #[case::project(Layer::Project, true)]
    #[case::user(Layer::User, false)]
    #[case::org(Layer::Org, false)]
    #[case::end_user(Layer::EndUser, false)]
    fn only_a_layer_under_a_team_leaves_its_list_to_the_teams(
        #[case] layer: fn(ModelList<'static>) -> Layer<'static>,
        #[case] reached: bool,
    ) {
        let models = owned(&["all-team-models"]).leak();

        assert_eq!(
            authorize([layer(ModelList::read(models))], &GPT).is_ok(),
            reached
        );
    }

    #[rstest]
    #[case::every_model(&[], Ok(()))]
    #[case::no_default_models(&["no-default-models"], Err(refused(DeniedList::Team)))]
    fn a_layer_is_judged_by_how_its_list_reads(
        #[case] models: &[&str],
        #[case] authorized: Result<(), AuthzDenial>,
    ) {
        let models = owned(models);

        assert_eq!(
            authorize([Layer::Team(ModelList::read(&models))], &GPT),
            authorized
        );
    }

    #[rstest]
    fn a_user_with_no_default_models_is_refused_with_the_keys_type() {
        let models = owned(&["no-default-models"]);

        assert_eq!(
            authorize([Layer::User(ModelList::read(&models))], &GPT),
            Err(refused(DeniedList::UserWithoutDefaults))
        );
    }

    #[rstest]
    #[case::every_layer_reaches_it(&["gpt"], &["gpt"], &["gpt"], Ok(()))]
    #[case::the_last_does_not(&["gpt"], &["gpt"], &["claude"], Err(refused(DeniedList::Member)))]
    #[case::one_in_the_middle_does_not(&["gpt"], &["claude"], &["gpt"], Err(refused(DeniedList::Team)))]
    #[case::the_first_of_two_that_do_not(&["claude"], &["claude"], &["gpt"], Err(refused(DeniedList::Key)))]
    #[case::a_key_that_takes_the_teams_list_is_the_teams_to_refuse(
        &["all-team-models"], &["claude"], &[], Err(refused(DeniedList::Team))
    )]
    #[case::a_key_that_takes_the_teams_list_is_the_teams_to_allow(
        &["all-team-models"], &["gpt"], &[], Ok(())
    )]
    fn a_request_has_to_be_reached_by_every_layer(
        #[case] key: &[&str],
        #[case] team: &[&str],
        #[case] member: &[&str],
        #[case] authorized: Result<(), AuthzDenial>,
    ) {
        let (key, team, member) = (owned(key), owned(team), owned(member));
        let layers = [
            Layer::Key(ModelList::read(&key)),
            Layer::Team(ModelList::read(&team)),
            Layer::Member(ModelList::member(&member)),
        ];

        assert_eq!(authorize(layers, &GPT), authorized);
    }

    #[rstest]
    #[case::key(Layer::Key, true)]
    #[case::team(Layer::Team, true)]
    #[case::member(Layer::Member, false)]
    #[case::user(Layer::User, false)]
    #[case::project(Layer::Project, false)]
    #[case::org(Layer::Org, false)]
    #[case::end_user(Layer::EndUser, false)]
    fn only_the_key_and_the_team_reach_a_name_the_team_aliases(
        #[case] layer: fn(ModelList<'static>) -> Layer<'static>,
        #[case] reached: bool,
    ) {
        let claude = owned(&["claude"]).leak();
        let aliased = ModelAsk {
            team_alias: true,
            ..GPT
        };

        assert_eq!(
            authorize([layer(ModelList::read(claude))], &aliased).is_ok(),
            reached
        );
    }

    #[rstest]
    fn a_caller_with_no_layer_reaches_every_model() {
        assert_eq!(authorize([], &GPT), Ok(()));
    }
}
