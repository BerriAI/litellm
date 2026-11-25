//! What a caller belongs to: its user, team and membership, organization and project. The key
//! path and the JWT path both build it, and every struct that says who a request is for holds one
//! instead of declaring the slots again. The end user is named by the request, not by the caller,
//! so it sits beside a `Belongs`, never in it.

use crate::{Rows, record::Record};

/// No `Default`, so each place that builds one names every slot. An `Option` here is a name the
/// caller does not have, never a row a host did not read.
#[derive(Clone, Debug, PartialEq)]
pub struct Belongs<R: Rows> {
    pub holder: Holder<R>,
    /// Only a JWT's for now: the key path reads no organization.
    pub org: Option<Named<R::Org>>,
    pub project: Option<Named<R::Project>>,
}

impl<R: Rows> Belongs<R> {
    /// What the master key belongs to: nothing.
    pub fn nobody() -> Self {
        Self {
            holder: Holder::Nobody,
            org: None,
            project: None,
        }
    }
}

/// The user, team and membership a caller holds, in the combinations that can happen.
#[derive(Clone, Debug, PartialEq)]
pub enum Holder<R: Rows> {
    Nobody,
    User(Named<R::User>),
    Team(Team<R>),
    /// `membership` is the user's row in `team`, which only a JWT reads for now.
    Member {
        user: Named<R::User>,
        team: Team<R>,
        membership: Option<R::Membership>,
    },
}

impl<R: Rows> Holder<R> {
    /// A membership comes only with its team, and is kept only when there is a user too.
    pub fn new(
        user: Option<Named<R::User>>,
        team: Option<(Team<R>, Option<R::Membership>)>,
    ) -> Self {
        match (user, team) {
            (None, None) => Self::Nobody,
            (Some(user), None) => Self::User(user),
            (None, Some((team, _))) => Self::Team(team),
            (Some(user), Some((team, membership))) => Self::Member {
                user,
                team,
                membership,
            },
        }
    }

    pub fn user(&self) -> Option<&Named<R::User>> {
        match self {
            Self::User(user) | Self::Member { user, .. } => Some(user),
            Self::Nobody | Self::Team(_) => None,
        }
    }

    pub fn into_user(self) -> Option<Named<R::User>> {
        match self {
            Self::User(user) | Self::Member { user, .. } => Some(user),
            Self::Nobody | Self::Team(_) => None,
        }
    }

    pub fn team(&self) -> Option<&Team<R>> {
        match self {
            Self::Team(team) | Self::Member { team, .. } => Some(team),
            Self::Nobody | Self::User(_) => None,
        }
    }

    pub fn membership(&self) -> Option<&R::Membership> {
        match self {
            Self::Member { membership, .. } => membership.as_ref(),
            Self::Nobody | Self::User(_) | Self::Team(_) => None,
        }
    }
}

/// The team a caller acts in.
#[derive(Clone, Debug, PartialEq)]
pub enum Team<R: Rows> {
    /// Only a JWT's: a team the token names that no row stands behind. Unlike a `KeyCopy`, it
    /// never becomes a rebuilt team object.
    Unstored(String),
    Row(R::Team),
    KeyCopy(TeamCopy<R>),
}

impl<R: Rows> Team<R> {
    pub fn of(id: Option<String>, row: Option<R::Team>) -> Option<Self> {
        match (row, id) {
            (Some(row), _) => Some(Self::Row(row)),
            (None, Some(id)) => Some(Self::Unstored(id)),
            (None, None) => None,
        }
    }
}

impl<R: Rows<Team: Record>> Team<R> {
    pub fn id(&self) -> &str {
        match self {
            Self::Unstored(team_id) => team_id,
            Self::Row(team) => team.id(),
            Self::KeyCopy(copy) => &copy.team_id,
        }
    }
}

/// Only a key's: the UI team, or a team the host did not report. The key's own copy of the team's
/// values stands, which is Python's `LiteLLM_TeamRebuiltFromToken`.
#[derive(Clone, Debug, PartialEq)]
pub struct TeamCopy<R: Rows> {
    pub team_id: String,
    pub object_permission: Option<R::ObjectPermission>,
}

/// Something a caller names: its id when no row stands behind it, else the row, which holds the
/// id.
#[derive(Clone, Debug, PartialEq)]
pub enum Named<T> {
    Unstored(String),
    Stored(T),
}

impl<T> Named<T> {
    pub fn of(id: Option<String>, row: Option<T>) -> Option<Self> {
        match (row, id) {
            (Some(row), _) => Some(Self::Stored(row)),
            (None, Some(id)) => Some(Self::Unstored(id)),
            (None, None) => None,
        }
    }

    pub fn stored(&self) -> Option<&T> {
        match self {
            Self::Stored(row) => Some(row),
            Self::Unstored(_) => None,
        }
    }

    pub fn into_stored(self) -> Option<T> {
        match self {
            Self::Stored(row) => Some(row),
            Self::Unstored(_) => None,
        }
    }
}

impl<T: Record> Named<T> {
    pub fn id(&self) -> &str {
        match self {
            Self::Unstored(id) => id,
            Self::Stored(row) => row.id(),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::testing::Labels;

    fn user() -> Option<Named<String>> {
        Some(Named::Unstored("u1".to_owned()))
    }

    fn team() -> Option<(Team<Labels>, Option<String>)> {
        Some((
            Team::Row("team t".to_owned()),
            Some("membership".to_owned()),
        ))
    }

    #[rstest]
    #[case::nobody(None, None, Holder::Nobody)]
    #[case::user(user(), None, Holder::User(Named::Unstored("u1".to_owned())))]
    #[case::team(None, team(), Holder::Team(Team::Row("team t".to_owned())))]
    #[case::member(user(), team(), Holder::Member {
        user: Named::Unstored("u1".to_owned()),
        team: Team::Row("team t".to_owned()),
        membership: Some("membership".to_owned()),
    })]
    fn a_membership_is_kept_only_for_a_user_in_a_team(
        #[case] user: Option<Named<String>>,
        #[case] team: Option<(Team<Labels>, Option<String>)>,
        #[case] expected: Holder<Labels>,
    ) {
        assert_eq!(Holder::new(user, team), expected);
    }

    #[rstest]
    #[case::row_over_id(Some("u1"), Some("row"), Some(Named::Stored("row".to_owned())))]
    #[case::id_without_row(Some("u1"), None, Some(Named::Unstored("u1".to_owned())))]
    #[case::neither(None, None, None)]
    fn a_name_is_its_row_when_one_is_stored(
        #[case] id: Option<&str>,
        #[case] row: Option<&str>,
        #[case] expected: Option<Named<String>>,
    ) {
        assert_eq!(
            Named::of(id.map(str::to_owned), row.map(str::to_owned)),
            expected
        );
    }
}
