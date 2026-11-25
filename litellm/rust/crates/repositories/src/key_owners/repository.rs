//! The key owners repository: the user, team, membership, organization and project a key's auth checks read, in
//! one round trip, for the auth prefetch to warm their cache entries.

use litellm_db::Call;

use super::queries::{self, KeyOwners};

use crate::{Database, Side};

const PREFETCH_AUTH_OBJECTS: Call = Call {
    call_type: "prefetch_auth_objects",
    table: None,
    operation: None,
};

/// The ids of the rows to read. `None` skips that row.
#[derive(Clone, Copy, Debug, Default)]
pub struct KeyOwnerIds<'a> {
    pub user_id: Option<&'a str>,
    pub team_id: Option<&'a str>,
    pub membership_user_id: Option<&'a str>,
    pub organization_id: Option<&'a str>,
    pub project_id: Option<&'a str>,
}

/// Reads the user, team, team membership, organization and project rows in one statement.
pub struct KeyOwnersRepository {
    database: Database,
}

impl KeyOwnersRepository {
    pub fn new(database: Database) -> Self {
        Self { database }
    }

    /// The rows `ids` names; the membership is the one of `membership_user_id` in `team_id`.
    pub async fn find(&self, ids: KeyOwnerIds<'_>) -> Result<KeyOwners, litellm_db::Error> {
        self.database
            .run(PREFETCH_AUTH_OBJECTS, || async {
                let mut session = self.database.session(Side::Read).await?;
                queries::find_key_owners(
                    session.executor(),
                    ids.user_id,
                    ids.team_id,
                    ids.membership_user_id,
                    ids.organization_id,
                    ids.project_id,
                )
                .await
            })
            .await
    }
}
