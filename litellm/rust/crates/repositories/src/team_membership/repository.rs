//! The team membership repository: a user's membership of a team with its member budget, read when
//! `get_team_membership` misses the auth cache.

use super::queries::{self, TeamMembership};
use crate::{Database, Side};
use litellm_db::Call;

const FIND_TEAM_MEMBERSHIP: Call = Call {
    call_type: "_fetch_team_membership_from_db",
    table: None,
    operation: None,
};

/// Reads `LiteLLM_TeamMembership` joined to the member's `LiteLLM_BudgetTable` row.
pub struct TeamMembershipRepository {
    database: Database,
}

impl TeamMembershipRepository {
    pub fn new(database: Database) -> Self {
        Self { database }
    }

    /// `user_id`'s membership of `team_id`, or `None` when the user is not a member.
    pub async fn find_with_budget(
        &self,
        user_id: &str,
        team_id: &str,
        side: Side,
    ) -> Result<Option<TeamMembership>, litellm_db::Error> {
        self.database
            .run(FIND_TEAM_MEMBERSHIP, || async {
                let mut session = self.database.session(side).await?;
                queries::find_team_membership(session.executor(), user_id, team_id).await
            })
            .await
    }
}
