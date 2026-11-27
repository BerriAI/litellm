//! A team membership as the database holds it, in the shape the auth cache holds it.

use std::sync::Arc;

use litellm_redis_store::auth_cache::{ExcludedIfNone, TeamMembershipEntry};
use litellm_repositories::{Side, TeamMembership, TeamMembershipRepository};

use crate::{Error, budget::budget_full_entry, error::within};

/// The team membership as the database holds it, in the shape the auth cache holds it: the row
/// with its budget.
pub struct TeamMembershipDatabase {
    memberships: Arc<TeamMembershipRepository>,
}

impl TeamMembershipDatabase {
    pub fn new(memberships: Arc<TeamMembershipRepository>) -> Self {
        Self { memberships }
    }

    /// `user_id`'s membership of `team_id`, or `None` when the user is not a member.
    pub async fn team_membership(
        &self,
        user_id: &str,
        team_id: &str,
        side: Side,
    ) -> Result<Option<TeamMembershipEntry>, Error> {
        self.memberships
            .find_with_budget(user_id, team_id, side)
            .await?
            .map(team_membership_entry)
            .transpose()
    }
}

/// What `LiteLLM_TeamMembership.model_dump(mode="json")` holds for `membership`, or the column
/// the model would refuse the row for.
fn team_membership_entry(membership: TeamMembership) -> Result<TeamMembershipEntry, Error> {
    let TeamMembership {
        user_id,
        team_id,
        budget_id,
        spend,
        rollover_credit,
        total_spend,
        litellm_budget_table,
    } = membership;
    Ok(TeamMembershipEntry {
        user_id,
        team_id,
        budget_id,
        spend: Some(spend),
        total_spend: Some(total_spend),
        rollover_credit: ExcludedIfNone(rollover_credit),
        litellm_budget_table: litellm_budget_table
            .map(|budget| budget_full_entry(budget.0))
            .transpose()
            .map_err(within("litellm_budget_table"))?,
        unread: Vec::new(),
    })
}
