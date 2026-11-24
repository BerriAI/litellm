//! The team membership repository: a user's membership of a team with its member budget, read when
//! `get_team_membership` misses the auth cache, and the write that adds a user to a team.

use super::queries::{self, TeamMembership};
use crate::{Database, Side, database::Connections};
use jiff::Timestamp;
use litellm_db::Call;
use serde_json::Value;

const FIND_TEAM_MEMBERSHIP: Call = Call {
    call_type: "_fetch_team_membership_from_db",
    table: None,
    operation: None,
};

const UPSERT_TEAM_MEMBERSHIP: Call = Call {
    call_type: "upsert",
    table: Some("LiteLLM_TeamMembership"),
    operation: Some("upsert"),
};

/// A budget of the member's own, inserted in the member add's transaction so it lands only with the membership.
#[derive(Clone, Debug)]
pub struct NewMemberBudget {
    pub budget_id: String,
    pub allowed_models: Vec<String>,
    pub budget_duration: Option<String>,
    pub budget_reset_at: Option<Timestamp>,
    pub created_by: String,
    pub updated_by: String,
}

/// The budget a new membership is linked to.
#[derive(Clone, Copy, Debug)]
pub enum MemberBudget<'a> {
    None,
    /// A budget already stored, such as the team's shared default.
    Existing(&'a str),
    New(&'a NewMemberBudget),
}

/// What a member add stored: the membership row, which is an earlier one left as it was when the user had it
/// already, and the team's roster with the new entry last.
#[derive(Clone, Debug, PartialEq)]
pub struct AddedMember {
    pub membership: TeamMembership,
    pub roster: Value,
}

/// How adding a user to a team ended. Only `Added` wrote anything.
#[derive(Clone, Debug, PartialEq)]
pub enum MemberAdd {
    Added(Box<AddedMember>),
    /// The team's roster already lists the user's id or email.
    AlreadyMember,
    TeamMissing,
    UserMissing,
}

/// Reads and writes `LiteLLM_TeamMembership` joined to the member's `LiteLLM_BudgetTable` row.
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

    /// Adds `user_id` to `team_id` as a `user` in one transaction: the team on the user's `teams`, the member's
    /// budget when it is a new one, the membership row linked to it, and the user on the team's roster. It holds
    /// the advisory lock the management endpoints take on the team, so neither side overwrites the other's roster.
    pub async fn add_member(
        &self,
        user_id: &str,
        team_id: &str,
        budget: MemberBudget<'_>,
    ) -> Result<MemberAdd, litellm_db::Error> {
        self.database
            .run(UPSERT_TEAM_MEMBERSHIP, || async {
                self.database
                    .transaction(async |transaction| {
                        add_member_once(transaction, user_id, team_id, budget).await
                    })
                    .await
            })
            .await
    }
}

/// The steps of a member add, in the lock order the management endpoints use: the team's advisory lock, then the
/// user row.
pub(super) async fn add_member_once(
    connections: &mut impl Connections,
    user_id: &str,
    team_id: &str,
    budget: MemberBudget<'_>,
) -> Result<MemberAdd, litellm_db::Error> {
    connections
        .with(Side::Writer, async |db| {
            queries::lock_team(db, team_id).await
        })
        .await?;
    let listed = connections
        .with(Side::Writer, async |db| {
            queries::roster_lists_user(db, team_id, user_id).await
        })
        .await?;
    match listed {
        None => return Ok(MemberAdd::TeamMissing),
        Some(true) => return Ok(MemberAdd::AlreadyMember),
        Some(false) => {}
    }
    let locked = connections
        .with(Side::Writer, async |db| {
            queries::lock_user(db, user_id).await
        })
        .await?;
    let Some(user_email) = locked else {
        return Ok(MemberAdd::UserMissing);
    };
    connections
        .with(Side::Writer, async |db| {
            queries::add_team_to_user(db, user_id, team_id).await
        })
        .await?;
    let budget_id = match budget {
        MemberBudget::None => None,
        MemberBudget::Existing(budget_id) => Some(budget_id),
        MemberBudget::New(new_budget) => {
            connections
                .with(Side::Writer, async |db| {
                    queries::insert_member_budget(db, new_budget, user_id, team_id).await
                })
                .await?;
            Some(new_budget.budget_id.as_str())
        }
    };
    let membership = connections
        .with(Side::Writer, async |db| {
            queries::insert_team_membership(db, user_id, team_id, budget_id).await
        })
        .await?;
    let roster = connections
        .with(Side::Writer, async |db| {
            queries::append_member_to_roster(db, team_id, user_id, user_email.as_deref()).await
        })
        .await?;
    Ok(MemberAdd::Added(Box::new(AddedMember {
        membership,
        roster,
    })))
}
