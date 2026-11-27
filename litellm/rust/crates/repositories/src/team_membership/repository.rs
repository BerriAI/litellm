//! The team membership repository: a user's membership of a team with its member budget, read when
//! `get_team_membership` misses the auth cache, and the write that adds a user to a team.

use super::queries::{self, TeamMembership};
use crate::{Database, MappedClaim, RawJson, Side, database::Connections};
use jiff::Timestamp;
use litellm_db::Call;
use serde_json::Value;

const FIND_TEAM_MEMBERSHIP: Call = Call {
    call_type: "_fetch_team_membership_from_db",
    table: None,
    operation: None,
};

const DELETE_TEAM_MEMBER: Call = Call {
    call_type: "team_member_delete",
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

/// A budget of the member's own with a cap, as `resolve_member_budget_id` creates for a default team's
/// `max_budget_in_team`, inserted in the member add's transaction. `allowed_models` is the team's default member
/// models, left at the column's default when the team has none.
#[derive(Clone, Debug)]
pub struct NewCappedMemberBudget {
    pub budget_id: String,
    pub max_budget: f64,
    pub allowed_models: Option<Vec<String>>,
    pub created_by: String,
    pub updated_by: String,
}

/// The role a roster entry gives its member, as `Member.role` takes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr, strum::VariantArray)]
pub enum RosterRole {
    #[strum(serialize = "user")]
    User,
    #[strum(serialize = "admin")]
    Admin,
}

/// The budget a member joining as `add_member_as` adds them is linked to: one `add_member` takes, or a new one
/// with a cap.
#[derive(Clone, Copy, Debug)]
pub enum JoinBudget<'a> {
    Member(MemberBudget<'a>),
    Capped(&'a NewCappedMemberBudget),
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

/// What a member removal changed: the roster before and after, the same when the roster did not list the user,
/// the hashes of the user's keys in the team it deleted, and the JWT claims mapped to them, which went with them.
#[derive(Clone, Debug, PartialEq)]
pub struct RemovedMember {
    pub roster_before: RawJson,
    pub roster_after: RawJson,
    pub deleted_tokens: Vec<String>,
    pub mapped_claims: Vec<MappedClaim>,
}

/// A roster the caller's check refused, read under the team's lock before anything was written, as the management
/// endpoints validate it before they write.
#[derive(Clone, Debug, PartialEq)]
pub struct RefusedRoster(pub RawJson);

/// How removing a user from a team ended. Only `Removed` wrote anything.
#[derive(Clone, Debug, PartialEq)]
pub enum MemberRemove {
    Removed(Box<RemovedMember>),
    /// Neither the roster nor the user's teams list the team, which `team_member_delete` refuses as "User not
    /// found in team".
    NotMember,
    TeamMissing,
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

    /// [`Self::add_member`] with the member's `role` and a budget that may carry a cap, as
    /// `add_new_user_to_default_team` adds a user created at sign-in to each team in
    /// `default_internal_user_params.teams`, in the same transaction and lock order. Nothing is written when
    /// `accepts` refuses the roster read under the lock.
    pub async fn add_member_as(
        &self,
        user_id: &str,
        team_id: &str,
        role: RosterRole,
        budget: JoinBudget<'_>,
        accepts: &(dyn Fn(&RawJson) -> bool + Sync),
    ) -> Result<Result<MemberAdd, RefusedRoster>, litellm_db::Error> {
        self.database
            .run(UPSERT_TEAM_MEMBERSHIP, || async {
                self.database
                    .transaction(async |transaction| {
                        add_member_as_once(transaction, user_id, team_id, role, budget, accepts)
                            .await
                    })
                    .await
            })
            .await
    }

    /// Removes `user_id` from `team_id` in one transaction, as `team_member_delete` does for a proxy admin: the
    /// user's entries off the roster, the team off the user's `teams`, the membership row, and the user's keys in
    /// the team, archived to `LiteLLM_DeletedVerificationToken` first. The member's budget row stays. It holds the
    /// team's advisory lock, as the management endpoints do, and writes nothing when `accepts` refuses the roster.
    pub async fn remove_member(
        &self,
        user_id: &str,
        team_id: &str,
        accepts: &(dyn Fn(&RawJson) -> bool + Sync),
    ) -> Result<Result<MemberRemove, RefusedRoster>, litellm_db::Error> {
        self.database
            .run(DELETE_TEAM_MEMBER, || async {
                self.database
                    .transaction(async |transaction| {
                        remove_member_once(transaction, user_id, team_id, accepts).await
                    })
                    .await
            })
            .await
    }
}

/// The steps every member add starts with, in the lock order the management endpoints use: the team's advisory
/// lock, the roster check, then the user row, locked, and the team on its `teams`. The user's email, or how the add
/// ended without writing.
async fn claim_member(
    connections: &mut impl Connections,
    user_id: &str,
    team_id: &str,
) -> Result<Result<Option<String>, MemberAdd>, litellm_db::Error> {
    connections
        .with(Side::Writer, async |db| {
            queries::lock_team(db, team_id).await
        })
        .await?;
    claim_locked_member(connections, user_id, team_id).await
}

/// [`claim_member`] once the team's advisory lock is held.
async fn claim_locked_member(
    connections: &mut impl Connections,
    user_id: &str,
    team_id: &str,
) -> Result<Result<Option<String>, MemberAdd>, litellm_db::Error> {
    let listed = connections
        .with(Side::Writer, async |db| {
            queries::roster_lists_user(db, team_id, user_id).await
        })
        .await?;
    match listed {
        None => return Ok(Err(MemberAdd::TeamMissing)),
        Some(true) => return Ok(Err(MemberAdd::AlreadyMember)),
        Some(false) => {}
    }
    let locked = connections
        .with(Side::Writer, async |db| {
            queries::lock_user(db, user_id).await
        })
        .await?;
    let Some(user_email) = locked else {
        return Ok(Err(MemberAdd::UserMissing));
    };
    connections
        .with(Side::Writer, async |db| {
            queries::add_team_to_user(db, user_id, team_id).await
        })
        .await?;
    Ok(Ok(user_email))
}

/// The steps of a member add, in the lock order the management endpoints use: the team's advisory lock, then the
/// user row.
pub(super) async fn add_member_once(
    connections: &mut impl Connections,
    user_id: &str,
    team_id: &str,
    budget: MemberBudget<'_>,
) -> Result<MemberAdd, litellm_db::Error> {
    let user_email = match claim_member(connections, user_id, team_id).await? {
        Ok(user_email) => user_email,
        Err(ended) => return Ok(ended),
    };
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

/// The steps of [`TeamMembershipRepository::add_member_as`]: those of a member add, with the capped budget and the
/// entry's role.
pub(super) async fn add_member_as_once(
    connections: &mut impl Connections,
    user_id: &str,
    team_id: &str,
    role: RosterRole,
    budget: JoinBudget<'_>,
    accepts: &(dyn Fn(&RawJson) -> bool + Sync),
) -> Result<Result<MemberAdd, RefusedRoster>, litellm_db::Error> {
    connections
        .with(Side::Writer, async |db| {
            queries::lock_team(db, team_id).await
        })
        .await?;
    let Some(roster) = connections
        .with(Side::Writer, async |db| {
            queries::find_roster(db, team_id).await
        })
        .await?
    else {
        return Ok(Ok(MemberAdd::TeamMissing));
    };
    if !accepts(&roster) {
        return Ok(Err(RefusedRoster(roster)));
    }
    let user_email = match claim_locked_member(connections, user_id, team_id).await? {
        Ok(user_email) => user_email,
        Err(ended) => return Ok(Ok(ended)),
    };
    let budget_id = match budget {
        JoinBudget::Member(MemberBudget::None) => None,
        JoinBudget::Member(MemberBudget::Existing(budget_id)) => Some(budget_id),
        JoinBudget::Member(MemberBudget::New(new_budget)) => {
            connections
                .with(Side::Writer, async |db| {
                    queries::insert_member_budget(db, new_budget, user_id, team_id).await
                })
                .await?;
            Some(new_budget.budget_id.as_str())
        }
        JoinBudget::Capped(new_budget) => {
            connections
                .with(Side::Writer, async |db| {
                    queries::insert_capped_member_budget(db, new_budget, user_id, team_id).await
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
            queries::append_member_as_to_roster(db, team_id, user_id, user_email.as_deref(), role)
                .await
        })
        .await?;
    Ok(Ok(MemberAdd::Added(Box::new(AddedMember {
        membership,
        roster,
    }))))
}

/// A roster entry as far as a removal matches it.
#[derive(serde::Deserialize)]
struct RosterEntry {
    user_id: Option<String>,
}

/// Whether `roster` holds an entry naming `user_id`, as `cleanup_members_with_roles` matches a removal by id.
fn lists_user(roster: &RawJson, user_id: &str) -> bool {
    serde_json::from_str::<Vec<RosterEntry>>(roster.0.get()).is_ok_and(|entries| {
        entries
            .iter()
            .any(|entry| entry.user_id.as_deref() == Some(user_id))
    })
}

/// The steps of a member removal, in the order `team_member_delete` makes them under the team's advisory lock.
pub(super) async fn remove_member_once(
    connections: &mut impl Connections,
    user_id: &str,
    team_id: &str,
    accepts: &(dyn Fn(&RawJson) -> bool + Sync),
) -> Result<Result<MemberRemove, RefusedRoster>, litellm_db::Error> {
    connections
        .with(Side::Writer, async |db| {
            queries::lock_team(db, team_id).await
        })
        .await?;
    let Some(roster_before) = connections
        .with(Side::Writer, async |db| {
            queries::find_roster(db, team_id).await
        })
        .await?
    else {
        return Ok(Ok(MemberRemove::TeamMissing));
    };
    if !accepts(&roster_before) {
        return Ok(Err(RefusedRoster(roster_before)));
    }
    let listed = lists_user(&roster_before, user_id);
    let holds_team = connections
        .with(Side::Writer, async |db| {
            queries::user_holds_team(db, user_id, team_id).await
        })
        .await?
        .unwrap_or(false);
    if !listed && !holds_team {
        return Ok(Ok(MemberRemove::NotMember));
    }
    let roster_after = match listed {
        true => {
            connections
                .with(Side::Writer, async |db| {
                    queries::remove_member_from_roster(db, team_id, user_id).await
                })
                .await?
        }
        false => roster_before.clone(),
    };
    connections
        .with(Side::Writer, async |db| {
            queries::remove_team_from_user(db, user_id, team_id).await
        })
        .await?;
    connections
        .with(Side::Writer, async |db| {
            queries::delete_team_membership(db, user_id, team_id).await
        })
        .await?;
    let deleted = connections
        .with(Side::Writer, async |db| {
            queries::archive_and_delete_member_keys(db, user_id, team_id).await
        })
        .await?;
    let mapped_claims = deleted
        .iter()
        .filter_map(|key| {
            Some(MappedClaim {
                jwt_issuer: key.jwt_issuer.clone()?,
                jwt_claim_name: key.jwt_claim_name.clone()?,
                jwt_claim_value: key.jwt_claim_value.clone()?,
            })
        })
        .collect();
    let mut deleted_tokens: Vec<String> = deleted.into_iter().map(|key| key.token).collect();
    deleted_tokens.dedup();
    Ok(Ok(MemberRemove::Removed(Box::new(RemovedMember {
        roster_before,
        roster_after,
        deleted_tokens,
        mapped_claims,
    }))))
}
