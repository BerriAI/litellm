//! A user as the database holds it, in the shape the auth cache holds it.

use std::sync::Arc;

use litellm_redis_store::auth_cache::{ExcludedIfNone, OrganizationMembershipEntry, UserEntry};
use litellm_repositories::{OrganizationMembership, Side, SignIn, User, UserRepository};

use crate::{
    Error,
    budget_limit::windows,
    column::{instant, nullable, required, time},
    error::within,
};

/// The user as the database holds it, in the shape the auth cache holds it: the row with its
/// organization memberships, and nothing else nested.
pub struct UserDatabase {
    users: Arc<UserRepository>,
}

impl UserDatabase {
    pub fn new(users: Arc<UserRepository>) -> Self {
        Self { users }
    }

    /// The user `user_id`, or `None` when no row has that id. It is the repository's resolve of a
    /// sign-in that carries only the id, which `get_user_object` makes for a plain lookup: with no
    /// SSO id or email there is nothing to fall back to or link, so it only reads. The object
    /// permission the user points at is not read.
    pub async fn user(&self, user_id: &str, side: Side) -> Result<Option<UserEntry>, Error> {
        self.users
            .resolve(&SignIn::by_id(user_id), side)
            .await?
            .map(user_entry)
            .transpose()
    }
}

fn organization_membership_entry(
    membership: OrganizationMembership,
) -> Result<OrganizationMembershipEntry, Error> {
    let OrganizationMembership {
        user_id,
        organization_id,
        user_role,
        spend,
        budget_id,
        created_at,
        updated_at,
    } = membership;
    Ok(OrganizationMembershipEntry {
        user_id,
        organization_id,
        user_role,
        spend: required("spend", spend)?,
        budget_id,
        created_at: instant("created_at", required("created_at", created_at)?)?,
        updated_at: instant("updated_at", required("updated_at", updated_at)?)?,
        user: None,
        litellm_budget_table: None,
        user_email: None,
        unread: Vec::new(),
    })
}

fn user_entry(user: User) -> Result<UserEntry, Error> {
    let User {
        user_id,
        user_alias,
        team_id,
        sso_user_id,
        organization_id,
        object_permission_id,
        password_reset_required,
        last_breach_check_at,
        teams,
        user_role,
        max_budget,
        rollover_max_budget,
        rollover_credit,
        budget_limits,
        spend,
        user_email,
        models,
        metadata,
        max_parallel_requests,
        tpm_limit,
        rpm_limit,
        budget_duration,
        budget_reset_at,
        allowed_cache_controls,
        policies,
        model_spend,
        model_max_budget,
        created_at,
        updated_at,
        organization_memberships,
    } = user;
    Ok(UserEntry {
        user_id,
        user_alias,
        team_id,
        sso_user_id,
        organization_id,
        object_permission_id,
        password_reset_required,
        last_breach_check_at: time("last_breach_check_at", last_breach_check_at)?,
        teams,
        user_role,
        max_budget,
        rollover_max_budget: ExcludedIfNone(rollover_max_budget),
        rollover_credit: ExcludedIfNone(rollover_credit),
        budget_limits: windows(nullable("budget_limits", budget_limits.as_ref())?),
        spend,
        user_email,
        models,
        metadata: nullable("metadata", Some(&metadata))?,
        max_parallel_requests: max_parallel_requests.map(i64::from),
        tpm_limit,
        rpm_limit,
        budget_duration,
        budget_reset_at: time("budget_reset_at", budget_reset_at)?,
        allowed_cache_controls,
        policies,
        model_spend: nullable("model_spend", Some(&model_spend))?,
        model_max_budget: nullable("model_max_budget", Some(&model_max_budget))?,
        created_at: time("created_at", created_at)?,
        updated_at: time("updated_at", updated_at)?,
        organization_memberships: Some(
            organization_memberships
                .0
                .into_iter()
                .map(organization_membership_entry)
                .collect::<Result<_, _>>()
                .map_err(within("organization_memberships"))?,
        ),
        object_permission: None,
        unread: Vec::new(),
    })
}
