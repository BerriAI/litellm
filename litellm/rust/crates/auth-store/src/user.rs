//! A user as the database holds it, in the shape the auth cache holds it.

use std::sync::Arc;

use litellm_redis_store::auth_cache::{ExcludedIfNone, OrganizationMembershipEntry, UserEntry};
use litellm_repositories::{
    Inserted, NewUserDefaults, OrganizationMembership, Resolution, Side, SignIn, SsoLink, User,
    UserRepository,
};

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

    /// The user `sign_in` resolves to by id, then by SSO id, then by email, linking its SSO id as
    /// `get_user_object` does, or `None` when no row matches. The entry is the row as the resolve
    /// read it, which Python caches, so a user that just had its own id linked as its SSO id still
    /// holds none.
    pub async fn sign_in(&self, sign_in: &SignIn, side: Side) -> Result<Option<SignedIn>, Error> {
        let Some(Resolution { user, sso_link }) = self.users.resolve_sign_in(sign_in, side).await?
        else {
            return Ok(None);
        };
        let owes_email = sign_in.user_email.is_some() && user.user_email.is_none();
        Ok(Some(SignedIn {
            user: user_entry(user)?,
            sso_link,
            owes_email,
        }))
    }

    /// Creates the user `sign_in` names with `defaults`, as `get_user_object` does under
    /// `user_id_upsert`, and returns the entry for the row as stored. A sign-in that loses to a
    /// concurrent one gets the entry for the row that won, or `Conflict(None)` when that row is
    /// gone again.
    pub async fn insert(
        &self,
        sign_in: &SignIn,
        defaults: &NewUserDefaults,
    ) -> Result<Inserted<UserEntry>, Error> {
        Ok(match self.users.insert(sign_in, defaults).await? {
            Inserted::Created(user) => Inserted::Created(user_entry(user)?),
            Inserted::Conflict(user) => Inserted::Conflict(user.map(user_entry).transpose()?),
        })
    }

    /// Stores `user_email` on `user_id` while the row has none, as `_backfill_null_user_email`
    /// does, and returns the entry for the row as stored, also when it already held an email.
    /// `None` when another user holds the email and `links_any_user` is off, or the row is gone.
    pub async fn backfill_email(
        &self,
        user_id: &str,
        user_email: &str,
        links_any_user: bool,
    ) -> Result<Option<UserEntry>, Error> {
        self.users
            .backfill_email_returning(user_id, user_email, links_any_user)
            .await?
            .map(user_entry)
            .transpose()
    }

    /// Sets the role of `user_id`, as the JWT role sync does, and returns the entry for the row as
    /// stored, or `None` when no row has that id.
    pub async fn set_role(
        &self,
        user_id: &str,
        user_role: &str,
    ) -> Result<Option<UserEntry>, Error> {
        self.users
            .set_role(user_id, user_role)
            .await?
            .map(user_entry)
            .transpose()
    }
}

/// The user a sign-in found, how its SSO id was linked, and whether the sign-in carries an email
/// the row lacks, which `_backfill_null_user_email` would store.
#[derive(Debug, PartialEq)]
pub struct SignedIn {
    pub user: UserEntry,
    pub sso_link: SsoLink,
    pub owes_email: bool,
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
