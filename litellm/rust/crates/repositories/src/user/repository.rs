//! The user repository: resolves a sign-in to its `LiteLLM_UserTable` row by id, then by SSO id, then by email,
//! linking the SSO id the way the Python `get_user_object` did, and backfills a missing email.

use jiff::Timestamp;
use litellm_db::Call;
use serde_json::Value;

use super::queries::{self, User};
use crate::{Database, Inserted, Side, database::Connections};

const GET_USER_OBJECT: Call = Call {
    call_type: "get_user_object",
    table: None,
    operation: None,
};

const FIND_FIRST_USER: Call = Call {
    call_type: "find_first",
    table: Some("LiteLLM_UserTable"),
    operation: Some("select"),
};

const UPDATE_MANY_USERS: Call = Call {
    call_type: "update_many",
    table: Some("LiteLLM_UserTable"),
    operation: Some("update"),
};

const UPDATE_USER: Call = Call {
    call_type: "update",
    table: Some("LiteLLM_UserTable"),
    operation: Some("update"),
};

const CREATE_USER: Call = Call {
    call_type: "create",
    table: Some("LiteLLM_UserTable"),
    operation: Some("insert"),
};

/// Who is signing in: the requested id plus the sign-in's SSO id and email. `links_any_user` is
/// `LITELLM_EMAIL_FALLBACK_LINKS_ANY_USER`, which lets an email match any row and skips the SSO link guards.
#[derive(Clone, Debug)]
pub struct SignIn {
    pub user_id: String,
    pub sso_user_id: Option<String>,
    pub user_email: Option<String>,
    pub email_verified: Option<bool>,
    pub links_any_user: bool,
}

/// What `default_internal_user_params` sets on a user created at sign-in. The host resolves them, the reset time
/// of `budget_duration` and the opening windows of `budget_limits` included. `None` leaves the column unset.
#[derive(Clone, Debug, Default)]
pub struct NewUserDefaults {
    pub user_role: Option<String>,
    pub max_budget: Option<f64>,
    pub budget_duration: Option<String>,
    pub budget_reset_at: Option<Timestamp>,
    pub models: Option<Vec<String>>,
    pub budget_limits: Option<Value>,
    pub tpm_limit: Option<i64>,
    pub rpm_limit: Option<i64>,
    pub max_parallel_requests: Option<i32>,
    pub user_alias: Option<String>,
    pub team_id: Option<String>,
    pub organization_id: Option<String>,
    pub object_permission_id: Option<String>,
    pub password_reset_required: Option<bool>,
    pub metadata: Option<Value>,
    pub allowed_cache_controls: Option<Vec<String>>,
    pub policies: Option<Vec<String>>,
    pub model_max_budget: Option<Value>,
}

/// Reads, links and provisions `LiteLLM_UserTable` rows.
pub struct UserRepository {
    database: Database,
}

impl UserRepository {
    pub fn new(database: Database) -> Self {
        Self { database }
    }

    /// The user `sign_in` resolves to, with its organization memberships, or `None` when no row matches.
    /// `side` is where the id and SSO id lookup reads, the writer for callers that must see their own writes.
    pub async fn resolve(
        &self,
        sign_in: &SignIn,
        side: Side,
    ) -> Result<Option<User>, litellm_db::Error> {
        let resolved = self
            .database
            .run(GET_USER_OBJECT, || async {
                resolve_once(&mut &self.database, sign_in, side).await
            })
            .await?;
        if let Some((user_id, sso_user_id)) = resolved.link_later {
            self.set_sso_user_id_later(user_id, sso_user_id);
        }
        Ok(resolved.user)
    }

    /// Whether a user other than `user_id` holds `user_email`, ignoring case.
    pub async fn email_belongs_to_another_user(
        &self,
        user_email: &str,
        user_id: &str,
    ) -> Result<bool, litellm_db::Error> {
        self.database
            .run(FIND_FIRST_USER, || async {
                let mut session = self.database.session(Side::Read).await?;
                queries::email_belongs_to_another_user(session.executor(), user_email, user_id)
                    .await
            })
            .await
    }

    /// Sets `user_email` on a user with none, unless another user holds it and `links_any_user` is off. Returns
    /// the email now stored, or `None` when another user holds it or the row is gone.
    pub async fn backfill_email(
        &self,
        user_id: &str,
        user_email: &str,
        links_any_user: bool,
    ) -> Result<Option<String>, litellm_db::Error> {
        self.database
            .run(UPDATE_MANY_USERS, || async {
                let mut session = self.database.session(Side::Writer).await?;
                queries::backfill_user_email(
                    session.executor(),
                    user_id,
                    user_email,
                    links_any_user,
                )
                .await
            })
            .await
    }

    /// Sets the role of `user_id` and returns the row as stored, or `None` when no row has that id.
    pub async fn set_role(
        &self,
        user_id: &str,
        user_role: &str,
    ) -> Result<Option<User>, litellm_db::Error> {
        self.database
            .run(UPDATE_USER, || async {
                let mut session = self.database.session(Side::Writer).await?;
                queries::set_user_role(session.executor(), user_id, user_role).await
            })
            .await
    }

    /// Creates the user `sign_in` names with `defaults`, as `get_user_object` did under `user_id_upsert`. The
    /// email is stored unless another user holds it and `links_any_user` is off, and the SSO id only while
    /// `links_any_user` is off. A sign-in that loses to a concurrent one gets the row that won.
    pub async fn insert(
        &self,
        sign_in: &SignIn,
        defaults: &NewUserDefaults,
    ) -> Result<Inserted<User>, litellm_db::Error> {
        self.database
            .run(CREATE_USER, || async {
                insert_once(&mut &self.database, sign_in, defaults).await
            })
            .await
    }

    /// Records `sso_user_id` on `user_id` in the background, as Python did for an email match under
    /// `links_any_user`; the sign-in does not wait for it.
    fn set_sso_user_id_later(&self, user_id: String, sso_user_id: String) {
        let database = self.database.clone();
        litellm_call_scope::spawn_in_call(async move {
            let outcome = database
                .run(UPDATE_USER, || async {
                    let mut session = database.session(Side::Writer).await?;
                    queries::set_sso_user_id(session.executor(), &user_id, &sso_user_id).await
                })
                .await;
            if let Err(error) = outcome {
                tracing::warn!(
                    error = %litellm_error::Report::new(&error),
                    "linking an SSO id to a user matched by email failed"
                );
            }
        });
    }
}

/// The insert, then on a conflict the read a later sign-in would make, in a statement of its own because the
/// insert cannot see a row committed after it began.
pub(super) async fn insert_once(
    connections: &mut impl Connections,
    sign_in: &SignIn,
    defaults: &NewUserDefaults,
) -> Result<Inserted<User>, litellm_db::Error> {
    let sso_user_id = sign_in
        .sso_user_id
        .as_deref()
        .filter(|_| !sign_in.links_any_user);
    let new_user = queries::NewUser {
        user_id: &sign_in.user_id,
        user_email: sign_in.user_email.as_deref(),
        email_may_be_shared: sign_in.links_any_user,
        sso_user_id,
        defaults,
    };
    let created = connections
        .with(Side::Writer, async |db| {
            queries::insert_user(db, &new_user).await
        })
        .await?;
    if let Some(user) = created {
        return Ok(Inserted::Created(user));
    }
    connections
        .with(Side::Writer, async |db| {
            queries::find_user(db, Some(&sign_in.user_id), sso_user_id).await
        })
        .await
        .map(Inserted::Conflict)
}

/// What a resolve found, and the SSO id link it leaves for the background under `links_any_user`.
pub(super) struct Resolved {
    pub user: Option<User>,
    pub link_later: Option<(String, String)>,
}

impl From<Option<User>> for Resolved {
    fn from(user: Option<User>) -> Self {
        Self {
            user,
            link_later: None,
        }
    }
}

/// The id or SSO id match, linking the requested id as its own SSO id when the sign-in carries it, else the email
/// match.
pub(super) async fn resolve_once(
    connections: &mut impl Connections,
    sign_in: &SignIn,
    side: Side,
) -> Result<Resolved, litellm_db::Error> {
    let found = connections
        .with(side, async |db| {
            queries::find_user(db, Some(&sign_in.user_id), sign_in.sso_user_id.as_deref()).await
        })
        .await?;
    match found {
        Some(user) if user.user_id == sign_in.user_id => {
            if user.sso_user_id.is_none()
                && sign_in.sso_user_id.as_deref() == Some(user.user_id.as_str())
                && !sign_in.links_any_user
            {
                connections
                    .with(Side::Writer, async |db| {
                        queries::link_null_sso_user_id(db, &user.user_id, &user.user_id).await
                    })
                    .await?;
            }
            Ok(Some(user).into())
        }
        Some(user) => Ok(Some(user).into()),
        None => resolve_by_email(connections, sign_in, side).await,
    }
}

/// A user with no row for the requested id or SSO id, found by email unless the email is unverified. With an SSO
/// id, only an unlinked row matches, and it is linked only if no other sign-in linked that SSO id first.
async fn resolve_by_email(
    connections: &mut impl Connections,
    sign_in: &SignIn,
    side: Side,
) -> Result<Resolved, litellm_db::Error> {
    let Some(user_email) = &sign_in.user_email else {
        return Ok(None.into());
    };
    if !(sign_in.links_any_user || sign_in.email_verified != Some(false)) {
        return Ok(None.into());
    }
    let unlinked_only = !sign_in.links_any_user && sign_in.sso_user_id.is_some();
    let by_email = connections
        .with(side, async |db| {
            queries::find_user_by_email(db, user_email, unlinked_only).await
        })
        .await?;
    let Some(sso_user_id) = &sign_in.sso_user_id else {
        return Ok(by_email.into());
    };
    if sign_in.links_any_user {
        let link_later = by_email
            .as_ref()
            .map(|user| (user.user_id.clone(), sso_user_id.clone()));
        return Ok(Resolved {
            user: by_email,
            link_later,
        });
    }
    if let Some(mut user) = by_email {
        let linked = connections
            .with(Side::Writer, async |db| {
                queries::link_null_sso_user_id(db, &user.user_id, sso_user_id).await
            })
            .await?;
        if linked {
            user.sso_user_id = Some(sso_user_id.clone());
            return Ok(Some(user).into());
        }
    }
    connections
        .with(Side::Writer, async |db| {
            queries::find_user(db, None, Some(sso_user_id)).await
        })
        .await
        .map(Resolved::from)
}
