//! A JWT's user, and the sign-in its row still owes. Authentication only reads a user, so a
//! request that goes to Python has written nothing here and Python makes the sign-in itself. The
//! write the read found owed (an SSO id to link, an email to fill, a user to create) rides on the
//! user's row to `admit`, which makes it last, once every check has passed and the native app is
//! certain to serve the request.

use std::{fmt, sync::Arc};

use litellm_gateway_auth::Authenticated;
use litellm_gateway_auth_core::{
    Named,
    identity::jwt::User,
    record::{Record, UserRecord},
};
use litellm_repositories::SignIn;

use super::{AuthStore, NativeRows, affiliation};

/// The user the core carries for this host: its record, and the sign-in it still owes.
#[derive(Clone, Debug, PartialEq)]
pub struct UserRow {
    pub record: UserRecord,
    owed: Option<Arc<Owed>>,
}

impl From<UserRecord> for UserRow {
    fn from(record: UserRecord) -> Self {
        Self { record, owed: None }
    }
}

impl Record for UserRow {
    fn id(&self) -> &str {
        self.record.id()
    }

    fn alias(&self) -> Option<&str> {
        self.record.alias()
    }
}

impl UserRow {
    /// The sign-in this user's row still owes.
    pub(super) fn owed(&self) -> Option<&Owed> {
        self.owed.as_deref()
    }
}

/// A sign-in the host read a user for and has not made.
pub struct Owed {
    store: Arc<AuthStore>,
    sign_in: SignIn,
    upsert: bool,
    /// The user the read found or, for one the sign-in creates, the id it names.
    user_id: String,
}

impl Owed {
    pub(super) fn new(store: Arc<AuthStore>, sign_in: SignIn, upsert: bool, user_id: &str) -> Self {
        Self {
            store,
            sign_in,
            upsert,
            user_id: user_id.to_owned(),
        }
    }

    /// Makes the sign-in, as `get_user_object` makes it. `false` when it failed, or left another
    /// user than the one this request was authenticated as or one the native app does not serve,
    /// which a write that landed between the read and now can do.
    pub(crate) async fn settle(&self) -> bool {
        let signed_in = self.store.signed_in(&self.sign_in, self.upsert).await;
        match signed_in {
            Ok(Some(user)) => user.user_id == self.user_id && affiliation::user(&user).is_ok(),
            Err(error) => {
                tracing::warn!(
                    error = %litellm_error::Report::new(&error),
                    "a sign-in the request owed was not made"
                );
                false
            }
            Ok(None) => false,
        }
    }
}

impl PartialEq for Owed {
    fn eq(&self, other: &Self) -> bool {
        (&self.sign_in, self.upsert, &self.user_id)
            == (&other.sign_in, other.upsert, &other.user_id)
    }
}

impl fmt::Debug for Owed {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Owed")
            .field("user_id", &self.user_id)
            .field("upsert", &self.upsert)
            .finish_non_exhaustive()
    }
}

/// `user` as the core carries it, with the sign-in its row owes.
pub(super) fn carrying(user: User<UserRecord>, owed: Option<Owed>) -> User<UserRow> {
    let User {
        user_id,
        user_email,
        user_role,
        teams,
        deactivated,
        row,
    } = user;
    User {
        user_id,
        user_email,
        user_role,
        teams,
        deactivated,
        row: UserRow {
            record: row,
            owed: owed.map(Arc::new),
        },
    }
}

/// The user `sign_in` creates, as the core reads the row the insert stores: no role, no team, and
/// the email the sign-in carries.
pub(super) fn created(sign_in: &SignIn) -> User<UserRecord> {
    User {
        user_id: sign_in.user_id.clone(),
        user_email: sign_in.user_email.clone(),
        user_role: None,
        teams: Vec::new(),
        deactivated: false,
        row: UserRecord {
            user_id: sign_in.user_id.clone(),
            user_email: sign_in.user_email.clone(),
            ..UserRecord::default()
        },
    }
}

/// The sign-in `caller`'s user still owes.
pub(crate) fn owed(caller: &Authenticated<NativeRows>) -> Option<&Owed> {
    let user = caller.affiliation.holder.user().and_then(Named::stored)?;
    user.owed()
}
