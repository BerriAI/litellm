//! The virtual key repository: the `combined_view` read and the rotation grace-period fallback with its memory
//! tier, each view stamped with when it was read.

use std::{
    sync::{Mutex, PoisonError},
    time::{SystemTime, UNIX_EPOCH},
};

use chrono::{DateTime, Utc};
use litellm_db::Call;
use serde::Serialize;

use super::{
    memory::DeprecatedKeys,
    queries::{self, KeyView},
};
use crate::{Database, Side};

const KEY_VIEW: Call = Call {
    call_type: "get_data",
    table: Some("combined_view"),
    operation: None,
};

/// Reads virtual keys by hash. A hash that is not a key but was rotated away from one still within its grace
/// period resolves to the key it was rotated into.
pub struct KeyRepository {
    database: Database,
    deprecated: Mutex<DeprecatedKeys>,
}

impl KeyRepository {
    pub fn new(database: Database) -> Self {
        Self {
            database,
            deprecated: Mutex::new(DeprecatedKeys::default()),
        }
    }

    /// The view of the key whose hash is `token`, or `None` when neither a key nor a rotated-away hash in its
    /// grace period has it. `Side::Writer` reads the latest commit and skips the grace-period memory tier.
    pub async fn find(
        &self,
        token: &str,
        side: Side,
    ) -> Result<Option<FetchedKey>, litellm_db::Error> {
        let now = SystemTime::now();
        self.database
            .run(KEY_VIEW, || self.find_once(token, side, now))
            .await
    }

    async fn find_once(
        &self,
        token: &str,
        side: Side,
        now: SystemTime,
    ) -> Result<Option<FetchedKey>, litellm_db::Error> {
        let mut session = self.database.session(side).await?;
        if let Some(row) = queries::find_key_view(session.executor(), token).await? {
            return Ok(Some(FetchedKey::new(row, now)));
        }
        let cached = match side {
            Side::Read => self.deprecated_keys().active_token(token, now),
            Side::Writer => None,
        };
        let active_token = match cached {
            Some(active_token) => active_token,
            None => {
                let at = DateTime::<Utc>::from(now).naive_utc();
                match queries::find_deprecated_key(session.executor(), token, at).await {
                    Ok(Some(deprecated)) => {
                        self.deprecated_keys().insert(
                            token,
                            &deprecated.active_token_id,
                            now,
                            deprecated.revoke_at,
                        );
                        deprecated.active_token_id
                    }
                    Ok(None) => return Ok(None),
                    Err(error) => {
                        tracing::debug!(error = %litellm_error::Report::new(&error), "deprecated key lookup skipped");
                        return Ok(None);
                    }
                }
            }
        };
        Ok(queries::find_key_view(session.executor(), &active_token)
            .await?
            .map(|row| FetchedKey::new(row, now)))
    }

    fn deprecated_keys(&self) -> std::sync::MutexGuard<'_, DeprecatedKeys> {
        self.deprecated
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

/// A key view stamped with when it was read.
#[derive(Serialize)]
pub struct FetchedKey {
    #[serde(flatten)]
    view: KeyView,
    pub last_refreshed_at: f64,
}

impl FetchedKey {
    fn new(view: KeyView, now: SystemTime) -> Self {
        Self {
            view,
            last_refreshed_at: now
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64(),
        }
    }
}
