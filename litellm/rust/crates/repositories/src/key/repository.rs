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
    queries::{self, KeyViewRow},
    view::KeyView,
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
            return Ok(Some(FetchedKey::new(grouped(row)?, now)));
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
        queries::find_key_view(session.executor(), &active_token)
            .await?
            .map(|row| Ok(FetchedKey::new(grouped(row)?, now)))
            .transpose()
    }

    fn deprecated_keys(&self) -> std::sync::MutexGuard<'_, DeprecatedKeys> {
        self.deprecated
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

fn grouped(row: KeyViewRow) -> Result<KeyView, litellm_db::Error> {
    KeyView::try_from(row).map_err(|orphaned| sqlx::Error::Decode(Box::new(orphaned)).into())
}

/// A key view stamped with when it was read.
#[derive(Serialize)]
pub struct FetchedKey {
    #[serde(flatten)]
    view: KeyView,
    pub last_refreshed_at: f64,
}

impl FetchedKey {
    pub fn view(&self) -> &KeyView {
        &self.view
    }

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

#[cfg(test)]
mod tests {
    use std::time::{Duration, UNIX_EPOCH};

    use serde_json::{Value, json};

    use super::FetchedKey;
    use crate::test_support::{PLAIN_KEY_JSON, plain_key};

    #[test]
    fn a_fetched_key_writes_the_flat_row_and_when_it_was_read() {
        let fetched = FetchedKey::new(plain_key("hash"), UNIX_EPOCH + Duration::from_millis(1500));

        let mut expected: Value = serde_json::from_str(PLAIN_KEY_JSON).unwrap();
        expected.as_object_mut().unwrap().extend(
            json!({
                "token": "hash",
                "created_at": "2026-11-19T10:00:00.123+00:00",
                "updated_at": "2026-11-19T10:00:00.123+00:00",
                "last_refreshed_at": 1.5
            })
            .as_object()
            .unwrap()
            .clone(),
        );
        assert_eq!(serde_json::to_value(&fetched).unwrap(), expected);
    }
}
