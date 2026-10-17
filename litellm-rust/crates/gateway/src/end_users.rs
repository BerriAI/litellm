//! End users' budgets and limits, read from the proxy's database and reused for a minute.

use std::{
    collections::HashMap,
    sync::Mutex,
    time::{Duration, Instant},
};

use litellm_db::{EndUser, PgPool};

/// How long an end user's budget is reused before it is read again.
const END_USER_TTL: Duration = Duration::from_secs(60);
/// End users held in memory; the cache starts over past this.
const MAX_END_USERS: usize = 10_000;

/// An end user and the default and override budgets it was read with.
type EndUserKey = (String, Option<String>, Option<String>);

pub(crate) struct EndUsers {
    pool: PgPool,
    cache: Mutex<HashMap<EndUserKey, (EndUser, Instant)>>,
}

impl EndUsers {
    pub(crate) fn new(pool: PgPool) -> Self {
        Self {
            pool,
            cache: Mutex::default(),
        }
    }

    /// `None` when the database could not be read.
    pub(crate) async fn get(
        &self,
        id: &str,
        default_budget_id: Option<&str>,
        override_budget_id: Option<&str>,
    ) -> Option<EndUser> {
        let key = (
            id.to_owned(),
            default_budget_id.map(str::to_owned),
            override_budget_id.map(str::to_owned),
        );
        let now = Instant::now();
        if let Some((end_user, read_at)) = self
            .cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&key)
            && now.duration_since(*read_at) < END_USER_TTL
        {
            return Some(end_user.clone());
        }
        match litellm_db::find_end_user(&self.pool, id, default_budget_id, override_budget_id).await
        {
            Ok(end_user) => {
                let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
                if cache.len() >= MAX_END_USERS {
                    cache.clear();
                }
                cache.insert(key, (end_user.clone(), now));
                Some(end_user)
            }
            Err(error) => {
                tracing::warn!(%error, "reading an end user's budget failed; not checking it");
                None
            }
        }
    }
}
