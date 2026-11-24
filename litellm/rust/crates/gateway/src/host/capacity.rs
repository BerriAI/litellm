//! The gateway's answers to the model capacity engine's host calls, the counterpart of the
//! proxy's `CapacityHost`: the registry of callers active on a model, and the proxy's log lines.
//!
//! The rest is answered with nothing, since no caller the gateway serves reaches it: a per-model
//! limit that borrows idle capacity belongs to a config `admit` leaves to Python, and the share
//! and throttle metrics belong to Prometheus, which is a callback of its own.

use std::{
    collections::HashMap,
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

use litellm_callbacks_model_capacity::{
    CapacityError, CapacityHost, Estimate, Raised, RaisedKind, SeamFuture, ShareGauge, Throttle,
    Touched,
};
use litellm_python_compat::json::loaded_str;
use litellm_redis_store::active_teams::{ActiveMember, ActiveTeams};
use parking_lot::Mutex;
use serde_json::Value;

/// `UNLICENSED_PRIORITY_MESSAGE`.
const UNLICENSED_PRIORITY: &str = "PREMIUM FEATURE: Reserving tpm/rpm by priority is a premium feature. Please add a 'LITELLM_LICENSE' to your .env to enable this.\nGet a license: https://docs.litellm.ai/docs/enterprise.";

/// One touch the engine asked for and has yet to wait on.
struct Touch {
    model: String,
    member: String,
    priority_class: Value,
    window: Duration,
}

pub struct NativeCapacityHost {
    active: ActiveTeams,
    pending: Mutex<HashMap<u64, Touch>>,
    next: AtomicU64,
}

/// Every member as the engine reads it. The caller keeps the class it was asked under as given,
/// and every other member has the text the registry holds.
fn members(listed: Vec<ActiveMember>, touch: &Touch) -> Vec<(String, Value)> {
    listed
        .into_iter()
        .map(|active| {
            let priority_class = if active.member == touch.member {
                touch.priority_class.clone()
            } else {
                Value::String(active.priority_class)
            };
            (active.member, priority_class)
        })
        .collect()
}

/// A registry that could not be read, which holds the caller to its class pool instead of a slice.
fn unavailable(error: &litellm_redis_store::Error) -> CapacityError {
    CapacityError::Raised(Raised {
        token: 0,
        kind: RaisedKind::RegistryUnavailable,
        message: litellm_error::Report::new(error).to_string(),
    })
}

impl NativeCapacityHost {
    pub fn new(active: ActiveTeams) -> Self {
        Self {
            active,
            pending: Mutex::new(HashMap::new()),
            next: AtomicU64::new(0),
        }
    }
}

impl CapacityHost for NativeCapacityHost {
    fn touch(
        &self,
        model: &str,
        member: &str,
        priority_class: &Value,
        window: i64,
    ) -> Result<Touched, CapacityError> {
        let touch = Touch {
            model: model.to_owned(),
            member: member.to_owned(),
            priority_class: priority_class.clone(),
            window: Duration::from_secs(u64::try_from(window).unwrap_or(0)),
        };
        if self.active.has_redis() {
            let token = self.next.fetch_add(1, Ordering::Relaxed);
            self.pending.lock().insert(token, touch);
            return Ok(Touched::Pending(token));
        }
        let listed = self.active.touch_local(
            &touch.model,
            &touch.member,
            &loaded_str(&touch.priority_class),
            touch.window,
        );
        Ok(Touched::Members(members(listed, &touch)))
    }

    fn touched(&self, pending: u64) -> SeamFuture<'_, Vec<(String, Value)>> {
        Box::pin(async move {
            let Some(touch) = self.pending.lock().remove(&pending) else {
                return Ok(Vec::new());
            };
            let listed = self
                .active
                .touch(
                    &touch.model,
                    &touch.member,
                    &loaded_str(&touch.priority_class),
                    touch.window,
                )
                .await
                .map_err(|error| unavailable(&error))?;
            Ok(members(listed, &touch))
        })
    }

    fn metrics_enabled(&self) -> SeamFuture<'_, bool> {
        Box::pin(async { Ok(false) })
    }

    fn throttled<'a>(&'a self, _: &'a Throttle) -> SeamFuture<'a, ()> {
        Box::pin(async { Ok(()) })
    }

    fn set_share<'a>(&'a self, _: &'a ShareGauge) -> SeamFuture<'a, ()> {
        Box::pin(async { Ok(()) })
    }

    fn unlicensed_priority(&self) -> Result<(), CapacityError> {
        tracing::error!("{UNLICENSED_PRIORITY}");
        Ok(())
    }

    fn log(&self, level: &'static str, message: &str) -> Result<(), CapacityError> {
        match level {
            "error" => tracing::error!("{message}"),
            "warning" => tracing::warn!("{message}"),
            _ => tracing::info!("{message}"),
        }
        Ok(())
    }

    fn model_group<'a>(&'a self, model: &'a str) -> SeamFuture<'a, String> {
        Box::pin(async move { Ok(model.to_owned()) })
    }

    fn estimate_tokens<'a>(
        &'a self,
        _: i64,
        _: Option<&'a str>,
        _: Option<&'a str>,
    ) -> SeamFuture<'a, Estimate> {
        Box::pin(async { Ok(Estimate::NoBody) })
    }

    fn post_call_success<'a>(&'a self, _: &'a [(String, Value)]) -> SeamFuture<'a, ()> {
        Box::pin(async { Ok(()) })
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    const WINDOW: i64 = 60;

    fn touched(host: &NativeCapacityHost, member: &str, class: &Value) -> Vec<(String, Value)> {
        match host.touch("claude", member, class, WINDOW).unwrap() {
            Touched::Members(members) => members,
            Touched::Pending(_) => panic!("memory answers at once"),
        }
    }

    #[rstest]
    fn without_redis_every_caller_seen_on_the_model_is_listed_at_once() {
        let host = NativeCapacityHost::new(ActiveTeams::new(None));

        touched(&host, "key\u{1f}a", &json!("prod"));

        assert_eq!(
            touched(&host, "key\u{1f}b", &json!("dev")),
            [
                ("key\u{1f}a".to_owned(), json!("prod")),
                ("key\u{1f}b".to_owned(), json!("dev")),
            ]
        );
    }

    #[rstest]
    fn the_caller_keeps_the_class_it_was_asked_under_as_given() {
        let host = NativeCapacityHost::new(ActiveTeams::new(None));

        assert_eq!(
            touched(&host, "key\u{1f}a", &json!(7)),
            [("key\u{1f}a".to_owned(), json!(7))]
        );
    }
}
