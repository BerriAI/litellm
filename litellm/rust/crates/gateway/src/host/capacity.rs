//! The gateway's answers to the model capacity engine's host calls, the counterpart of the
//! proxy's `CapacityHost`: the registry of callers active on a model, the tokens the body of the
//! call being admitted asks for, and the proxy's log lines.
//!
//! One is made for each hook of a call, since the proxy's host reads the request it is live in.
//! The rest is answered with nothing, since no caller the gateway serves reaches it: the share
//! and throttle metrics belong to Prometheus, which is a callback of its own.

use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use litellm_callbacks_model_capacity::{
    CapacityError, CapacityHost, Estimate, Raised, RaisedKind, SeamFuture, ShareGauge, Throttle,
    TokenRateLimitType, Touched,
};
use litellm_callbacks_parallel_request_limiter::{
    CapSettings, EstimateRequest, OutputCapFacts, PythonServed, explicit_output_cap,
    input_and_output_tokens, output_cap, write_output_cap,
};
use litellm_gateway_inference::CallType;
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

/// The call being admitted: its body and which of its tokens a limit counts.
pub struct Asked {
    pub body: Value,
    pub counted: TokenRateLimitType,
}

pub struct NativeCapacityHost {
    active: Arc<ActiveTeams>,
    pending: Mutex<HashMap<u64, Touch>>,
    next: AtomicU64,
    /// `None` on a hook that admits no call, which estimates nothing.
    asked: Option<Asked>,
}

/// `cap_output_for_limit`: the output cap the proxy writes into a body that sets none, so one
/// unbounded answer cannot overshoot a share of `tpm_share` tokens. `Err` for a body whose own
/// cap only Python reads.
pub(crate) fn cap_output(
    body: &mut Value,
    call_type: &str,
    tpm_share: i64,
) -> Result<(), PythonServed> {
    let cap = CapSettings::default();
    let explicit = match &*body {
        Value::Object(request) => explicit_output_cap(request, call_type, &cap)?,
        _ => None,
    };
    let implicit = output_cap(&OutputCapFacts {
        min_configured_limit: Some(tpm_share),
        responses: call_type == <&str>::from(CallType::Aresponses),
        exempt: explicit.is_some(),
        configured_output_tokens: None,
        fraction: cap.fraction,
        default_max_tokens_estimate: cap.default_max_tokens_estimate,
        responses_min_output_tokens: cap.responses_min_output_tokens,
    });
    if let Some(implicit) = implicit {
        write_output_cap(body, call_type, implicit);
    }
    Ok(())
}

impl Asked {
    /// `request_token_estimate` of the body as it is dispatched under a share of `tpm_share`
    /// tokens, which is with the output cap that share writes into it.
    fn tokens(&self, tpm_share: i64, call_type: &str) -> Result<i64, PythonServed> {
        let mut dispatched = self.body.clone();
        cap_output(&mut dispatched, call_type, tpm_share)?;
        let (input, output) = input_and_output_tokens(
            EstimateRequest {
                data: &dispatched,
                call_type,
                min_configured_tpm_limit: Some(tpm_share),
                configured_output_tokens: None,
                include_media: true,
            },
            &CapSettings::default(),
        )?;
        let tokens = match self.counted {
            TokenRateLimitType::Input => Some(input),
            TokenRateLimitType::Output => Some(output),
            TokenRateLimitType::Total => input.checked_add(output),
        };
        Ok(tokens.map_or(i64::MAX, |tokens| i64::try_from(tokens).unwrap_or(i64::MAX)))
    }
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
    /// `asked` is the call a pre-call hook admits.
    pub fn new(active: Arc<ActiveTeams>, asked: Option<Asked>) -> Self {
        Self {
            active,
            pending: Mutex::new(HashMap::new()),
            next: AtomicU64::new(0),
            asked,
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

    /// A body only Python's estimator reads, one that carries media, holds the least, as the
    /// proxy holds one token when its estimate fails.
    fn estimate_tokens<'a>(
        &'a self,
        tpm_share: i64,
        _: &'a str,
        call_type: Option<&'a str>,
    ) -> SeamFuture<'a, Estimate> {
        let estimate = match &self.asked {
            None => Estimate::NoBody,
            Some(asked) => asked
                .tokens(tpm_share, call_type.unwrap_or_default())
                .map_or(Estimate::Failed, Estimate::Tokens),
        };
        Box::pin(async move { Ok(estimate) })
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
        let host = NativeCapacityHost::new(Arc::new(ActiveTeams::new(None)), None);

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
        let host = NativeCapacityHost::new(Arc::new(ActiveTeams::new(None)), None);

        assert_eq!(
            touched(&host, "key\u{1f}a", &json!(7)),
            [("key\u{1f}a".to_owned(), json!(7))]
        );
    }
}
