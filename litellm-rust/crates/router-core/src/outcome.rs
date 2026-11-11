//! Which of a request's outcomes reach the deployment's health. The Python router learns them from
//! logging callbacks it registers on the process lists; `_core_serve_through_request_logging` and
//! `_core_discard` take those off, and `_core_bind_to_request_logging` puts them back on one
//! request's logging object. This core registers none, so the host answers, for each request, what
//! would have run ([`OutcomeCallbacks`]), and every write a callback owns passes through
//! [`OutcomeCallbacks::record`].
//!
//! What the Python router writes outside its callbacks is not gated, and reaches [`crate::Health`]
//! with the default value: `make_call`'s usage, `async_routing_strategy_pre_call_checks`'
//! cooldown, `_trigger_cooldown_for_failed_deployment`, `RouterHealth.record_failure` and a
//! realtime session's success.

use std::future::Future;

use serde::Deserialize;

use crate::error::Error;

/// Which of the router's outcome callbacks litellm's logging runs for a request. The default is a
/// router whose callbacks are on the process lists.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct OutcomeCallbacks {
    /// `deployment_callback_on_success`, `sync_deployment_callback_on_success` for a sync call.
    /// `_core_bind_to_request_logging` binds only the former.
    pub success: bool,
    /// `deployment_callback_on_failure` and `async_deployment_callback_on_failure`.
    pub failure: bool,
    /// The attempt runs through `make_call`, which counts its usage whatever the callbacks do.
    pub make_call: bool,
    /// The request's attempts log on one logging object the caller passed (`litellm_logging_obj`),
    /// which logs one success a request, the response that serves it: an attempt whose blocked
    /// response is raised for a fallback counts no success.
    pub shared_logging: bool,
}

impl Default for OutcomeCallbacks {
    fn default() -> Self {
        Self {
            success: true,
            failure: true,
            make_call: true,
            shared_logging: false,
        }
    }
}

/// What an outcome writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutcomeWrite {
    /// The success counter, or the usage the success callback adds at the end of a stream.
    Success,
    /// The tokens and request of an attempt that returned: `make_call`'s write, or the success
    /// callback's for an endpoint that skips `make_call`.
    AttemptUsage,
    /// The failure counter, the allowed-fails counter, the cooldown and the failed request.
    Failure,
}

impl OutcomeCallbacks {
    /// Run `record` when `write` happens for this request, else record nothing.
    pub(crate) async fn record<T: Default>(
        self,
        write: OutcomeWrite,
        record: impl Future<Output = Result<T, Error>>,
    ) -> Result<T, Error> {
        let runs = match write {
            OutcomeWrite::Success => self.success,
            OutcomeWrite::AttemptUsage => self.success || self.make_call,
            OutcomeWrite::Failure => self.failure,
        };
        if runs { record.await } else { Ok(T::default()) }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    const REGISTERED: OutcomeCallbacks = OutcomeCallbacks {
        success: true,
        failure: true,
        make_call: true,
        shared_logging: false,
    };
    const UNBOUND: OutcomeCallbacks = OutcomeCallbacks {
        success: false,
        failure: false,
        make_call: true,
        shared_logging: false,
    };
    const UNBOUND_WITHOUT_MAKE_CALL: OutcomeCallbacks = OutcomeCallbacks {
        success: false,
        failure: false,
        make_call: false,
        shared_logging: false,
    };
    const BOUND_SYNC: OutcomeCallbacks = OutcomeCallbacks {
        success: false,
        failure: true,
        make_call: true,
        shared_logging: false,
    };

    #[rstest]
    #[case::registered_success(REGISTERED, OutcomeWrite::Success, true)]
    #[case::registered_usage(REGISTERED, OutcomeWrite::AttemptUsage, true)]
    #[case::registered_failure(REGISTERED, OutcomeWrite::Failure, true)]
    #[case::unbound_success(UNBOUND, OutcomeWrite::Success, false)]
    #[case::unbound_usage(UNBOUND, OutcomeWrite::AttemptUsage, true)]
    #[case::unbound_failure(UNBOUND, OutcomeWrite::Failure, false)]
    #[case::no_make_call_usage(UNBOUND_WITHOUT_MAKE_CALL, OutcomeWrite::AttemptUsage, false)]
    #[case::bound_sync_success(BOUND_SYNC, OutcomeWrite::Success, false)]
    #[case::bound_sync_failure(BOUND_SYNC, OutcomeWrite::Failure, true)]
    #[tokio::test]
    async fn a_write_runs_when_its_callback_does(
        #[case] callbacks: OutcomeCallbacks,
        #[case] write: OutcomeWrite,
        #[case] runs: bool,
    ) {
        let recorded = callbacks.record(write, async { Ok(true) }).await;

        assert_eq!(recorded.ok(), Some(runs));
    }

    #[test]
    fn a_router_on_the_process_lists_is_the_default() {
        let parsed: OutcomeCallbacks = serde_json::from_str("{}").expect("an empty document");

        assert_eq!(parsed, REGISTERED);
    }
}
