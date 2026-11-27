//! The proxy's spend log callback on the ledger path, the Rust port of `_ProxyDBLogger`
//! (`litellm/proxy/hooks/proxy_track_cost_callback.py`).
//!
//! It subscribes to `attempt_ended` (successful attempts only, since a failed attempt a fallback
//! recovers is charged by the success) and to `post_call_failure`, and writes the settled budget
//! reservation back on both, so Python's later release paths see it finalized. Per request it
//! keeps the cost callback's order (`_update_database_and_spend_counters`):
//!
//! 1. the gate (`_should_track_cost_callback`) and the deferrals that release or keep the
//!    budget reservation ([`policy`]);
//! 2. the search free tier discount ([`search_free_tier`]), then the customer surcharge
//!    ([`customer_margin`]), which failure rows take too;
//! 3. the row from [`litellm_callbacks_spend_log::SpendLogRowBuilder`], under the request id the
//!    typed `spend_log_id` field carries, and its ledger columns;
//! 4. the settle-once claim of a batch or video job's cost row, straight to the ledger store, or
//!    the enqueue into the Rust spend log writer, whose flush is requested at once for a
//!    Responses turn the next turn reads back from the database;
//! 5. the rest of `update_database` in one fire-and-forget host call ([`host::AfterWrite`]);
//! 6. the post-call counters, which settle the reservation (`charge_counters`);
//! 7. one fire-and-forget host call for `update_cache` and the customer spend alert.
//!
//! A cost it cannot track raises the failed tracking alert, and a batch's costing hears whether
//! the spend write began and whether tracking failed.
//!
//! Python hands over what it alone holds through the `spend_log_input` field
//! ([`hook_input::SpendLogInput`]) and answers the host calls in [`host`]. The Python spend log
//! monitor owns the writer's flushes and its shutdown drain.

mod charge;
mod customer_margin;
pub mod hook_input;
pub mod host;
mod policy;
mod search_free_tier;

use std::sync::Arc;

use futures_util::future::BoxFuture;
use litellm_callbacks::{
    Activation, BilledRetryAttempt, BudgetReservation, Callback, CallbackError, Cost, Field,
    FieldSet, Hook, HookInput, HookOutput, Host, Manifest, Outcome, Patch, Selection, SpendLogId,
    Subscription, Verdict, typed,
};
use litellm_callbacks_budgets::Budgets;
use litellm_redaction::TokenRedactor;
use litellm_redis_store::RequestRedis;
use litellm_spend_log_writer::{LedgerStore, SpendLogWriter};

pub use hook_input::SpendLogInput;

pub const NAME: &str = "spend_log";

fn manifest() -> Manifest {
    let writes = FieldSet::of(&[Field::BudgetReservation]);
    Manifest {
        selection: Selection {
            activation: Activation::Required,
            opt_out_by: Vec::new(),
            honors_no_log: false,
            ..Selection::named(NAME)
        },
        subscriptions: vec![
            Subscription {
                reads: FieldSet::of(&[
                    Field::Outcome,
                    Field::SpendLogInput,
                    Field::SpendLogId,
                    Field::Cost,
                    Field::BudgetReservation,
                ]),
                writes,
                ..Subscription::on(Hook::AttemptEnded)
            },
            Subscription {
                reads: FieldSet::of(&[
                    Field::SpendLogInput,
                    Field::SpendLogId,
                    Field::BilledRetryAttempts,
                    Field::Cost,
                    Field::BudgetReservation,
                ]),
                writes,
                ..Subscription::on(Hook::PostCallFailure)
            },
        ],
        has_request_state: false,
    }
}

/// What the bridge builds the callback from: the writer the proxy's monitor flushes, the budget
/// engine over the shared spend counters and the process's token redactor.
pub struct SpendLogParts<S> {
    pub writer: Arc<SpendLogWriter<S>>,
    pub budgets: Arc<Budgets>,
    pub tokens: Arc<TokenRedactor>,
}

/// The typed contract fields a hook reads besides the input.
pub(crate) struct Typed {
    pub spend_log_id: Option<SpendLogId>,
    pub cost: Option<Cost>,
    pub attempts: Vec<BilledRetryAttempt>,
}

/// The spend log callback. `S` is the writer's ledger store: Postgres in the proxy, a fake in
/// tests.
pub struct SpendLogCallback<S> {
    manifest: Manifest,
    writer: Arc<SpendLogWriter<S>>,
    budgets: Arc<Budgets>,
    tokens: Arc<TokenRedactor>,
}

impl<S: LedgerStore + 'static> SpendLogCallback<S> {
    pub fn new(parts: SpendLogParts<S>) -> Self {
        Self {
            manifest: manifest(),
            writer: parts.writer,
            budgets: parts.budgets,
            tokens: parts.tokens,
        }
    }

    pub fn writer(&self) -> &Arc<SpendLogWriter<S>> {
        &self.writer
    }

    /// One hook run for a request whose Redis batches are `redis`: the counter step's increments
    /// ride its post-call batch, which tracks the request's own committed charge until it lands.
    /// `None` outside a request sends them at once.
    pub async fn handle_with(
        &self,
        input: &HookInput,
        host: Host,
        redis: Option<&RequestRedis>,
    ) -> HookOutput {
        if input.hook == Hook::AttemptEnded
            && matches!(
                input.fields.get::<typed::Outcome>()?,
                Some(Outcome::Failure | Outcome::FallbackFailure)
            )
        {
            return Ok(None);
        }
        let Some(raw) = input.fields.get::<typed::SpendLogInput>()? else {
            return Ok(None);
        };
        let parsed: SpendLogInput = match serde_json::from_value(raw) {
            Ok(parsed) => parsed,
            Err(error) => {
                tracing::error!(%error, "Error in tracking cost callback - the spend log input does not parse");
                return Ok(None);
            }
        };
        let fields = Typed {
            spend_log_id: input.fields.get::<typed::SpendLogId>()?,
            cost: input.fields.get::<typed::Cost>()?,
            attempts: if input.hook == Hook::PostCallFailure {
                input
                    .fields
                    .get::<typed::BilledRetryAttempts>()?
                    .unwrap_or_default()
            } else {
                Vec::new()
            },
        };
        let mut reservation: Option<BudgetReservation> =
            input.fields.get::<typed::BudgetReservation>()?;
        let before = reservation.clone();
        match (input.hook, parsed) {
            (Hook::AttemptEnded, SpendLogInput::Success(success)) => {
                self.on_success(*success, &fields, &mut reservation, &host, redis)
                    .await;
            }
            (Hook::PostCallFailure, SpendLogInput::Failure(failure)) => {
                self.on_failure(*failure, &fields, &mut reservation, &host, redis)
                    .await;
            }
            _ => {}
        }
        settled(before, reservation)
    }
}

/// The reservation written back once the hook settled it, so the Python dict carries the same
/// `finalized` and `applied_adjustment` stamps.
fn settled(before: Option<BudgetReservation>, after: Option<BudgetReservation>) -> HookOutput {
    let Some(after) = after else { return Ok(None) };
    if before.as_ref() == Some(&after) {
        return Ok(None);
    }
    let value =
        serde_json::to_value(&after).map_err(|error| CallbackError::Failed(error.to_string()))?;
    Ok(Some(Verdict::Patch(
        Patch::default().set(Field::BudgetReservation, value),
    )))
}

impl<S: LedgerStore + 'static> Callback for SpendLogCallback<S> {
    type State = ();

    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Every charge needs the proxy's host calls, so a run with no host is refused.
    fn handle<'a>(
        &'a self,
        _input: &'a HookInput,
        _state: Option<&'a ()>,
    ) -> BoxFuture<'a, HookOutput> {
        Box::pin(async {
            Err(CallbackError::Failed(
                "the spend log callback needs the proxy's host calls".to_owned(),
            ))
        })
    }

    fn handle_hosted<'a>(
        &'a self,
        input: &'a HookInput,
        _state: Option<&'a ()>,
        host: Host,
    ) -> BoxFuture<'a, HookOutput> {
        Box::pin(self.handle_with(input, host, None))
    }
}
