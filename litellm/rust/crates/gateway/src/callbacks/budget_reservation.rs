//! The proxy's budget reservation (`reserve_budget_for_request`) for a native `/v1/messages`
//! call: before the call runs, what it may cost is held against every budget its caller holds,
//! and a caller with nothing left is refused with Python's budget error. The hold rides the
//! call's record as its `BudgetReservation`, which the spend log callback settles when the call
//! is charged, as the proxy's spend tracking does.
//!
//! A call that ends any other way gives its hold back when its state for the request is dropped,
//! by what the record says then: nothing once the hold is finalized, everything for a failed call
//! or one that never reached the provider, and all but its input cost for one abandoned after it
//! did, as Python's failure hook and `release_budget_reservation_on_cancel` do. Accepted
//! difference: the provider counts as reached once the pre-call hook ends, before routing.
//!
//! Each counter is first warmed through `SpendReader`, as `_initialize_reservation_counters` runs
//! `ensure_spend_counter_initialized`: a cold one is seeded from the spend its row holds, or raised
//! to the spend its caller was authenticated with when it has no row, so concurrent calls charge it
//! from the spend already made. A counter that cannot be warmed is dropped everywhere and not held.
//! It is then read as `get_current_spend` reads it, raised to its row when it reads below.
//!
//! What the call holds is what `admit` priced from the body the client sent, as Python prices it
//! when it authenticates, and `admit` hands any call it cannot price to Python, so one reaching
//! here unpriced is refused, never let through unbudgeted.

use std::sync::Arc;

use axum::http::StatusCode;
use futures_util::future::{BoxFuture, join_all};
use litellm_callbacks::{
    Abort, AbortError, BudgetPlan, BudgetPlanCounter, BudgetReservation, Callback, CallbackError,
    Field, FieldSet, Hook, HookInput, HookOutput, Host, Manifest, Patch, Selection, Subscription,
    Verdict,
};
use litellm_callbacks_budgets::{
    Admission, BudgetCounterSpend, Budgets, PlannedCounter, RowSpend, SpendReader, post_call,
};
use litellm_gateway_auth::Authenticated;
use litellm_gateway_inference::{
    Refusal,
    record::{self, CallState, Ended},
};
use litellm_redis_store::spend_counters::SpendCounter;
use parking_lot::Mutex;
use serde_json::{Map, Value};

use super::{Registered, budgets};
use crate::{
    Admitted,
    host::{Hold, NativeRows, Unheld},
};

pub(crate) const NAME: &str = "budget_reservation";
/// `ProxyErrorTypes.budget_exceeded`.
const BUDGET_EXCEEDED: &str = "budget_exceeded";
/// `ProxyErrorTypes.auth_error`.
const AUTH_ERROR: &str = "auth_error";
/// `litellm.budget_exceeded_status_code`. A config that sets it goes to Python.
const BUDGET_EXCEEDED_STATUS: u16 = 422;
const UNHELD: &str = "the gateway cannot hold this call against its budget";

/// Holds what a native `/v1/messages` call may cost against every budget its caller holds,
/// before the call runs.
pub struct Reserver {
    manifest: Manifest,
    budgets: Arc<Budgets>,
    spend: SpendReader,
}

/// The counters one request's caller reserves against, and what it holds on them, which it gives
/// back when dropped unless the call's record says the hold was finalized.
pub struct Holding {
    counters: Vec<PlannedCounter>,
    priced: Result<Option<Hold>, Unheld>,
    held: Mutex<Option<BudgetReservation>>,
    budgets: Arc<Budgets>,
}

/// What a hold still open when its request state drops is owed.
#[derive(strum::VariantArray)]
enum Unwind {
    Kept,
    Released,
    InputCharged,
}

impl Unwind {
    fn of(entries: &CallState) -> Self {
        let finalized = entries
            .budget_reservation()
            .and_then(|held| held.get("finalized"))
            == Some(&Value::Bool(true));
        match (finalized, entries.ended(), entries.api_call_start()) {
            (true, ..) => Self::Kept,
            (false, Some(Ended::Failed), _) | (false, _, None) => Self::Released,
            (false, _, Some(_)) => Self::InputCharged,
        }
    }
}

impl Drop for Holding {
    fn drop(&mut self) {
        let Some(mut reservation) = self.held.get_mut().take() else {
            return;
        };
        let Some(unwind) = record::read(Unwind::of) else {
            return;
        };
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let budgets = self.budgets.clone();
        let host = Host::detached();
        match unwind {
            Unwind::Kept => {}
            Unwind::Released => drop(runtime.spawn(async move {
                post_call::release_or_invalidate(&budgets, &host, &mut reservation, &[]).await;
            })),
            Unwind::InputCharged => drop(runtime.spawn(async move {
                post_call::release_on_cancel(&budgets, &host, &mut reservation, &[]).await;
            })),
        }
    }
}

fn manifest() -> Manifest {
    Manifest {
        selection: Selection::named(NAME),
        subscriptions: vec![Subscription {
            writes: FieldSet::of(&[Field::BudgetReservation]),
            may_abort: true,
            ..Subscription::on(Hook::PreCall)
        }],
        has_request_state: true,
    }
}

fn plan_counter(counter: &PlannedCounter) -> BudgetPlanCounter {
    BudgetPlanCounter {
        counter_key: Some(counter.counter_key.clone()),
        max_budget: Some(counter.max_budget),
        fallback_spend: Some(counter.fallback_spend),
        entity_type: Some(counter.entity_type.clone()),
        entity_id: Some(counter.entity_id.clone()),
        spend_log_entity_id: counter.spend_log_entity_id.clone(),
    }
}

/// The exception `_refusal` raises for the engine's refusal, as an abort.
fn refused(kind: &str, fields: Map<String, Value>) -> Abort {
    let text = |field: &str| {
        fields
            .get(field)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned()
    };
    let (status, message, error) = match kind {
        BUDGET_EXCEEDED => (
            BUDGET_EXCEEDED_STATUS,
            text("message"),
            Some(BUDGET_EXCEEDED),
        ),
        "budget_unavailable" => (503, text("detail"), None),
        _ => (
            401,
            format!("Authentication Error, {}", text("message")),
            Some(AUTH_ERROR),
        ),
    };
    Abort {
        code: error.map(|_| status.to_string()),
        error: error.map(|kind| AbortError {
            kind: kind.to_owned(),
            fields,
        }),
        ..Abort::new(status, message)
    }
}

/// The `ProxyException` Python answers a refused reservation with, for an abort this callback
/// made.
pub(crate) fn refusal(abort: &Abort) -> Option<Refusal> {
    let kind = abort
        .error
        .as_ref()
        .map(|error| error.kind.as_str())
        .filter(|kind| [BUDGET_EXCEEDED, AUTH_ERROR].contains(kind))?;
    Some(Refusal {
        kind: Some(kind.to_owned()),
        code: abort.code.clone(),
        at_authentication: true,
        ..Refusal::new(
            StatusCode::from_u16(abort.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            abort.message.clone(),
        )
    })
}

impl Reserver {
    pub fn new(budgets: Arc<Budgets>, stored: Arc<dyn RowSpend>) -> Arc<Self> {
        Arc::new(Self {
            manifest: manifest(),
            spend: SpendReader::new(budgets.store().clone(), stored),
            budgets,
        })
    }

    pub(crate) fn registered(self: &Arc<Self>) -> Registered {
        let reserver = self.clone();
        Registered::for_caller(self.clone(), move |identity, admitted| {
            reserver.holding(identity, admitted)
        })
    }

    fn holding(
        &self,
        identity: &Authenticated<NativeRows>,
        admitted: Option<&Admitted>,
    ) -> Holding {
        Holding {
            counters: budgets::counters(identity, budgets::Settings::default()),
            priced: admitted.map(|admitted| admitted.hold).ok_or(Unheld),
            held: Mutex::new(None),
            budgets: self.budgets.clone(),
        }
    }

    /// The counters that warmed, in order. One that did not is dropped everywhere and skipped, as
    /// `_reservation_counter_loaded` skips it when budget enforcement is not fail closed.
    async fn warmed(&self, counters: &[PlannedCounter]) -> Vec<PlannedCounter> {
        let warmed = join_all(
            counters
                .iter()
                .map(async |planned| self.warm(planned).await),
        )
        .await;
        counters
            .iter()
            .zip(warmed)
            .filter(|(_, warmed)| *warmed)
            .map(|(planned, _)| planned.clone())
            .collect()
    }

    async fn warm(&self, planned: &PlannedCounter) -> bool {
        let Ok(counter) = SpendCounter::parse(&planned.counter_key) else {
            return true;
        };
        let Err(error) = self.spend.warm(&counter, planned.fallback_spend).await else {
            return true;
        };
        tracing::warn!(
            counter = planned.counter_key,
            %error,
            "Skipping budget reservation because spend counter reservation failed"
        );
        self.budgets
            .store()
            .invalidate(std::slice::from_ref(&counter))
            .await;
        false
    }

    /// What each counter holds, checked against its budget as the proxy's authentication checks
    /// it before the reservation.
    async fn observed(&self, counters: &[PlannedCounter]) -> Vec<BudgetCounterSpend> {
        join_all(counters.iter().map(async |planned| BudgetCounterSpend {
            counter_key: Some(planned.counter_key.clone()),
            spend: self.current(planned).await,
        }))
        .await
    }

    async fn current(&self, planned: &PlannedCounter) -> Option<f64> {
        let counter = SpendCounter::parse(&planned.counter_key).ok()?;
        Some(
            self.spend
                .current(&counter, planned.fallback_spend, Some(planned.max_budget))
                .await,
        )
    }

    async fn pre_call(&self, holding: &Holding) -> HookOutput {
        if holding.counters.is_empty() {
            return Ok(None);
        }
        let hold = match holding.priced {
            Ok(Some(hold)) => hold,
            Ok(None) => return Ok(None),
            Err(Unheld) => return Ok(Some(Verdict::Abort(Abort::new(500, UNHELD)))),
        };
        let counters = self.warmed(&holding.counters).await;
        if counters.is_empty() {
            return Ok(None);
        }
        let plan = BudgetPlan {
            counters: Some(counters.iter().map(plan_counter).collect()),
            reservation_cost: Some(hold.reservation_cost),
            input_cost: Some(hold.input_cost),
            input_tokens: Some(hold.input_tokens),
            fail_closed: Some(false),
            status_code: Some(BUDGET_EXCEEDED_STATUS),
            ..BudgetPlan::default()
        };
        let observed = self.observed(&counters).await;
        match self.budgets.admit(plan, &observed).await {
            Ok(Admission::Skipped) => Ok(None),
            Ok(Admission::Held { reservation }) => {
                *holding.held.lock() = Some(reservation.clone());
                let reservation = serde_json::to_value(reservation)
                    .map_err(|error| CallbackError::Failed(error.to_string()))?;
                Ok(Some(Verdict::Patch(
                    Patch::default().set(Field::BudgetReservation, reservation),
                )))
            }
            Ok(Admission::Refused { kind, fields }) => {
                Ok(Some(Verdict::Abort(refused(kind, fields))))
            }
            Err(message) => Err(CallbackError::Failed(message)),
        }
    }
}

impl Callback for Reserver {
    type State = Holding;

    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn handle<'a>(
        &'a self,
        input: &'a HookInput,
        state: Option<&'a Holding>,
    ) -> BoxFuture<'a, HookOutput> {
        Box::pin(async move {
            match (input.hook, state) {
                (Hook::PreCall, Some(holding)) => self.pre_call(holding).await,
                _ => Ok(None),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{convert::Infallible, time::Duration};

    use axum::{
        Router,
        body::{Body, Bytes},
        http::Request,
        routing::post,
    };
    use futures_util::StreamExt;
    use litellm_callbacks::{ErasedCallback, PatchOp};
    use litellm_gateway_auth_core::{
        Belongs, Caller, Credential,
        record::{Budget, EndUserRecord},
    };
    use litellm_gateway_inference::{CallHooks, record::CallRecord};
    use litellm_redis_store::{Number, spend_counters::SpendCounters};
    use rstest::rstest;
    use serde_json::json;
    use tower::ServiceExt;

    use super::*;
    use crate::{
        callbacks::Callbacks,
        host::{HoldPrices, Key, test_prices},
        test_support::master,
    };

    const CONFIG: &str = "model_list:\n  - model_name: held-model\n    litellm_params:\n      model: anthropic/held-model\n      api_key: provider-key\n  - model_name: renamed\n    litellm_params:\n      model: anthropic/held-model\n      api_key: provider-key\n";
    /// The hold for [`input`]'s call to `held-model`: nine counted input tokens and 16 output.
    const HELD: f64 = INPUT_COST + 16.0 * test_prices::OUTPUT;
    const INPUT_COST: f64 = 9.0 * test_prices::INPUT;
    /// The spend the key's row holds in the tests that drop a hold.
    const ROW: f64 = 1.0;

    /// The spend a key row holds, `None` for a read that finds none.
    struct Row(Option<f64>);

    impl RowSpend for Row {
        fn stored<'a>(&'a self, _: &'a SpendCounter) -> BoxFuture<'a, Option<f64>> {
            Box::pin(async move { self.0 })
        }
    }

    fn key(max_budget: f64) -> Authenticated<NativeRows> {
        spent_key(max_budget, 0.0)
    }

    /// A key authenticated with `spend` already made against `max_budget`.
    fn spent_key(max_budget: f64, spend: f64) -> Authenticated<NativeRows> {
        Authenticated::from(Caller {
            credential: Credential::Key {
                key: Key::stored("hash", None).with_budget(Budget {
                    max: Some(max_budget),
                    spend,
                }),
            },
            belongs: Belongs::nobody(),
            end_user: EndUserRecord::default(),
        })
    }

    fn body(model: &str) -> Map<String, Value> {
        let Value::Object(body) = json!({"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": "Say hi"}]})
        else {
            unreachable!("the body is an object")
        };
        body
    }

    /// What admit leaves on a request for [`body`]'s call to `model`, `None` for one it hands to
    /// Python since it cannot price it.
    async fn admitted(model: &str) -> Option<Admitted> {
        let hold = HoldPrices::of(CONFIG)
            .await
            .hold(model, &body(model))
            .ok()?;
        Some(Admitted { hold })
    }

    struct Reserved {
        verdict: Option<Verdict>,
        counter: Option<f64>,
    }

    fn counter() -> SpendCounter {
        SpendCounter::Key {
            token: "hash".to_owned(),
            window: None,
        }
    }

    fn counted(store: &SpendCounters) -> Option<f64> {
        store
            .get_local(&counter())
            .unwrap()
            .map(|held| held.as_f64())
    }

    /// What the key's counter holds once a dropped state's hold moved it off `held`, or after a
    /// moment when nothing moves it.
    async fn settled(store: &SpendCounters, held: Option<f64>) -> Option<f64> {
        let wait = async {
            while counted(store) == held {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        };
        let _ = tokio::time::timeout(Duration::from_millis(200), wait).await;
        counted(store)
    }

    /// A reserver over a counter that holds `warm` and a row that holds `row`.
    async fn reserver(warm: Option<f64>, row: Option<f64>) -> (Arc<SpendCounters>, Arc<Reserver>) {
        let store = Arc::new(SpendCounters::new(None));
        if let Some(spend) = warm {
            store
                .set_local(&counter(), Number::Float(spend), None)
                .unwrap();
        }
        let reserver = Reserver::new(Arc::new(Budgets::new(store.clone())), Arc::new(Row(row)));
        (store, reserver)
    }

    /// One pre-call hook for `identity`, over a counter that holds `warm` and a row that holds
    /// `row`.
    async fn reserve(
        identity: &Authenticated<NativeRows>,
        model: &str,
        warm: Option<f64>,
        row: Option<f64>,
    ) -> Reserved {
        let (store, reserver) = reserver(warm, row).await;
        let admitted = admitted(model).await;
        record::scope(CallRecord::new(), async {
            let holding = reserver.holding(identity, admitted.as_ref());
            let verdict = reserver.pre_call(&holding).await.unwrap();
            Reserved {
                verdict,
                counter: counted(&store),
            }
        })
        .await
    }

    fn reservation(verdict: Option<Verdict>) -> BudgetReservation {
        let Some(Verdict::Patch(patch)) = verdict else {
            panic!("the call was not held: {verdict:?}");
        };
        let [PatchOp::Set { field, value }] = patch.ops.as_slice() else {
            panic!("the hold wrote {:?}", patch.ops);
        };
        assert_eq!(*field, Field::BudgetReservation);
        serde_json::from_value(value.clone()).unwrap()
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_call_holds_what_it_may_cost_on_its_keys_counter() {
        let reserved = reserve(&key(10.0), "held-model", None, Some(1.0)).await;

        let reservation = reservation(reserved.verdict);
        assert_eq!(
            (
                reservation.reserved_cost,
                reservation.input_tokens,
                reservation
                    .entries
                    .unwrap_or_default()
                    .into_iter()
                    .flatten()
                    .map(|entry| entry.counter_key)
                    .collect::<Vec<_>>(),
                reserved.counter,
            ),
            (
                Some(HELD),
                Some(9),
                vec![Some("spend:key:hash".to_owned())],
                Some(1.0 + HELD),
            )
        );
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_cold_counter_starts_from_its_rows_spend_and_a_spent_key_is_refused() {
        let reserved = reserve(&key(1e-05), "held-model", None, Some(1e-05)).await;

        let Some(Verdict::Abort(abort)) = reserved.verdict else {
            panic!("the call was held: {:?}", reserved.verdict);
        };
        let refused = refusal(&abort).unwrap();
        assert_eq!(
            (
                refused.status,
                refused.kind.as_deref(),
                refused.code.as_deref(),
                refused.at_authentication,
                reserved
                    .counter
                    .is_some_and(|spend| (spend - 1e-05).abs() < 1e-12),
            ),
            (
                StatusCode::UNPROCESSABLE_ENTITY,
                Some(BUDGET_EXCEEDED),
                Some("422"),
                true,
                true,
            )
        );
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_warm_counter_is_not_started_again_from_its_row() {
        let reserved = reserve(&key(10.0), "held-model", Some(2.0), Some(10.0)).await;

        assert_eq!(
            (
                reservation(reserved.verdict).reserved_cost,
                reserved.counter
            ),
            (Some(HELD), Some(2.0 + HELD))
        );
    }

    /// `ensure_spend_counter_initialized` raises a cold counter whose key has no row to the spend
    /// the key was authenticated with before any call is held on it, so two calls racing on it
    /// cannot both fit a budget only one of them fits.
    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn concurrent_calls_on_a_cold_counter_without_a_row_start_from_the_keys_spend() {
        let identity = spent_key(1.0 + HELD, 1.0);
        let (store, reserver) = reserver(None, None).await;
        let admitted = admitted("held-model").await;
        let both_reserved = tokio::sync::Barrier::new(2);
        let call = || {
            record::scope(CallRecord::new(), async {
                let holding = reserver.holding(&identity, admitted.as_ref());
                let verdict = reserver.pre_call(&holding).await.unwrap();
                both_reserved.wait().await;
                let held = counted(&store);
                let outcome = match verdict {
                    Some(Verdict::Patch(_)) => "held",
                    Some(Verdict::Abort(abort)) => match refusal(&abort) {
                        Some(refused) if refused.kind.as_deref() == Some(BUDGET_EXCEEDED) => {
                            "refused"
                        }
                        _ => "aborted",
                    },
                    None => "skipped",
                };
                (outcome, held)
            })
        };

        let ((first, held), (second, _)) = tokio::join!(call(), call());

        let mut outcomes = [first, second];
        outcomes.sort_unstable();
        assert_eq!(
            (
                outcomes,
                held.is_some_and(|spend| (spend - (1.0 + HELD)).abs() < 1e-12),
            ),
            (["held", "refused"], true)
        );
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_call_the_gateway_cannot_hold_is_refused_never_let_through() {
        let reserved = reserve(&key(10.0), "renamed", None, None).await;

        assert_eq!(
            (reserved.verdict, reserved.counter),
            (Some(Verdict::Abort(Abort::new(500, UNHELD))), None)
        );
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_caller_with_no_budget_holds_nothing() {
        let reserved = reserve(&master(), "claude", None, None).await;

        assert_eq!((reserved.verdict, reserved.counter), (None, None));
    }

    fn failed(entries: &mut CallState) {
        entries.call_provider();
        entries.fail();
    }

    fn never_called(_: &mut CallState) {}

    fn abandoned(entries: &mut CallState) {
        entries.call_provider();
    }

    fn finalized(entries: &mut CallState) {
        entries.call_provider();
        entries.set_budget_reservation(json!({"finalized": true}));
    }

    #[rstest]
    #[case::a_failed_call(failed, ROW)]
    #[case::a_call_dropped_before_the_provider(never_called, ROW)]
    #[case::a_call_abandoned_after_the_provider(abandoned, ROW + INPUT_COST)]
    #[case::a_hold_the_spend_logger_finalized(finalized, ROW + HELD)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_hold_is_given_back_by_how_the_call_ended_when_its_request_state_is_dropped(
        #[case] ended: fn(&mut CallState),
        #[case] left: f64,
    ) {
        let (store, reserver) = reserver(None, Some(ROW)).await;

        let admitted = admitted("held-model").await;
        let held = record::scope(CallRecord::new(), async {
            let holding = reserver.holding(&key(10.0), admitted.as_ref());
            reservation(reserver.pre_call(&holding).await.unwrap());
            let held = counted(&store);
            record::write(ended);
            drop(holding);
            held
        })
        .await;

        assert_eq!(
            (held, settled(&store, held).await),
            (Some(ROW + HELD), Some(left))
        );
    }

    /// Refuses every call on the pre-call hook.
    struct Refuses(Manifest);

    impl Callback for Refuses {
        type State = ();

        fn manifest(&self) -> &Manifest {
            &self.0
        }

        fn handle<'a>(&'a self, _: &'a HookInput, _: Option<&'a ()>) -> BoxFuture<'a, HookOutput> {
            Box::pin(async { Ok(Some(Verdict::Abort(Abort::new(429, "slow down")))) })
        }
    }

    fn callbacks(reserver: &Arc<Reserver>) -> Arc<Callbacks> {
        let refuses = Refuses(Manifest {
            selection: Selection::named("refuses"),
            subscriptions: vec![Subscription {
                may_abort: true,
                ..Subscription::on(Hook::PreCall)
            }],
            has_request_state: false,
        });
        Arc::new(
            Callbacks::new(vec![
                reserver.registered(),
                Registered::from(Arc::new(refuses) as Arc<dyn ErasedCallback>),
            ])
            .unwrap(),
        )
    }

    /// Moves the call on the pre-call hook to `renamed`, a group the gateway does not price.
    struct Renames(Manifest);

    impl Callback for Renames {
        type State = ();

        fn manifest(&self) -> &Manifest {
            &self.0
        }

        fn handle<'a>(&'a self, _: &'a HookInput, _: Option<&'a ()>) -> BoxFuture<'a, HookOutput> {
            Box::pin(async {
                Ok(Some(Verdict::Patch(
                    Patch::default().set(Field::Model, json!("renamed")),
                )))
            })
        }
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_call_holds_what_admit_priced_whatever_an_earlier_hook_writes() {
        let (store, reserver) = reserver(None, Some(ROW)).await;
        let renames = Renames(Manifest {
            selection: Selection::named("renames"),
            subscriptions: vec![Subscription {
                writes: FieldSet::of(&[Field::Model]),
                ..Subscription::on(Hook::PreCall)
            }],
            has_request_state: false,
        });
        let callbacks = Arc::new(
            Callbacks::new(vec![
                Registered::from(Arc::new(renames) as Arc<dyn ErasedCallback>),
                reserver.registered(),
            ])
            .unwrap(),
        );

        let (served, held) = record::scope(CallRecord::new(), async {
            let call = callbacks.begin(&key(10.0), admitted("held-model").await.as_ref());
            let served = call.pre_call(body("held-model")).await.is_ok();
            (served, counted(&store))
        })
        .await;

        assert_eq!((served, held), (true, Some(ROW + HELD)));
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_call_a_later_callback_refuses_gives_its_hold_back() {
        let (store, reserver) = reserver(None, Some(ROW)).await;
        let callbacks = callbacks(&reserver);

        let (refused, held) = record::scope(CallRecord::new(), async {
            let call = callbacks.begin(&key(10.0), admitted("held-model").await.as_ref());
            let refused = call.pre_call(body("held-model")).await.is_err();
            let held = counted(&store);
            call.failed().await;
            (refused, held)
        })
        .await;

        assert_eq!(
            (refused, held, settled(&store, held).await),
            (true, Some(ROW + HELD), Some(ROW))
        );
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_call_whose_response_body_is_dropped_keeps_only_its_input_cost() {
        let (store, reserver) = reserver(None, Some(ROW)).await;
        let callbacks = Arc::new(Callbacks::new(vec![reserver.registered()]).unwrap());
        let app = Router::new()
            .route(
                "/",
                post(move || async move {
                    let call = callbacks.begin(&key(10.0), admitted("held-model").await.as_ref());
                    call.pre_call(body("held-model")).await.unwrap();
                    let stream = futures_util::stream::pending::<Result<Bytes, Infallible>>().map(
                        move |frame| {
                            let _ = &call;
                            frame
                        },
                    );
                    Body::from_stream(stream)
                }),
            )
            .layer(axum::middleware::from_fn(record::scoped));

        let response = app
            .oneshot(Request::post("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let held = counted(&store);
        drop(response);

        assert_eq!(
            (held, settled(&store, held).await),
            (Some(ROW + HELD), Some(ROW + INPUT_COST))
        );
    }
}
