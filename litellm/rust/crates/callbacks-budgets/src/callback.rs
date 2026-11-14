use std::sync::Arc;

use futures_util::future::BoxFuture;
use litellm_callbacks::{
    Abort, AbortError, Activation, BudgetReservation, Callback, CallbackError, Field, FieldSet,
    Hook, HookInput, HookOutput, Host, Manifest, OptOutBy, Patch, ReservationEntry, Selection,
    Subscription, Verdict, typed,
};
use litellm_redis_store::spend_counters::SpendCounters;
use serde_json::{Map, json};

use crate::{
    engine::{self, EngineError, Reservation, Settlement, UNAVAILABLE},
    plan::Plan,
    window::Moment,
};

pub const NAME: &str = "budgets";

pub struct Budgets {
    store: Arc<SpendCounters>,
    manifest: Manifest,
}

fn manifest() -> Manifest {
    Manifest {
        selection: Selection {
            activation: Activation::Required,
            opt_out_by: vec![OptOutBy::Admin],
            ..Selection::named(NAME)
        },
        subscriptions: vec![Subscription {
            reads: FieldSet::of(&[Field::BudgetPlan]),
            writes: FieldSet::of(&[Field::BudgetReservation]),
            may_abort: true,
            ..Subscription::on(Hook::PreCall)
        }],
        has_request_state: false,
    }
}

fn abort(error: &EngineError, status: u16) -> Abort {
    let (status, kind, fields) = match error {
        EngineError::Exceeded(exceeded) => (
            status,
            "budget_exceeded",
            Map::from_iter([
                ("current_cost".to_owned(), json!(exceeded.current_cost)),
                ("max_budget".to_owned(), json!(exceeded.max_budget)),
                ("message".to_owned(), json!(exceeded.message)),
                ("entity_type".to_owned(), json!(exceeded.entity_type)),
                ("entity_id".to_owned(), json!(exceeded.entity_id)),
            ]),
        ),
        EngineError::Unavailable => (
            503,
            "budget_unavailable",
            Map::from_iter([("detail".to_owned(), json!(UNAVAILABLE))]),
        ),
        other => (
            401,
            "budget_reservation_failed",
            Map::from_iter([("message".to_owned(), json!(other.to_string()))]),
        ),
    };
    Abort {
        error: Some(AbortError {
            kind: kind.to_owned(),
            fields,
        }),
        ..Abort::new(status, error.to_string())
    }
}

fn reservation_record(reservation: &Reservation) -> BudgetReservation {
    BudgetReservation {
        reserved_cost: Some(reservation.reserved_cost),
        entries: Some(
            reservation
                .entries
                .iter()
                .map(|entry| {
                    Some(ReservationEntry {
                        counter_key: Some(entry.counter_key.clone()),
                        entity_type: Some(entry.entity_type.clone()),
                        entity_id: Some(entry.entity_id.clone()),
                        reserved_cost: Some(entry.reserved_cost),
                        applied_adjustment: Some(entry.applied_adjustment),
                        ..ReservationEntry::default()
                    })
                })
                .collect(),
        ),
        input_cost: Some(reservation.input_cost),
        input_tokens: reservation.input_tokens,
        ..BudgetReservation::default()
    }
}

impl Budgets {
    pub fn new(store: Arc<SpendCounters>) -> Self {
        Self {
            store,
            manifest: manifest(),
        }
    }

    pub async fn settle(
        &self,
        reservation: &BudgetReservation,
        window_resets: &[Option<String>],
        actual: f64,
    ) -> Result<Settlement, EngineError> {
        engine::settle(
            &self.store,
            reservation,
            window_resets,
            actual,
            &Moment::now(),
        )
        .await
    }

    async fn admit(&self, input: &HookInput, host: Host) -> HookOutput {
        let Some(record) = input.fields.get::<typed::BudgetPlan>()? else {
            return Ok(None);
        };
        let plan =
            Plan::try_from(record).map_err(|error| CallbackError::Failed(error.to_string()))?;
        match engine::reserve(&self.store, &plan, &host).await {
            Ok(None) => Ok(None),
            Ok(Some(reservation)) => {
                let record = serde_json::to_value(reservation_record(&reservation))
                    .map_err(|error| CallbackError::Failed(error.to_string()))?;
                Ok(Some(Verdict::Patch(
                    Patch::default().set(Field::BudgetReservation, record),
                )))
            }
            Err(EngineError::Host(error)) => Err(CallbackError::Host(error)),
            Err(error) => Ok(Some(Verdict::Abort(abort(&error, plan.status_code)))),
        }
    }
}

pub struct Shared(pub Arc<Budgets>);

impl Callback for Shared {
    type State = ();

    fn manifest(&self) -> &Manifest {
        &self.0.manifest
    }

    fn handle<'a>(
        &'a self,
        input: &'a HookInput,
        state: Option<&'a ()>,
    ) -> BoxFuture<'a, HookOutput> {
        self.handle_hosted(input, state, Host::detached())
    }

    fn handle_hosted<'a>(
        &'a self,
        input: &'a HookInput,
        _state: Option<&'a ()>,
        host: Host,
    ) -> BoxFuture<'a, HookOutput> {
        Box::pin(async move {
            match input.hook {
                Hook::PreCall => self.0.admit(input, host).await,
                _ => Ok(None),
            }
        })
    }
}
