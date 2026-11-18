use std::sync::Arc;

use litellm_callbacks::{BudgetPlan, BudgetReservation, Host, ReservationEntry};
use litellm_redis_store::{RequestRedis, spend_counters::SpendCounters};
use litellm_tracing::Logger;
use serde::Serialize;
use serde_json::{Map, Value, json};

use crate::{
    charge::{self, CounterOutcome, CounterStep},
    engine::{self, BudgetCounterSpend, EngineError, Reservation, Settle, Settlement, UNAVAILABLE},
    plan::Plan,
    window::Moment,
};

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "admission", rename_all = "snake_case")]
pub enum Admission {
    Skipped,
    Held {
        reservation: BudgetReservation,
    },
    Refused {
        kind: &'static str,
        fields: Map<String, Value>,
    },
}

/// The budget engine over one spend counter store: admission holds, settlements, and the post-call
/// counter step.
pub struct Budgets {
    store: Arc<SpendCounters>,
    logger: Option<Logger>,
}

fn refusal(error: &EngineError) -> Admission {
    let (kind, fields) = match error {
        EngineError::Exceeded(exceeded) => (
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
            "budget_unavailable",
            Map::from_iter([("detail".to_owned(), json!(UNAVAILABLE))]),
        ),
        other => (
            "budget_reservation_failed",
            Map::from_iter([("message".to_owned(), json!(other.to_string()))]),
        ),
    };
    Admission::Refused { kind, fields }
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
            logger: None,
        }
    }

    pub fn with_logger(self, logger: Logger) -> Self {
        Self {
            logger: Some(logger),
            ..self
        }
    }

    async fn logged<T>(&self, future: impl Future<Output = T>) -> T {
        match &self.logger {
            Some(logger) => logger.instrument(future).await,
            None => future.await,
        }
    }

    pub async fn settle(&self, request: &Settle<'_>) -> Result<Settlement, EngineError> {
        self.logged(engine::settle(&self.store, request, &Moment::now()))
            .await
    }

    /// The whole post-call counter step for one charged request; see [`charge::charge_counters`].
    pub async fn charge_counters(
        &self,
        host: &Host,
        request: Option<&RequestRedis>,
        step: &CounterStep,
        reservation: Option<&BudgetReservation>,
    ) -> CounterOutcome {
        self.logged(charge::charge_counters(
            &self.store,
            host,
            request,
            step,
            reservation,
        ))
        .await
    }

    pub async fn admit(
        &self,
        plan: BudgetPlan,
        observed: &[BudgetCounterSpend],
    ) -> Result<Admission, String> {
        let plan = Plan::try_from(plan).map_err(|error| error.to_string())?;
        Ok(
            match self
                .logged(engine::reserve(&self.store, &plan, observed))
                .await
            {
                Ok(None) => Admission::Skipped,
                Ok(Some(reservation)) => Admission::Held {
                    reservation: reservation_record(&reservation),
                },
                Err(error) => refusal(&error),
            },
        )
    }
}
