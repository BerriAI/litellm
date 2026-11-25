//! The budget work a spend log does when a call is not charged: releasing the reservation
//! admission held.
//!
//! [`release`] ports `release_budget_reservation` (`reconcile_budget_reservation` to nothing spent,
//! with `settle_native_reservation`): the reserved counters are read, [`crate::settle`] decides
//! each entry's adjustment and applies the consistent ones, a counter it found cold is reseeded
//! from its stored spend through the host (`reseed_reserved_counter`), the targets are stamped
//! on the reservation and it is finalized. A finalized reservation is left alone. A charged call
//! settles its reservation in the counter step instead ([`crate::charge_counters`]).

use litellm_callbacks::{BudgetReservation, Host};
use litellm_redis_store::{
    Number,
    spend_counters::{SpendCounter, SpendCounters},
};

use crate::{
    Budgets, EngineError, Settle,
    charge::{held_counters, reseed_reserved},
    engine::{self},
    window::Moment,
};

fn reserved_counter_keys(reservation: &BudgetReservation) -> Vec<Option<String>> {
    reservation
        .entries
        .iter()
        .flatten()
        .map(|entry| entry.as_ref().and_then(|entry| entry.counter_key.clone()))
        .collect()
}

async fn observed(store: &SpendCounters, keys: &[Option<String>]) -> Vec<Option<f64>> {
    let mut values = Vec::with_capacity(keys.len());
    for key in keys {
        let value = match key.as_deref().map(SpendCounter::parse) {
            Some(Ok(counter)) => store
                .read(&counter)
                .await
                .ok()
                .and_then(|reading| reading.value())
                .map(Number::as_f64),
            _ => None,
        };
        values.push(value);
    }
    values
}

/// `release_budget_reservation`: reconcile to nothing spent and finalize. `window_resets` holds
/// each entry's `window_reset_at`, which the typed record does not carry.
pub async fn release(
    budgets: &Budgets,
    host: &Host,
    reservation: &mut BudgetReservation,
    window_resets: &[Option<String>],
) -> Result<(), EngineError> {
    reconcile(budgets, host, reservation, window_resets, 0.0).await
}

/// `release_budget_reservation_on_cancel`: a call abandoned after the provider was asked keeps
/// its input cost, since the provider bills the input tokens it was sent.
pub async fn release_on_cancel(
    budgets: &Budgets,
    host: &Host,
    reservation: &mut BudgetReservation,
    window_resets: &[Option<String>],
) {
    let input_cost = reservation.input_cost.unwrap_or(0.0);
    if let Err(error) = reconcile(budgets, host, reservation, window_resets, input_cost).await {
        tracing::debug!(%error, "Failed to release a cancelled budget reservation");
    }
}

/// `reconcile_budget_reservation`: settle each held entry at `actual` and finalize.
async fn reconcile(
    budgets: &Budgets,
    host: &Host,
    reservation: &mut BudgetReservation,
    window_resets: &[Option<String>],
    actual: f64,
) -> Result<(), EngineError> {
    if reservation.finalized == Some(true) {
        return Ok(());
    }
    let store = budgets.store();
    let keys = reserved_counter_keys(reservation);
    let observed = observed(store, &keys).await;
    let settled = engine::settle(
        store,
        &Settle {
            reservation,
            window_resets,
            observed: &observed,
            actual,
            apply_consistent: true,
        },
        &Moment::now(),
    )
    .await?;
    if let Err(error) =
        reseed_reserved(store, host, &held_counters(reservation), &settled.reseed).await
    {
        tracing::debug!(%error, "Unable to reseed a cold reserved counter on release");
    }
    if let Some(entries) = reservation.entries.as_mut() {
        for (entry, target) in entries.iter_mut().zip(&settled.adjustments) {
            if let (Some(entry), Some(target)) = (entry.as_mut(), target) {
                entry.applied_adjustment = Some(*target);
            }
        }
    }
    reservation.finalized = Some(true);
    Ok(())
}

/// `invalidate_budget_reservation_counters`: drop every reserved counter so the next read
/// reseeds it.
pub async fn invalidate(store: &SpendCounters, reservation: &BudgetReservation) {
    let counters: Vec<SpendCounter> = reserved_counter_keys(reservation)
        .into_iter()
        .flatten()
        .filter_map(|key| SpendCounter::parse(&key).ok())
        .collect();
    store.invalidate(&counters).await;
}

/// [`release`], falling back to invalidating the reserved counters and finalizing the
/// reservation when the release fails, as the cost callback does.
pub async fn release_or_invalidate(
    budgets: &Budgets,
    host: &Host,
    reservation: &mut BudgetReservation,
    window_resets: &[Option<String>],
) {
    if let Err(error) = release(budgets, host, reservation, window_resets).await {
        tracing::error!(%error, "Failed to release budget reservation");
        invalidate(budgets.store(), reservation).await;
        reservation.finalized = Some(true);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use litellm_callbacks::{HostRequest, ReservationEntry, hosted};
    use litellm_coroutine::CoroutineState;
    use rstest::rstest;

    use super::*;

    const KEY: &str = "spend:key:hashed";

    fn budgets(seed: Option<f64>) -> Budgets {
        let store = Arc::new(SpendCounters::new(None));
        if let Some(value) = seed {
            store
                .set_local(
                    &SpendCounter::parse(KEY).unwrap(),
                    Number::Float(value),
                    None,
                )
                .unwrap();
        }
        Budgets::new(store)
    }

    fn reservation(reserved: f64) -> BudgetReservation {
        BudgetReservation {
            reserved_cost: Some(reserved),
            entries: Some(vec![Some(ReservationEntry {
                counter_key: Some(KEY.to_owned()),
                entity_type: Some("key".to_owned()),
                entity_id: Some("hashed".to_owned()),
                reserved_cost: Some(reserved),
                applied_adjustment: Some(0.0),
                ..ReservationEntry::default()
            })]),
            ..BudgetReservation::default()
        }
    }

    fn value(budgets: &Budgets) -> Option<f64> {
        budgets
            .store()
            .get_local(&SpendCounter::parse(KEY).unwrap())
            .unwrap()
            .map(Number::as_f64)
    }

    #[rstest]
    #[tokio::test]
    async fn a_release_gives_the_reserved_cost_back_and_finalizes() {
        let budgets = budgets(Some(5.0));
        let mut held = reservation(2.0);

        release(&budgets, &Host::detached(), &mut held, &[None])
            .await
            .unwrap();

        assert_eq!(value(&budgets), Some(3.0));
        assert_eq!(held.finalized, Some(true));
        let entry = held.entries.unwrap()[0].clone().unwrap();
        assert_eq!(entry.applied_adjustment, Some(-2.0));
    }

    #[rstest]
    #[tokio::test]
    async fn a_cancelled_call_keeps_its_input_cost_and_gives_the_rest_back() {
        let budgets = budgets(Some(5.0));
        let mut held = BudgetReservation {
            input_cost: Some(0.5),
            ..reservation(2.0)
        };

        release_on_cancel(&budgets, &Host::detached(), &mut held, &[None]).await;

        assert_eq!((value(&budgets), held.finalized), (Some(3.5), Some(true)));
    }

    #[rstest]
    #[tokio::test]
    async fn a_finalized_reservation_is_left_alone() {
        let budgets = budgets(Some(5.0));
        let mut held = BudgetReservation {
            finalized: Some(true),
            ..reservation(2.0)
        };

        release(&budgets, &Host::detached(), &mut held, &[None])
            .await
            .unwrap();

        assert_eq!(value(&budgets), Some(5.0));
    }

    #[rstest]
    #[tokio::test]
    async fn a_cold_reserved_counter_is_reseeded_from_its_stored_spend_on_release() {
        let budgets = Arc::new(budgets(None));
        let released = Arc::clone(&budgets);
        let mut body = hosted(move |host| async move {
            let mut held = reservation(2.0);
            release(&released, &host, &mut held, &[None]).await.unwrap();
            held
        });
        let mut asked = Vec::new();
        let held = loop {
            match body.resume().await.unwrap() {
                CoroutineState::Yielded(HostRequest { call, reply, .. }) => {
                    asked.push(call);
                    reply.send(Ok(serde_json::json!([7.0])));
                }
                CoroutineState::Complete(held) => break held,
            }
        };

        assert_eq!(asked, ["stored_spend"]);
        assert_eq!(value(&budgets), Some(7.0));
        assert_eq!(held.finalized, Some(true));
    }

    #[rstest]
    #[tokio::test]
    async fn invalidating_drops_the_reserved_counters() {
        let budgets = budgets(Some(5.0));

        invalidate(budgets.store(), &reservation(2.0)).await;

        assert_eq!(value(&budgets), None);
    }
}
