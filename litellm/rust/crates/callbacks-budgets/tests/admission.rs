use std::sync::Arc;

use litellm_callbacks::{BudgetPlan, BudgetPlanCounter, BudgetReservation};
use litellm_callbacks_budgets::{Admission, BudgetCounterSpend, Budgets, Settle, Settlement};
use litellm_redis_store::{
    Number,
    spend_counters::{SpendCounter, SpendCounters},
};
use rstest::rstest;
use serde_json::json;

fn counter(key: &str, max_budget: f64, entity_type: &str, entity_id: &str) -> BudgetPlanCounter {
    BudgetPlanCounter {
        counter_key: Some(key.to_owned()),
        max_budget: Some(max_budget),
        fallback_spend: Some(0.0),
        entity_type: Some(entity_type.to_owned()),
        entity_id: Some(entity_id.to_owned()),
        spend_log_entity_id: None,
    }
}

fn plan(counters: Vec<BudgetPlanCounter>, cost: f64) -> BudgetPlan {
    BudgetPlan {
        counters: Some(counters),
        reservation_cost: Some(cost),
        input_cost: Some(cost / 4.0),
        input_tokens: Some(12),
        throttle_counter: None,
        fail_closed: Some(false),
        status_code: Some(400),
        reservation_id: Some("reservation-1".to_owned()),
        ..BudgetPlan::default()
    }
}

fn store(seed: &[(&str, f64)]) -> Arc<SpendCounters> {
    let store = Arc::new(SpendCounters::new(None));
    for (key, value) in seed {
        store
            .set_local(
                &SpendCounter::parse(key).unwrap(),
                Number::Float(*value),
                None,
            )
            .unwrap();
    }
    store
}

fn value(store: &SpendCounters, key: &str) -> Option<f64> {
    store
        .get_local(&SpendCounter::parse(key).unwrap())
        .unwrap()
        .map(Number::as_f64)
}

const KEY: &str = "spend:key:hashed";

fn observed(spends: &[(&str, f64)]) -> Vec<BudgetCounterSpend> {
    spends
        .iter()
        .map(|(key, spend)| BudgetCounterSpend {
            counter_key: Some((*key).to_owned()),
            spend: Some(*spend),
        })
        .collect()
}

async fn admit(budgets: &Budgets, plan: &BudgetPlan, spends: &[(&str, f64)]) -> Admission {
    budgets
        .admit(plan.clone(), &observed(spends))
        .await
        .unwrap()
}

async fn held(budgets: &Budgets, plan: &BudgetPlan) -> BudgetReservation {
    let Admission::Held { reservation } = admit(budgets, plan, &[]).await else {
        panic!("expected a reservation");
    };
    reservation
}

fn stamp(record: &mut BudgetReservation, adjustments: &[Option<f64>]) {
    for (entry, target) in record.entries.iter_mut().flatten().zip(adjustments) {
        if let (Some(entry), Some(target)) = (entry, target) {
            entry.applied_adjustment = Some(*target);
        }
    }
}

async fn settle(
    budgets: &Budgets,
    store: &SpendCounters,
    record: &BudgetReservation,
    actual: f64,
) -> Settlement {
    let mut observed = Vec::new();
    for entry in record.entries.iter().flatten().flatten() {
        let counter = SpendCounter::parse(entry.counter_key.as_deref().unwrap()).unwrap();
        observed.push(
            store
                .read(&counter)
                .await
                .unwrap()
                .value()
                .map(Number::as_f64),
        );
    }
    budgets
        .settle(&Settle {
            reservation: record,
            window_resets: &[],
            observed: &observed,
            actual,
            apply_consistent: true,
        })
        .await
        .unwrap()
}

#[rstest]
#[tokio::test]
async fn an_admission_holds_and_returns_the_hold_as_the_reservation() {
    let store = store(&[(KEY, 1.0)]);
    let budgets = Budgets::new(Arc::clone(&store));
    let plan = plan(vec![counter(KEY, 10.0, "Key", "hashed")], 2.0);
    let mut record = held(&budgets, &plan).await;
    assert_eq!(value(&store, KEY), Some(3.0));
    assert_eq!(record.reserved_cost, Some(2.0));
    assert_eq!(record.input_cost, Some(0.5));
    assert_eq!(record.input_tokens, Some(12));
    assert_eq!(record.finalized, None);
    let settled = settle(&budgets, &store, &record, 0.5).await;
    assert_eq!(settled.adjustments, vec![Some(-1.5)]);
    assert!(settled.reseed.is_empty());
    assert_eq!(value(&store, KEY), Some(1.5));
    stamp(&mut record, &settled.adjustments);
    let again = settle(&budgets, &store, &record, 0.5).await;
    assert_eq!(again.adjustments, vec![None]);
    assert_eq!(value(&store, KEY), Some(1.5));
}

#[rstest]
#[case::no_counters(BudgetPlan { counters: None, ..plan(vec![], 2.0) })]
#[case::no_status_code(BudgetPlan { status_code: None, ..plan(vec![counter(KEY, 10.0, "Key", "hashed")], 2.0) })]
#[tokio::test]
async fn a_plan_missing_a_required_member_fails_without_holding(#[case] plan: BudgetPlan) {
    let store = store(&[(KEY, 1.0)]);
    let budgets = Budgets::new(Arc::clone(&store));
    assert!(budgets.admit(plan, &[]).await.is_err());
    assert_eq!(value(&store, KEY), Some(1.0));
}

#[rstest]
#[case::once(&[0.25])]
#[case::twice_on_the_same_cost(&[0.25, 0.25])]
#[case::grown_then_shrunk(&[0.75, 0.25])]
#[tokio::test]
async fn a_settlement_through_the_reservation_dict_lands_on_the_last_cost(#[case] costs: &[f64]) {
    let store = store(&[(KEY, 1.0)]);
    let budgets = Budgets::new(Arc::clone(&store));
    let plan = plan(vec![counter(KEY, 10.0, "Key", "hashed")], 2.0);
    let mut record = held(&budgets, &plan).await;
    for cost in costs {
        let settled = settle(&budgets, &store, &record, *cost).await;
        stamp(&mut record, &settled.adjustments);
    }
    assert_eq!(value(&store, KEY), Some(1.0 + costs[costs.len() - 1]));
    assert_eq!(record.finalized, None);
}

#[rstest]
#[tokio::test]
async fn a_refusal_returns_the_typed_budget_error() {
    let store = store(&[(KEY, 10.0)]);
    let budgets = Budgets::new(Arc::clone(&store));
    let plan = plan(vec![counter(KEY, 10.0, "Key", "hashed")], 2.0);
    let Admission::Refused { kind, fields } = admit(&budgets, &plan, &[]).await else {
        panic!("expected a refusal");
    };
    assert_eq!(kind, "budget_exceeded");
    assert_eq!(
        serde_json::Value::Object(fields),
        json!({
            "current_cost": 12.0,
            "max_budget": 10.0,
            "message": "Budget has been exceeded! Key=hashed Current cost: 12.0, Max budget: 10.0",
            "entity_type": "key",
            "entity_id": "hashed",
        })
    );
    assert_eq!(value(&store, KEY), Some(10.0));
}

#[rstest]
#[case::observed(&[(KEY, 1.0)])]
#[case::not_observed(&[])]
#[tokio::test]
async fn a_counter_python_observed_is_charged_on_the_store(#[case] observed: &[(&str, f64)]) {
    let store = store(&[(KEY, 1.0)]);
    let budgets = Budgets::new(Arc::clone(&store));
    let plan = plan(vec![counter(KEY, 10.0, "Key", "hashed")], 2.0);
    let Admission::Held { .. } = admit(&budgets, &plan, observed).await else {
        panic!("expected the reservation");
    };
    assert_eq!(value(&store, KEY), Some(3.0));
}

#[rstest]
#[tokio::test]
async fn an_observed_spend_with_no_room_charges_counters_one_at_a_time_and_refuses_the_full_one() {
    const TEAM: &str = "spend:team:team-1";
    let store = store(&[(KEY, 10.0), (TEAM, 0.0)]);
    let budgets = Budgets::new(Arc::clone(&store));
    let plan = plan(
        vec![
            counter(TEAM, 10.0, "Team", "team-1"),
            counter(KEY, 10.0, "Key", "hashed"),
        ],
        2.0,
    );
    let Admission::Refused { kind, .. } = admit(&budgets, &plan, &[(KEY, 10.0), (TEAM, 0.0)]).await
    else {
        panic!("expected the refusal");
    };
    assert_eq!(kind, "budget_exceeded");
    assert_eq!(value(&store, KEY), Some(10.0));
    assert_eq!(value(&store, TEAM), Some(0.0));
}
