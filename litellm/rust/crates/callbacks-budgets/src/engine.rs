use std::sync::atomic::{AtomicUsize, Ordering};

use litellm_callbacks::{
    BudgetReservation, Host, HostCall, HostCallError, ReservationEntry, Scalar,
};
use litellm_python_compat::repr::float_repr;
use litellm_redis_store::{
    Number,
    spend_counters::{NotASpendCounter, SpendCounter, SpendCounters},
};
use serde::Serialize;

use crate::{
    plan::{Counter, Plan, entity_type_value},
    window::{self, Moment, ReservationError},
};

const EPSILON: f64 = 1e-12;

static ENTRY_IDS: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Entry {
    pub counter_key: String,
    pub entity_type: String,
    pub entity_id: String,
    pub reserved_cost: f64,
    pub applied_adjustment: f64,
    #[serde(skip)]
    id: usize,
    #[serde(skip)]
    counter: SpendCounter,
}

impl Entry {
    pub fn counter(&self) -> &SpendCounter {
        &self.counter
    }

    fn new(counter: &Counter, reserved_cost: f64) -> Self {
        Self {
            counter_key: counter.planned.counter_key.clone(),
            entity_type: counter.planned.entity_type.clone(),
            entity_id: counter.planned.entity_id.clone(),
            reserved_cost,
            applied_adjustment: 0.0,
            id: ENTRY_IDS.fetch_add(1, Ordering::Relaxed),
            counter: counter.counter.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Reservation {
    pub reserved_cost: f64,
    pub entries: Vec<Entry>,
    pub input_cost: f64,
    pub input_tokens: Option<i64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Exceeded {
    pub current_cost: f64,
    pub max_budget: f64,
    pub message: String,
    pub entity_type: Option<&'static str>,
    pub entity_id: String,
}

#[derive(Debug)]
pub enum EngineError {
    Exceeded(Exceeded),
    Unavailable,
    Inconsistent(String),
    Store(litellm_redis_store::Error),
    Plan(NotASpendCounter),
    Reservation(ReservationError),
    Host(HostCallError),
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Exceeded(exceeded) => f.write_str(&exceeded.message),
            Self::Unavailable => f.write_str(UNAVAILABLE),
            Self::Inconsistent(message) => f.write_str(message),
            Self::Store(error) => write!(f, "{error}"),
            Self::Plan(error) => write!(f, "{error}"),
            Self::Reservation(error) => write!(f, "{error}"),
            Self::Host(error) => write!(f, "{error}"),
        }
    }
}

impl From<ReservationError> for EngineError {
    fn from(error: ReservationError) -> Self {
        Self::Reservation(error)
    }
}

impl From<litellm_redis_store::Error> for EngineError {
    fn from(error: litellm_redis_store::Error) -> Self {
        Self::Store(error)
    }
}

pub const UNAVAILABLE: &str = "Budget enforcement unavailable: the budget reservation could not be written to the spend counter backend, and fail_closed_budget_enforcement is enabled, so the request was rejected to avoid exceeding the configured budget. Retry shortly.";

fn exceeded(counter: &Counter, current_cost: f64, estimated_cost: Option<f64>) -> EngineError {
    let planned = &counter.planned;
    let estimate = estimated_cost
        .map(|cost| format!("Estimated request cost: {}, ", float_repr(cost)))
        .unwrap_or_default();
    EngineError::Exceeded(Exceeded {
        current_cost,
        max_budget: planned.max_budget,
        message: format!(
            "Budget has been exceeded! {}={} Current cost: {}, {}Max budget: {}",
            planned.entity_type,
            planned.entity_id,
            float_repr(current_cost),
            estimate,
            float_repr(planned.max_budget),
        ),
        entity_type: entity_type_value(&planned.entity_type),
        entity_id: planned
            .spend_log_entity_id
            .clone()
            .unwrap_or_else(|| planned.entity_id.clone()),
    })
}

#[derive(Serialize)]
pub struct CurrentSpend {
    pub counter_key: String,
    pub fallback_spend: Scalar,
}

impl HostCall for CurrentSpend {
    const NAME: &'static str = "current_spend";
    type Reply = Scalar;
}

async fn current_value(
    store: &SpendCounters,
    counter: &Counter,
    host: &Host,
) -> Result<f64, EngineError> {
    if let Some(value) = store.read(&counter.counter).await?.value() {
        return Ok(value.as_f64());
    }
    if !host.is_attached() {
        return Ok(counter.planned.fallback_spend);
    }
    let call = CurrentSpend {
        counter_key: counter.planned.counter_key.clone(),
        fallback_spend: Scalar::Float(counter.planned.fallback_spend),
    };
    match host.call(call).await.map_err(EngineError::Host)? {
        Scalar::Float(spend) => Ok(spend),
        Scalar::Int(spend) => Ok(spend as f64),
        other => Err(EngineError::Host(HostCallError::Reply {
            call: CurrentSpend::NAME,
            message: format!("{other:?} is not a spend"),
        })),
    }
}

async fn can_apply(
    store: &SpendCounters,
    counter: &SpendCounter,
    adjustment: f64,
) -> Result<bool, EngineError> {
    Ok(match store.read(counter).await?.value() {
        None => false,
        Some(value) => !(adjustment < 0.0 && value.as_f64() + adjustment < -EPSILON),
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Inconsistent {
    Drop,
    Refuse,
    Report,
}

struct Adjustment {
    index: usize,
    counter: SpendCounter,
    counter_key: String,
    target: f64,
    adjustment: f64,
}

async fn apply(
    store: &SpendCounters,
    adjustments: &[Adjustment],
    inconsistent: Inconsistent,
) -> Result<Vec<bool>, EngineError> {
    let mut consistent = Vec::with_capacity(adjustments.len());
    for adjustment in adjustments {
        consistent.push(can_apply(store, &adjustment.counter, adjustment.adjustment).await?);
    }
    if inconsistent == Inconsistent::Refuse
        && let Some((adjustment, _)) = adjustments
            .iter()
            .zip(&consistent)
            .find(|(_, consistent)| !**consistent)
    {
        return Err(EngineError::Inconsistent(format!(
            "Cannot resize budget reservation against inconsistent counter {}",
            adjustment.counter_key
        )));
    }
    let applicable: Vec<(SpendCounter, Number)> = adjustments
        .iter()
        .zip(&consistent)
        .filter(|(_, consistent)| **consistent)
        .map(|(adjustment, _)| {
            (
                adjustment.counter.clone(),
                Number::Float(adjustment.adjustment),
            )
        })
        .collect();
    if !applicable.is_empty()
        && let Err(error) = store.increment_many(&applicable).await
    {
        let counters: Vec<SpendCounter> = applicable.into_iter().map(|(c, _)| c).collect();
        store.invalidate(&counters).await;
        return Err(error.into());
    }
    if inconsistent == Inconsistent::Drop {
        for (adjustment, _) in adjustments
            .iter()
            .zip(&consistent)
            .filter(|(_, consistent)| !**consistent)
        {
            store
                .invalidate(std::slice::from_ref(&adjustment.counter))
                .await;
        }
    }
    Ok(consistent)
}

pub(crate) async fn apply_actual(
    store: &SpendCounters,
    entries: &mut [Entry],
    actual: f64,
    inconsistent: Inconsistent,
) -> Result<(), EngineError> {
    let adjustments: Vec<Adjustment> = entries
        .iter()
        .enumerate()
        .filter_map(|(index, entry)| {
            let target = actual - entry.reserved_cost;
            let adjustment = target - entry.applied_adjustment;
            (adjustment != 0.0).then(|| Adjustment {
                index,
                counter: entry.counter.clone(),
                counter_key: entry.counter_key.clone(),
                target,
                adjustment,
            })
        })
        .collect();
    apply(store, &adjustments, inconsistent).await?;
    for adjustment in adjustments {
        entries[adjustment.index].applied_adjustment = adjustment.target;
    }
    Ok(())
}

pub(crate) async fn release_best_effort(store: &SpendCounters, entries: &mut [Entry]) {
    for entry in entries {
        if apply_actual(store, std::slice::from_mut(entry), 0.0, Inconsistent::Drop)
            .await
            .is_err()
        {
            tracing::error!(
                counter = entry.counter_key,
                "Failed to release partial budget reservation during exception cleanup"
            );
            store.invalidate(std::slice::from_ref(&entry.counter)).await;
        }
    }
}

struct Admission<'a> {
    store: &'a SpendCounters,
    host: &'a Host,
    plan: &'a Plan,
    applied: Vec<Entry>,
    cost: f64,
}

impl Admission<'_> {
    async fn charge(&mut self, counters: &[Counter]) -> Result<(), EngineError> {
        let mut values = Vec::with_capacity(counters.len());
        for counter in counters {
            values.push(current_value(self.store, counter, self.host).await?);
        }
        let fits = counters
            .iter()
            .zip(&values)
            .all(|(counter, value)| value + self.cost <= counter.planned.max_budget);
        let groups: Vec<&[Counter]> = if fits {
            vec![counters]
        } else {
            counters.chunks(1).collect()
        };
        for group in groups {
            self.charge_group(group).await?;
        }
        Ok(())
    }

    async fn charge_group(&mut self, group: &[Counter]) -> Result<(), EngineError> {
        let charged = self.cost;
        let entries: Vec<Entry> = group.iter().map(|c| Entry::new(c, charged)).collect();
        let ids: Vec<usize> = entries.iter().map(|entry| entry.id).collect();
        self.applied.extend(entries);
        let increments: Vec<(SpendCounter, Number)> = group
            .iter()
            .map(|counter| (counter.counter.clone(), Number::Float(charged)))
            .collect();
        let totals = match self.store.increment_many(&increments).await {
            Ok(totals) => totals,
            Err(error) => {
                tracing::warn!(%error, "Skipping budget reservation because spend counter reservation failed");
                let counters: Vec<SpendCounter> = increments.into_iter().map(|(c, _)| c).collect();
                self.store.invalidate(&counters).await;
                self.applied.retain(|entry| !ids.contains(&entry.id));
                if self.plan.fail_closed {
                    return Err(EngineError::Unavailable);
                }
                return Ok(());
            }
        };
        for ((counter, id), total) in group.iter().zip(ids).zip(totals) {
            if !self.applied.iter().any(|entry| entry.id == id) {
                continue;
            }
            let current = total.as_f64() - (charged - self.cost);
            if current > counter.planned.max_budget {
                self.over_budget(counter, id, current).await?;
            }
        }
        Ok(())
    }

    async fn over_budget(
        &mut self,
        counter: &Counter,
        id: usize,
        current: f64,
    ) -> Result<(), EngineError> {
        if self.plan.throttle_counter.as_deref() == Some(counter.key()) {
            if let Some(position) = self.applied.iter().position(|entry| entry.id == id) {
                let mut entry = self.applied.remove(position);
                release_best_effort(self.store, std::slice::from_mut(&mut entry)).await;
            }
            return Ok(());
        }
        let remaining = counter.planned.max_budget - (current - self.cost);
        if remaining <= EPSILON {
            return Err(exceeded(counter, current, None));
        }
        if self.plan.fail_closed && current - counter.planned.max_budget > EPSILON {
            return Err(exceeded(counter, current - self.cost, Some(self.cost)));
        }
        apply_actual(
            self.store,
            &mut self.applied,
            remaining,
            Inconsistent::Refuse,
        )
        .await?;
        for entry in &mut self.applied {
            entry.reserved_cost = remaining;
            entry.applied_adjustment = 0.0;
        }
        self.cost = remaining;
        Ok(())
    }
}

pub async fn reserve(
    store: &SpendCounters,
    plan: &Plan,
    host: &Host,
) -> Result<Option<Reservation>, EngineError> {
    let counters = plan.counters().map_err(EngineError::Plan)?;
    let mut admission = Admission {
        store,
        host,
        plan,
        applied: Vec::new(),
        cost: plan.reservation_cost,
    };
    if let Err(error) = admission.charge(&counters).await {
        release_best_effort(store, &mut admission.applied).await;
        return Err(error);
    }
    if admission.applied.is_empty() {
        return Ok(None);
    }
    Ok(Some(Reservation {
        reserved_cost: admission.cost,
        input_cost: plan.input_cost.min(admission.cost),
        input_tokens: plan.input_tokens,
        entries: admission.applied,
    }))
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Reseed {
    pub counter_key: String,
    pub actual_cost: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Settlement {
    pub adjustments: Vec<Option<f64>>,
    pub reseed: Vec<Reseed>,
}

fn rejected(rejected: Option<&[String]>, member: &str) -> bool {
    rejected.is_some_and(|names| names.iter().any(|name| name == member))
}

fn held_adjustment(
    index: usize,
    entry: &ReservationEntry,
    window_reset: Option<&str>,
    actual: f64,
    default_reserved_cost: f64,
    now: &Moment,
) -> Result<Option<(Adjustment, f64)>, EngineError> {
    let names = entry.rejected.as_deref();
    if rejected(names, "counter_key") {
        return Err(ReservationError::type_error(
            "str() of a reservation entry counter_key raises",
        )
        .into());
    }
    let Some(counter_key) = &entry.counter_key else {
        return Ok(None);
    };
    let settled_cost = match window_reset.map(window::parse_reset).transpose()?.flatten() {
        Some(reset_at) if reset_at <= *now => 0.0,
        _ => actual,
    };
    let reserved_cost = match entry.reserved_cost {
        Some(cost) if !rejected(names, "reserved_cost") => cost,
        _ => default_reserved_cost,
    };
    if rejected(names, "applied_adjustment") {
        return Err(ReservationError::value_error(
            "could not convert the reservation entry applied_adjustment to float",
        )
        .into());
    }
    let target = settled_cost - reserved_cost;
    let adjustment = target - entry.applied_adjustment.unwrap_or(0.0);
    if adjustment == 0.0 {
        return Ok(None);
    }
    Ok(Some((
        Adjustment {
            index,
            counter: SpendCounter::parse(counter_key).map_err(EngineError::Plan)?,
            counter_key: counter_key.clone(),
            target,
            adjustment,
        },
        settled_cost,
    )))
}

pub async fn settle(
    store: &SpendCounters,
    reservation: &BudgetReservation,
    window_resets: &[Option<String>],
    actual: f64,
    now: &Moment,
) -> Result<Settlement, EngineError> {
    let entries = reservation.entries.as_deref().unwrap_or_default();
    if entries.iter().any(Option::is_none) {
        return Err(
            ReservationError::type_error("a budget reservation entry is not a mapping").into(),
        );
    }
    let default_reserved_cost = reservation.reserved_cost.unwrap_or(0.0);
    let mut adjustments = Vec::new();
    let mut settled_costs = Vec::new();
    for (index, entry) in entries.iter().enumerate() {
        let Some(entry) = entry else { continue };
        let window_reset = window_resets.get(index).and_then(Option::as_deref);
        if let Some((adjustment, settled_cost)) = held_adjustment(
            index,
            entry,
            window_reset,
            actual,
            default_reserved_cost,
            now,
        )? {
            adjustments.push(adjustment);
            settled_costs.push(settled_cost);
        }
    }
    let consistent = apply(store, &adjustments, Inconsistent::Report).await?;
    let mut settlement = Settlement {
        adjustments: vec![None; entries.len()],
        reseed: Vec::new(),
    };
    for ((adjustment, settled_cost), consistent) in
        adjustments.iter().zip(settled_costs).zip(consistent)
    {
        if !consistent {
            settlement.reseed.push(Reseed {
                counter_key: adjustment.counter_key.clone(),
                actual_cost: settled_cost,
            });
        }
        settlement.adjustments[adjustment.index] = Some(adjustment.target);
    }
    Ok(settlement)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use litellm_redis_store::{
        Number,
        spend_counters::{SpendCounter, SpendCounters},
    };
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::{
        plan::{Plan, PlannedCounter},
        window::ErrorKind,
    };

    fn counter(key: &str, max_budget: f64, entity_type: &str, entity_id: &str) -> PlannedCounter {
        PlannedCounter {
            counter_key: key.to_owned(),
            max_budget,
            fallback_spend: 0.0,
            entity_type: entity_type.to_owned(),
            entity_id: entity_id.to_owned(),
            spend_log_entity_id: None,
        }
    }

    fn plan(counters: Vec<PlannedCounter>, cost: f64) -> Plan {
        Plan {
            counters,
            reservation_cost: cost,
            input_cost: cost / 4.0,
            input_tokens: Some(12),
            throttle_counter: None,
            fail_closed: false,
            status_code: 400,
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
    const TEAM: &str = "spend:team:t1";

    #[rstest]
    #[tokio::test]
    async fn counters_that_fit_are_charged_the_whole_estimate() {
        let store = store(&[(KEY, 1.0), (TEAM, 2.0)]);
        let plan = plan(
            vec![
                counter(KEY, 10.0, "Key", "hashed"),
                counter(TEAM, 10.0, "Team", "t1"),
            ],
            3.0,
        );
        let reservation = reserve(&store, &plan, &Host::detached())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(reservation.reserved_cost, 3.0);
        assert_eq!(reservation.input_cost, 0.75);
        assert_eq!(value(&store, KEY), Some(4.0));
        assert_eq!(value(&store, TEAM), Some(5.0));
        assert_eq!(
            serde_json::to_value(&reservation).unwrap()["entries"][1],
            json!({"counter_key": TEAM, "entity_type": "Team", "entity_id": "t1", "reserved_cost": 3.0, "applied_adjustment": 0.0})
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_partial_fit_shrinks_every_applied_entry_to_what_is_left() {
        let store = store(&[(KEY, 1.0), (TEAM, 9.0)]);
        let plan = plan(
            vec![
                counter(KEY, 10.0, "Key", "hashed"),
                counter(TEAM, 10.0, "Team", "t1"),
            ],
            3.0,
        );
        let reservation = reserve(&store, &plan, &Host::detached())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(reservation.reserved_cost, 1.0);
        assert_eq!(value(&store, KEY), Some(2.0));
        assert_eq!(value(&store, TEAM), Some(10.0));
    }

    #[rstest]
    #[case::spent(
        10.0,
        false,
        "Budget has been exceeded! Team=t1 Current cost: 11.0, Max budget: 10.0",
        11.0
    )]
    #[case::fail_closed(
        9.5,
        true,
        "Budget has been exceeded! Team=t1 Current cost: 9.5, Estimated request cost: 1.0, Max budget: 10.0",
        9.5
    )]
    #[rstest]
    #[tokio::test]
    async fn an_over_budget_counter_refuses_and_releases_what_was_charged(
        #[case] team_spend: f64,
        #[case] fail_closed: bool,
        #[case] message: &str,
        #[case] current_cost: f64,
    ) {
        let store = store(&[(KEY, 1.0), (TEAM, team_spend)]);
        let plan = Plan {
            fail_closed,
            ..plan(
                vec![
                    counter(KEY, 10.0, "Key", "hashed"),
                    counter(TEAM, 10.0, "Team", "t1"),
                ],
                1.0,
            )
        };
        let Err(EngineError::Exceeded(exceeded)) = reserve(&store, &plan, &Host::detached()).await
        else {
            panic!("expected a refusal");
        };
        assert_eq!(exceeded.message, message);
        assert_eq!(exceeded.current_cost, current_cost);
        assert_eq!(exceeded.entity_type, Some("team"));
        assert_eq!(value(&store, KEY), Some(1.0));
        assert_eq!(value(&store, TEAM), Some(team_spend));
    }

    #[rstest]
    #[tokio::test]
    async fn a_throttled_key_releases_its_own_entry_and_keeps_the_rest() {
        let store = store(&[(KEY, 10.0), (TEAM, 0.0)]);
        let plan = Plan {
            throttle_counter: Some(KEY.to_owned()),
            ..plan(
                vec![
                    counter(KEY, 10.0, "Key", "hashed"),
                    counter(TEAM, 10.0, "Team", "t1"),
                ],
                1.0,
            )
        };
        let reservation = reserve(&store, &plan, &Host::detached())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(reservation.entries.len(), 1);
        assert_eq!(reservation.entries[0].counter_key, TEAM);
        assert_eq!(value(&store, KEY), Some(10.0));
        assert_eq!(value(&store, TEAM), Some(1.0));
    }

    fn record(reservation: &Reservation) -> BudgetReservation {
        serde_json::from_value(serde_json::to_value(reservation).unwrap()).unwrap()
    }

    fn stamp(record: &mut BudgetReservation, settlement: &Settlement) {
        for (entry, target) in record
            .entries
            .iter_mut()
            .flatten()
            .zip(&settlement.adjustments)
        {
            if let (Some(entry), Some(target)) = (entry, target) {
                entry.applied_adjustment = Some(*target);
            }
        }
    }

    async fn settle_now(
        store: &SpendCounters,
        record: &BudgetReservation,
        resets: &[Option<String>],
        actual: f64,
    ) -> Result<Settlement, EngineError> {
        settle(
            store,
            record,
            resets,
            actual,
            &Moment::from_unix_micros(NOW),
        )
        .await
    }

    const NOW: i128 = 1_763_035_200_000_000;

    #[rstest]
    #[case::success(0.4, 1.4)]
    #[case::refund(0.0, 1.0)]
    #[tokio::test]
    async fn settling_moves_each_counter_from_the_hold_to_the_actual_cost(
        #[case] actual: f64,
        #[case] after: f64,
    ) {
        let store = store(&[(KEY, 1.0)]);
        let plan = plan(vec![counter(KEY, 10.0, "Key", "hashed")], 2.0);
        let mut record = record(
            &reserve(&store, &plan, &Host::detached())
                .await
                .unwrap()
                .unwrap(),
        );
        let settlement = settle_now(&store, &record, &[], actual).await.unwrap();
        assert_eq!(settlement.adjustments, vec![Some(actual - 2.0)]);
        stamp(&mut record, &settlement);
        let again = settle_now(&store, &record, &[], actual).await.unwrap();
        assert_eq!(again.adjustments, vec![None]);
        assert_eq!(value(&store, KEY), Some(after));
    }

    #[rstest]
    #[case::missing(None, 0.4, None)]
    #[case::would_go_negative(Some(0.5), 0.0, Some(0.5))]
    #[tokio::test]
    async fn settling_an_inconsistent_counter_reports_it_and_leaves_it_alone(
        #[case] before_settle: Option<f64>,
        #[case] actual: f64,
        #[case] after: Option<f64>,
    ) {
        let store = store(&[(KEY, 1.0), (TEAM, 0.0)]);
        let plan = plan(
            vec![
                counter(KEY, 10.0, "Key", "hashed"),
                counter(TEAM, 10.0, "Team", "t1"),
            ],
            2.0,
        );
        let record = record(
            &reserve(&store, &plan, &Host::detached())
                .await
                .unwrap()
                .unwrap(),
        );
        let key = SpendCounter::parse(KEY).unwrap();
        match before_settle {
            Some(spend) => store.set_local(&key, Number::Float(spend), None).unwrap(),
            None => store.invalidate(std::slice::from_ref(&key)).await,
        }
        let settlement = settle_now(&store, &record, &[], actual).await.unwrap();
        assert_eq!(
            settlement.reseed,
            vec![Reseed {
                counter_key: KEY.to_owned(),
                actual_cost: actual
            }]
        );
        assert_eq!(value(&store, KEY), after);
        assert_eq!(value(&store, TEAM), Some(2.0 + (actual - 2.0)));
        assert_eq!(settlement.adjustments, vec![Some(actual - 2.0); 2]);
    }

    #[rstest]
    #[case::reset_before_settlement("2025-11-13T11:59:59+00:00", 0.0)]
    #[case::reset_at_settlement("2025-11-13T12:00:00Z", 0.0)]
    #[case::still_open("2025-11-13T12:00:00.000001+00:00", 0.4)]
    #[case::unparsable("next month", 0.4)]
    #[tokio::test]
    async fn an_entry_whose_window_reset_before_settlement_settles_at_no_cost(
        #[case] reset: &str,
        #[case] settled: f64,
    ) {
        let store = store(&[(KEY, 1.0), (TEAM, 0.0)]);
        let plan = plan(
            vec![
                counter(KEY, 10.0, "Key", "hashed"),
                counter(TEAM, 10.0, "Team", "t1"),
            ],
            2.0,
        );
        let record = record(
            &reserve(&store, &plan, &Host::detached())
                .await
                .unwrap()
                .unwrap(),
        );
        store
            .invalidate(&[SpendCounter::parse(TEAM).unwrap()])
            .await;
        let settlement = settle_now(&store, &record, &[Some(reset.to_owned()), None], 0.4)
            .await
            .unwrap();
        assert_eq!(value(&store, KEY), Some(1.0 + settled));
        assert_eq!(
            settlement.adjustments,
            vec![Some(settled - 2.0), Some(0.4 - 2.0)]
        );
        assert_eq!(
            settlement.reseed,
            vec![Reseed {
                counter_key: TEAM.to_owned(),
                actual_cost: 0.4
            }]
        );
    }

    fn held(entries: serde_json::Value) -> BudgetReservation {
        serde_json::from_value(json!({"reserved_cost": 2.0, "entries": entries})).unwrap()
    }

    #[rstest]
    #[case::not_a_mapping(json!([{"counter_key": KEY, "reserved_cost": 2.0}, null]), None, ErrorKind::Type)]
    #[case::naive_reset(json!([{"counter_key": KEY, "reserved_cost": 2.0}]), Some("2025-11-13T12:00:00"), ErrorKind::Type)]
    #[case::applied_adjustment_float_refuses(json!([{"counter_key": KEY, "rejected": ["applied_adjustment"]}]), None, ErrorKind::Value)]
    #[tokio::test]
    async fn an_entry_python_raises_on_fails_the_settlement_before_any_counter_moves(
        #[case] entries: serde_json::Value,
        #[case] reset: Option<&str>,
        #[case] kind: ErrorKind,
    ) {
        let store = store(&[(KEY, 3.0)]);
        let Err(EngineError::Reservation(error)) =
            settle_now(&store, &held(entries), &[reset.map(str::to_owned)], 0.5).await
        else {
            panic!("expected a reservation error");
        };
        assert_eq!(error.kind, kind);
        assert_eq!(value(&store, KEY), Some(3.0));
    }

    #[rstest]
    #[case::no_counter_key(json!([{"reserved_cost": 2.0}]), 3.0, vec![None])]
    #[case::reserved_cost_float_refuses(json!([{"counter_key": KEY, "rejected": ["reserved_cost"]}]), 1.5, vec![Some(-1.5)])]
    #[case::reserved_cost_missing(json!([{"counter_key": KEY}]), 1.5, vec![Some(-1.5)])]
    #[case::own_reserved_cost(json!([{"counter_key": KEY, "reserved_cost": 1.0, "applied_adjustment": 0.25}]), 2.25, vec![Some(-0.5)])]
    #[tokio::test]
    async fn each_entry_member_is_read_as_budgets_reads_it(
        #[case] entries: serde_json::Value,
        #[case] after: f64,
        #[case] adjustments: Vec<Option<f64>>,
    ) {
        let store = store(&[(KEY, 3.0)]);
        let settlement = settle_now(&store, &held(entries), &[], 0.5).await.unwrap();
        assert_eq!(settlement.adjustments, adjustments);
        assert_eq!(value(&store, KEY), Some(after));
    }
}
