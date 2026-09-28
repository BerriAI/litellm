use std::{future::Future, task::Context};

use litellm_accounting::{
    ApplyResult, Backend, BudgetAdmission, Charges, Cost, Effect, EffectState, Error, Outcome,
    ProviderWork, ReportedUsage, Session, Settlement, SettlementStatus, Terminal, Usd,
};
use rstest::{fixture, rstest};
use rusty_money::{Money, iso};

fn usd(value: &str) -> Usd {
    Money::from_str(value, iso::USD).unwrap()
}

const EFFECTS: [Effect; 3] = [
    Effect::ReconcileBudget,
    Effect::RecordSpend,
    Effect::ReleaseBudgetReservation,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Failure {
    StoreUnavailable,
    LostAcknowledgement,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Usage {
    input: u64,
    output: u64,
}

struct Store {
    calls: Vec<Effect>,
    result: Option<(Effect, ApplyResult<Failure>)>,
    blocked: Option<Effect>,
    recorded: Vec<Charges>,
    observations: Vec<(Outcome, ReportedUsage<Usage>)>,
    admissions: Vec<Option<String>>,
    release_progress: Vec<[EffectState<Failure>; 2]>,
}

impl Backend<String, Usage> for Store {
    type Error = Failure;

    async fn apply(
        &mut self,
        effect: Effect,
        settlement: Settlement<'_, String, Usage, Failure>,
    ) -> ApplyResult<Failure> {
        self.calls.push(effect);
        self.admissions.push(settlement.admission.budget().cloned());
        self.observations.push((
            settlement.terminal.outcome(),
            settlement.terminal.usage().clone(),
        ));
        if effect == Effect::ReleaseBudgetReservation {
            self.release_progress.push([
                settlement.progress.effect(Effect::ReconcileBudget).clone(),
                settlement.progress.effect(Effect::RecordSpend).clone(),
            ]);
        }
        if self.blocked == Some(effect) {
            if effect == Effect::RecordSpend {
                self.recorded.push(settlement.terminal.charges());
            }
            return std::future::pending().await;
        }
        if let Some((target, result)) = &self.result
            && *target == effect
        {
            return result.clone();
        }
        if effect == Effect::RecordSpend {
            self.recorded.push(settlement.terminal.charges());
        }
        ApplyResult::Committed
    }
}

#[fixture]
fn store() -> Store {
    Store {
        calls: Vec::new(),
        result: None,
        blocked: None,
        recorded: Vec::new(),
        observations: Vec::new(),
        admissions: Vec::new(),
        release_progress: Vec::new(),
    }
}

#[fixture]
fn terminal() -> Terminal<Usage> {
    Terminal::new(
        Outcome::Succeeded,
        ProviderWork::Started,
        ReportedUsage::Known(Usage {
            input: 10,
            output: 2,
        }),
        Charges::new(Cost::Known(usd("4")), Cost::Known(usd("1"))).unwrap(),
    )
    .unwrap()
}

#[fixture]
fn session() -> Session<String, Usage, Failure> {
    Session::new(BudgetAdmission::new(Some("budget-receipt".to_string())))
}

#[rstest]
#[tokio::test]
async fn admission_cannot_settle_before_a_terminal_outcome(
    mut session: Session<String, Usage, Failure>,
    mut store: Store,
    terminal: Terminal<Usage>,
) {
    assert_eq!(session.status(), SettlementStatus::Open);
    assert_eq!(session.settle(&mut store).await, Err(Error::NotTerminal));
    assert!(store.calls.is_empty());
    session.finish(terminal).unwrap();
    assert_eq!(session.status(), SettlementStatus::Pending);
    assert!(store.calls.is_empty());
    assert_eq!(
        session.settle(&mut store).await,
        Ok(SettlementStatus::Committed)
    );
    assert_eq!(store.calls, EFFECTS);
    assert_eq!(store.recorded.len(), 1);
    assert_eq!(
        store.release_progress,
        [[const { EffectState::Committed }; 2]]
    );
}

#[rstest]
#[tokio::test]
async fn unreserved_calls_still_record_spend(mut store: Store, terminal: Terminal<Usage>) {
    let mut session = Session::new(BudgetAdmission::new(None));
    session.finish(terminal.clone()).unwrap();
    assert_eq!(
        session.settle(&mut store).await,
        Ok(SettlementStatus::Committed)
    );
    assert_eq!(store.calls, [Effect::RecordSpend]);
    assert_eq!(store.recorded, [terminal.charges()]);
    assert_eq!(
        session.progress().effect(Effect::ReleaseBudgetReservation),
        &EffectState::NotRequired
    );
}

#[rstest]
#[tokio::test]
async fn duplicate_terminal_delivery_cannot_replace_or_repeat_settlement(
    mut session: Session<String, Usage, Failure>,
    mut store: Store,
    terminal: Terminal<Usage>,
) {
    session.finish(terminal.clone()).unwrap();
    session.settle(&mut store).await.unwrap();
    assert_eq!(
        session.finish(terminal.clone()),
        Err(Error::AlreadyTerminal)
    );
    let conflicting = Terminal::new(
        Outcome::Failed,
        ProviderWork::NotStarted,
        ReportedUsage::Unknown,
        Charges::new(Cost::Known(usd("0")), Cost::Known(usd("0"))).unwrap(),
    )
    .unwrap();
    assert_eq!(session.finish(conflicting), Err(Error::AlreadyTerminal));
    assert_eq!(session.terminal(), Some(&terminal));
    assert_eq!(
        session.settle(&mut store).await,
        Ok(SettlementStatus::Committed)
    );
    assert_eq!(store.calls, EFFECTS);
    assert_eq!(store.recorded, [terminal.charges()]);
}

#[rstest]
#[case::budget(Effect::ReconcileBudget)]
#[case::spend(Effect::RecordSpend)]
#[case::release(Effect::ReleaseBudgetReservation)]
#[tokio::test]
async fn a_failed_effect_does_not_skip_cleanup_or_repeat_completed_work(
    mut session: Session<String, Usage, Failure>,
    mut store: Store,
    terminal: Terminal<Usage>,
    #[case] failed: Effect,
) {
    store.result = Some((failed, ApplyResult::NotApplied(Failure::StoreUnavailable)));
    session.finish(terminal).unwrap();
    assert_eq!(
        session.settle(&mut store).await,
        Ok(SettlementStatus::NeedsAttention)
    );
    assert_eq!(store.calls, EFFECTS);
    assert_eq!(
        session.progress().effect(failed),
        &EffectState::NotApplied(Failure::StoreUnavailable)
    );
    session.settle(&mut store).await.unwrap();
    assert_eq!(store.calls, EFFECTS);
    session.retry_not_applied(failed).unwrap();
    store.result = None;
    assert_eq!(
        session.settle(&mut store).await,
        Ok(SettlementStatus::Committed)
    );
    assert_eq!(store.calls, [EFFECTS.as_slice(), &[failed]].concat());
    assert_eq!(store.recorded.len(), 1);
    assert_eq!(
        session.retry_not_applied(failed),
        Err(Error::UnsafeRetry { effect: failed })
    );
}

#[rstest]
#[case::budget(Effect::ReconcileBudget)]
#[case::spend(Effect::RecordSpend)]
#[case::release(Effect::ReleaseBudgetReservation)]
#[tokio::test]
async fn uncertain_remote_results_are_not_retried(
    mut session: Session<String, Usage, Failure>,
    mut store: Store,
    terminal: Terminal<Usage>,
    #[case] uncertain: Effect,
) {
    store.result = Some((
        uncertain,
        ApplyResult::Indeterminate(Failure::LostAcknowledgement),
    ));
    session.finish(terminal).unwrap();
    session.settle(&mut store).await.unwrap();
    assert_eq!(
        session.progress().effect(uncertain),
        &EffectState::Indeterminate(Failure::LostAcknowledgement)
    );
    assert_eq!(
        session.retry_not_applied(uncertain),
        Err(Error::UnsafeRetry { effect: uncertain })
    );
    assert_eq!(
        session.settle(&mut store).await,
        Ok(SettlementStatus::NeedsAttention)
    );
    assert_eq!(store.calls, EFFECTS);
}

#[rstest]
#[case::budget(Effect::ReconcileBudget)]
#[case::spend(Effect::RecordSpend)]
#[case::release(Effect::ReleaseBudgetReservation)]
#[tokio::test]
async fn queue_acceptance_is_distinct_from_committed_settlement(
    mut session: Session<String, Usage, Failure>,
    mut store: Store,
    terminal: Terminal<Usage>,
    #[case] queued: Effect,
) {
    store.result = Some((queued, ApplyResult::Accepted));
    session.finish(terminal).unwrap();
    assert_eq!(
        session.settle(&mut store).await,
        Ok(SettlementStatus::Accepted)
    );
    assert_eq!(session.progress().effect(queued), &EffectState::Accepted);
    assert_eq!(
        session.retry_not_applied(queued),
        Err(Error::UnsafeRetry { effect: queued })
    );
    session.settle(&mut store).await.unwrap();
    assert_eq!(store.calls, EFFECTS);
}

#[rstest]
#[case::budget(Effect::ReconcileBudget)]
#[case::spend(Effect::RecordSpend)]
#[case::release(Effect::ReleaseBudgetReservation)]
#[tokio::test]
async fn dropping_settlement_retains_uncertainty_and_remaining_cleanup(
    mut session: Session<String, Usage, Failure>,
    mut store: Store,
    terminal: Terminal<Usage>,
    #[case] interrupted: Effect,
) {
    store.blocked = Some(interrupted);
    session.finish(terminal).unwrap();
    {
        let mut future = Box::pin(session.settle(&mut store));
        let mut context = Context::from_waker(std::task::Waker::noop());
        assert!(future.as_mut().poll(&mut context).is_pending());
    }
    assert_eq!(
        session.progress().effect(interrupted),
        &EffectState::InFlight
    );
    assert_eq!(session.status(), SettlementStatus::NeedsAttention);
    assert_eq!(
        session.retry_not_applied(interrupted),
        Err(Error::UnsafeRetry {
            effect: interrupted
        })
    );
    assert_eq!(
        session.settle(&mut store).await,
        Ok(SettlementStatus::NeedsAttention)
    );
    assert_eq!(store.calls, EFFECTS);
    if interrupted != Effect::ReleaseBudgetReservation {
        assert_eq!(
            session.progress().effect(Effect::ReleaseBudgetReservation),
            &EffectState::Committed
        );
    }
    assert_eq!(store.recorded.len(), 1);
}

#[rstest]
#[case::failure(Outcome::Failed)]
#[case::cancellation(Outcome::Cancelled)]
#[tokio::test]
async fn interrupted_execution_preserves_partial_usage_and_charges(
    mut session: Session<String, Usage, Failure>,
    mut store: Store,
    #[case] outcome: Outcome,
) {
    let usage = ReportedUsage::Known(Usage {
        input: 10,
        output: 1,
    });
    let charges = Charges::new(Cost::Known(usd("1")), Cost::Known(usd("0"))).unwrap();
    session
        .finish(Terminal::new(outcome, ProviderWork::Started, usage.clone(), charges).unwrap())
        .unwrap();
    assert_eq!(
        session.settle(&mut store).await,
        Ok(SettlementStatus::Committed)
    );
    assert_eq!(store.recorded, [charges]);
    assert_eq!(store.observations, vec![(outcome, usage); EFFECTS.len()]);
    assert_eq!(store.calls.last(), Some(&Effect::ReleaseBudgetReservation));
}

#[rstest]
#[case::unknown(ReportedUsage::Unknown, Cost::Unknown, SettlementStatus::Unpriced)]
#[case::known_zero(
    ReportedUsage::Known(Usage { input: 0, output: 0 }),
    Cost::Known(usd("0")),
    SettlementStatus::Committed
)]
#[case::usage_unknown_but_cost_known(
    ReportedUsage::Unknown,
    Cost::Known(usd("2")),
    SettlementStatus::Committed
)]
#[case::usage_reported_but_price_unknown(
    ReportedUsage::Known(Usage { input: 10, output: 2 }),
    Cost::Unknown,
    SettlementStatus::Unpriced
)]
#[tokio::test]
async fn unknown_usage_and_cost_remain_distinct_from_reported_zero(
    mut session: Session<String, Usage, Failure>,
    mut store: Store,
    #[case] usage: ReportedUsage<Usage>,
    #[case] cost: Cost,
    #[case] expected: SettlementStatus,
) {
    let terminal = Terminal::new(
        Outcome::Failed,
        ProviderWork::Started,
        usage.clone(),
        Charges::new(cost, Cost::Known(usd("0"))).unwrap(),
    )
    .unwrap();
    session.finish(terminal).unwrap();
    assert_eq!(session.settle(&mut store).await, Ok(expected));
    assert_eq!(store.recorded[0].provider(), cost);
    assert_eq!(store.recorded[0].total(), Ok(cost));
    assert_eq!(session.terminal().unwrap().usage(), &usage);
    assert_eq!(
        session.progress().effect(Effect::ReleaseBudgetReservation),
        &EffectState::Committed
    );
}

#[rstest]
#[case::reserved(true)]
#[case::unreserved(false)]
#[tokio::test]
async fn budget_receipt_is_preserved_for_every_effect(
    mut store: Store,
    terminal: Terminal<Usage>,
    #[case] reserved: bool,
) {
    let receipt = reserved.then(|| "budget-receipt".to_string());
    let admission = BudgetAdmission::new(receipt.clone());
    assert_eq!(admission.budget(), receipt.as_ref());
    let mut session = Session::new(admission);
    session.finish(terminal).unwrap();
    assert_eq!(
        session.settle(&mut store).await,
        Ok(SettlementStatus::Committed)
    );
    let expected = if reserved {
        EFFECTS.as_slice()
    } else {
        &[Effect::RecordSpend]
    };
    assert_eq!(store.calls, expected);
    assert_eq!(store.admissions, vec![receipt; expected.len()]);
    assert_eq!(store.recorded.len(), 1);
}

#[rstest]
#[case::not_applied(ApplyResult::NotApplied(Failure::StoreUnavailable))]
#[case::uncertain(ApplyResult::Indeterminate(Failure::LostAcknowledgement))]
#[case::queued(ApplyResult::Accepted)]
#[tokio::test]
async fn cleanup_receives_the_actual_settlement_progress(
    mut session: Session<String, Usage, Failure>,
    mut store: Store,
    terminal: Terminal<Usage>,
    #[case] result: ApplyResult<Failure>,
) {
    store.result = Some((Effect::ReconcileBudget, result.clone()));
    session.finish(terminal).unwrap();
    session.settle(&mut store).await.unwrap();
    assert_eq!(
        store.release_progress,
        [[result.into(), EffectState::Committed]]
    );
}

#[rstest]
#[tokio::test]
async fn retries_require_a_confirmed_not_applied_result(
    mut session: Session<String, Usage, Failure>,
    terminal: Terminal<Usage>,
) {
    assert_eq!(
        session.retry_not_applied(Effect::RecordSpend),
        Err(Error::UnsafeRetry {
            effect: Effect::RecordSpend
        })
    );
    session.finish(terminal).unwrap();
    assert_eq!(
        session.retry_not_applied(Effect::RecordSpend),
        Err(Error::UnsafeRetry {
            effect: Effect::RecordSpend
        })
    );
}

#[rstest]
#[tokio::test]
async fn unknown_charges_cannot_become_successful_settlement_through_queue_acceptance(
    mut session: Session<String, Usage, Failure>,
    mut store: Store,
) {
    store.result = Some((Effect::RecordSpend, ApplyResult::Accepted));
    session
        .finish(
            Terminal::new(
                Outcome::Cancelled,
                ProviderWork::Started,
                ReportedUsage::Unknown,
                Charges::new(Cost::Unknown, Cost::Known(usd("0"))).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(
        session.settle(&mut store).await,
        Ok(SettlementStatus::Unpriced)
    );
    assert_eq!(
        session.progress().effect(Effect::RecordSpend),
        &EffectState::Accepted
    );
    assert_eq!(
        session.progress().effect(Effect::ReleaseBudgetReservation),
        &EffectState::Committed
    );
}
