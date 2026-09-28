use std::future::Future;

use crate::{Charges, Cost, Error, ProviderWork, ReportedUsage};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BudgetAdmission<R> {
    budget: Option<R>,
}

impl<R> BudgetAdmission<R> {
    pub fn new(budget: Option<R>) -> Self {
        Self { budget }
    }

    pub fn budget(&self) -> Option<&R> {
        self.budget.as_ref()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Outcome {
    Succeeded,
    Failed,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Terminal<U> {
    outcome: Outcome,
    work: ProviderWork,
    usage: ReportedUsage<U>,
    charges: Charges,
}

impl<U> Terminal<U> {
    pub fn new(
        outcome: Outcome,
        work: ProviderWork,
        usage: ReportedUsage<U>,
        charges: Charges,
    ) -> Result<Self, Error> {
        let charges = charges.for_work(work);
        charges.total()?;
        Ok(Self {
            outcome,
            work,
            usage,
            charges,
        })
    }

    pub fn outcome(&self) -> Outcome {
        self.outcome
    }

    pub fn work(&self) -> ProviderWork {
        self.work
    }

    pub fn usage(&self) -> &ReportedUsage<U> {
        &self.usage
    }

    pub fn charges(&self) -> Charges {
        self.charges
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Effect {
    ReconcileBudget,
    RecordSpend,
    ReleaseBudgetReservation,
}

impl Effect {
    const ORDER: [Self; 3] = [
        Self::ReconcileBudget,
        Self::RecordSpend,
        Self::ReleaseBudgetReservation,
    ];

    fn index(self) -> usize {
        match self {
            Self::ReconcileBudget => 0,
            Self::RecordSpend => 1,
            Self::ReleaseBudgetReservation => 2,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ApplyResult<E> {
    Accepted,
    Committed,
    NotApplied(E),
    Indeterminate(E),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EffectState<E> {
    NotRequired,
    Pending,
    InFlight,
    Accepted,
    Committed,
    NotApplied(E),
    Indeterminate(E),
}

impl<E> From<ApplyResult<E>> for EffectState<E> {
    fn from(result: ApplyResult<E>) -> Self {
        match result {
            ApplyResult::Accepted => Self::Accepted,
            ApplyResult::Committed => Self::Committed,
            ApplyResult::NotApplied(error) => Self::NotApplied(error),
            ApplyResult::Indeterminate(error) => Self::Indeterminate(error),
        }
    }
}

pub struct Progress<E> {
    effects: [EffectState<E>; 3],
}

impl<E> Progress<E> {
    pub fn effect(&self, effect: Effect) -> &EffectState<E> {
        &self.effects[effect.index()]
    }
}

pub struct Settlement<'a, R, U, E> {
    pub admission: &'a BudgetAdmission<R>,
    pub terminal: &'a Terminal<U>,
    pub progress: &'a Progress<E>,
}

pub trait Backend<R, U> {
    type Error;

    fn apply(
        &mut self,
        effect: Effect,
        settlement: Settlement<'_, R, U, Self::Error>,
    ) -> impl Future<Output = ApplyResult<Self::Error>>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SettlementStatus {
    Open,
    Pending,
    NeedsAttention,
    Unpriced,
    Accepted,
    Committed,
}

pub struct Session<R, U, E> {
    admission: BudgetAdmission<R>,
    terminal: Option<Terminal<U>>,
    progress: Progress<E>,
}

impl<R, U, E> Session<R, U, E> {
    pub fn new(admission: BudgetAdmission<R>) -> Self {
        let effects = Effect::ORDER.map(|effect| {
            let required = match effect {
                Effect::RecordSpend => true,
                Effect::ReconcileBudget => admission.budget.is_some(),
                Effect::ReleaseBudgetReservation => admission.budget.is_some(),
            };
            if required {
                EffectState::Pending
            } else {
                EffectState::NotRequired
            }
        });
        Self {
            admission,
            terminal: None,
            progress: Progress { effects },
        }
    }

    pub fn finish(&mut self, terminal: Terminal<U>) -> Result<(), Error> {
        if self.terminal.is_some() {
            return Err(Error::AlreadyTerminal);
        }
        self.terminal = Some(terminal);
        Ok(())
    }

    pub fn terminal(&self) -> Option<&Terminal<U>> {
        self.terminal.as_ref()
    }

    pub fn progress(&self) -> &Progress<E> {
        &self.progress
    }

    pub fn status(&self) -> SettlementStatus {
        let Some(terminal) = &self.terminal else {
            return SettlementStatus::Open;
        };
        if self.progress.effects.iter().any(|state| {
            matches!(
                state,
                EffectState::InFlight | EffectState::NotApplied(_) | EffectState::Indeterminate(_)
            )
        }) {
            return SettlementStatus::NeedsAttention;
        }
        if self
            .progress
            .effects
            .iter()
            .any(|state| matches!(state, EffectState::Pending))
        {
            return SettlementStatus::Pending;
        }
        if terminal.charges.total() == Ok(Cost::Unknown) {
            return SettlementStatus::Unpriced;
        }
        if self
            .progress
            .effects
            .iter()
            .any(|state| matches!(state, EffectState::Accepted))
        {
            return SettlementStatus::Accepted;
        }
        SettlementStatus::Committed
    }

    pub fn retry_not_applied(&mut self, effect: Effect) -> Result<(), Error> {
        if !matches!(self.progress.effect(effect), EffectState::NotApplied(_)) {
            return Err(Error::UnsafeRetry { effect });
        }
        self.progress.effects[effect.index()] = EffectState::Pending;
        Ok(())
    }

    pub async fn settle<B: Backend<R, U, Error = E>>(
        &mut self,
        backend: &mut B,
    ) -> Result<SettlementStatus, Error> {
        let terminal = self.terminal.as_ref().ok_or(Error::NotTerminal)?;
        for effect in Effect::ORDER {
            if !matches!(self.progress.effects[effect.index()], EffectState::Pending) {
                continue;
            }
            self.progress.effects[effect.index()] = EffectState::InFlight;
            let result = backend
                .apply(
                    effect,
                    Settlement {
                        admission: &self.admission,
                        terminal,
                        progress: &self.progress,
                    },
                )
                .await;
            self.progress.effects[effect.index()] = result.into();
        }
        Ok(self.status())
    }
}
