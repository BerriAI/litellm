use std::marker::PhantomData;

use super::program::{Observations, Operation, OperationTicket, ProviderPermit, Transition};
use super::{CallLifecycle, Outcome};
use crate::Error;

pub(crate) mod sealed {
    pub trait Sealed {}
}

pub trait LifecycleRoute: Sized + sealed::Sealed {
    type Admission;
    type Options;
    type Decline;
    type State;

    fn program(state: &Self::State) -> &CallLifecycle;
    fn program_mut(state: &mut Self::State) -> &mut CallLifecycle;
    fn admit(
        admission: &Self::Admission,
        options: Self::Options,
    ) -> Result<Result<Self::State, Self::Decline>, Error>;
}

#[derive(Debug)]
pub struct Lifecycle<Route: LifecycleRoute> {
    pub(crate) state: Route::State,
    route: PhantomData<Route>,
}

impl<Route: LifecycleRoute> Lifecycle<Route> {
    pub fn admit(
        admission: &Route::Admission,
        options: Route::Options,
    ) -> Result<Result<Self, Route::Decline>, Error> {
        Route::admit(admission, options).map(|admission| {
            admission.map(|state| Self {
                state,
                route: PhantomData,
            })
        })
    }

    pub fn operation(&self) -> Operation {
        Route::program(&self.state).operation()
    }

    pub fn issue(&mut self) -> Result<OperationTicket, Error> {
        Route::program_mut(&mut self.state).issue()
    }

    pub fn complete_operation(
        &mut self,
        ticket: OperationTicket,
        outcome: Outcome,
        observations: Observations,
    ) -> Result<Transition, Error> {
        Route::program_mut(&mut self.state).complete_operation(ticket, outcome, observations)
    }

    pub fn preparation_permit(
        &mut self,
        ticket: &OperationTicket,
    ) -> Result<super::program::PreparationPermit<Route>, Error> {
        Route::program_mut(&mut self.state).preparation_permit(ticket)
    }

    pub fn provider_permit(
        &mut self,
        ticket: &OperationTicket,
    ) -> Result<ProviderPermit<Route>, Error> {
        Route::program_mut(&mut self.state).provider_permit(ticket)
    }

    #[cfg(test)]
    pub(crate) fn advance(
        &mut self,
        outcome: Outcome,
        observations: Observations,
    ) -> Result<Transition, Error> {
        let ticket = self.issue()?;
        self.complete_operation(ticket, outcome, observations)
    }
}
