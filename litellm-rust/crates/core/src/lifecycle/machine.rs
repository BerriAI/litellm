use std::marker::PhantomData;

use super::ActionBinding;

pub trait LifecycleRoute: Sized {
    type Admission;
    type Options;
    type Context;
    type Operation: Copy;
    type Observation;
    type Outcome;
    type Transition;
    type Error;
    type Decline;
    type State;

    fn program(state: &Self::State) -> &super::CallLifecycle;
    fn program_mut(state: &mut Self::State) -> &mut super::CallLifecycle;

    fn admit(
        admission: &Self::Admission,
        options: Self::Options,
    ) -> Result<Result<Self::State, Self::Decline>, Self::Error>;

    fn operation(state: &Self::State) -> Self::Operation;

    fn advance(
        state: &mut Self::State,
        outcome: Self::Outcome,
        observations: Self::Observation,
    ) -> Result<Self::Transition, Self::Error>;

    fn actions_for(operation: Self::Operation, context: &Self::Context)
    -> &'static [ActionBinding];
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
    ) -> Result<Result<Self, Route::Decline>, Route::Error> {
        Route::admit(admission, options).map(|admission| {
            admission.map(|state| Self {
                state,
                route: PhantomData,
            })
        })
    }

    pub fn operation(&self) -> Route::Operation {
        Route::operation(&self.state)
    }

    pub(crate) fn advance(
        &mut self,
        outcome: Route::Outcome,
        observations: Route::Observation,
    ) -> Result<Route::Transition, Route::Error> {
        Route::advance(&mut self.state, outcome, observations)
    }

    pub fn issue(&mut self) -> Result<super::program::OperationTicket, crate::Error> {
        Route::program_mut(&mut self.state).issue()
    }

    pub fn complete_operation(&mut self, ticket: super::program::OperationTicket, outcome: super::Outcome, observations: super::program::Observations) -> Result<super::program::Transition, crate::Error> {
        Route::program_mut(&mut self.state).complete_operation(ticket, outcome, observations)
    }

    pub fn provider_permit(&mut self, ticket: &super::program::OperationTicket) -> Result<super::program::ProviderPermit, crate::Error> {
        Route::program_mut(&mut self.state).provider_permit(ticket)
    }

    pub fn actions_for(&self, context: &Route::Context) -> &'static [ActionBinding] {
        Route::actions_for(self.operation(), context)
    }
}
