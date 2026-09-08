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

    pub fn advance(
        &mut self,
        outcome: Route::Outcome,
        observations: Route::Observation,
    ) -> Result<Route::Transition, Route::Error> {
        Route::advance(&mut self.state, outcome, observations)
    }

    pub fn actions_for(&self, context: &Route::Context) -> &'static [ActionBinding] {
        Route::actions_for(self.operation(), context)
    }
}
