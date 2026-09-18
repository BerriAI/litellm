use std::marker::PhantomData;
use std::sync::Arc;

use litellm_callbacks::machine::Machine;
use litellm_callbacks::route::{Layered, Route};

use crate::attempt::Attempt;
use crate::pick::DeploymentPicker;
use crate::plan::{DeploymentId, RoutePlan};
use crate::report::CallReport;
use crate::signals::{Candidate, Signals};

/// What the router asks its host: only what cannot be answered in process. A plan that
/// needs a model call or a database, a pick that reads shared usage from Redis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RoutingOp {
    ResolvePlan,
    /// Choose among the group's remaining deployments. `None` skips to the next group.
    /// The host applies its own signals; `load` here is what the router knows locally.
    Pick {
        group: u32,
        candidates: Vec<Candidate>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RoutingResult {
    Plan(RoutePlan),
    Picked(Option<DeploymentId>),
}

#[derive(Clone, Debug)]
pub enum PlanSource {
    Plan(RoutePlan),
    /// The router's first op is [`RoutingOp::ResolvePlan`].
    Host,
}

impl From<RoutePlan> for PlanSource {
    fn from(plan: RoutePlan) -> Self {
        Self::Plan(plan)
    }
}

#[derive(Clone)]
pub enum Picker {
    Local {
        picker: Arc<dyn DeploymentPicker>,
        signals: Arc<dyn Signals>,
    },
    /// Every pick is a [`RoutingOp::Pick`] the host answers.
    Host,
}

impl Picker {
    pub fn local(picker: impl DeploymentPicker + 'static, signals: impl Signals + 'static) -> Self {
        Self::Local {
            picker: Arc::new(picker),
            signals: Arc::new(signals),
        }
    }
}

/// The router's own half of the route it presents: its ops, and the report it completes
/// with in place of the attempt's response.
pub struct Routing<A>(PhantomData<fn() -> A>);

impl<A> Route for Routing<A>
where
    A: Attempt + 'static,
{
    type Response = CallReport<A::Complete, <A::Route as Route>::Error>;
    type Error = <A::Route as Route>::Error;
    type Op = RoutingOp;
    type OpResult = RoutingResult;
}

/// The route a [`crate::Router`] over attempts `A` presents to its host.
pub type Routed<A> = Layered<Routing<A>, <A as Machine>::Route>;
