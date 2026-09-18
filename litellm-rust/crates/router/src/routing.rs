use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;

use litellm_callbacks::failure::FailureClass;
use litellm_callbacks::machine::Machine;
use litellm_callbacks::route::{Layered, Route};

use crate::attempt::Attempt;
use crate::pick::DeploymentPicker;
use crate::plan::{Deployment, DeploymentId, RoutePlan};
use crate::report::CallReport;
use crate::signals::{Candidate, Signals};

/// What the router asks its host: every effect of a routing decision that cannot, or
/// should not, happen inside the machine. A plan that needs a model call or a database, a
/// pick that reads shared usage from Redis, the next fallback group, the wait before a
/// retry. Each one is visible in a trace.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RoutingOp {
    ResolvePlan,
    /// Choose among the group's remaining deployments. `None` leaves the group.
    /// The host applies its own signals; `load` here is what the router knows locally.
    Pick {
        group: u32,
        candidates: Vec<Candidate>,
    },
    /// The current group is done with: it ran out of retries and candidates for a failure
    /// of this class. `depth` counts the groups tried after the primary so far; `tried`
    /// is every deployment attempted. `None` ends the call with the last error.
    NextGroup {
        class: FailureClass,
        depth: u32,
        tried: Vec<DeploymentId>,
    },
    /// Wait before the next attempt. The host sleeps where the caller's task can be
    /// cancelled and answers `Slept`; a zero wait is still asked, as Python sleeps zero.
    Backoff {
        attempt: u32,
        duration: Duration,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RoutingResult {
    Plan(Box<RoutePlan>),
    Picked(Option<DeploymentId>),
    Group(Option<Vec<Deployment>>),
    Slept,
}

#[derive(Clone, Debug)]
pub enum PlanSource {
    Plan(Box<RoutePlan>),
    /// The router's first op is [`RoutingOp::ResolvePlan`].
    Host,
}

impl From<RoutePlan> for PlanSource {
    fn from(plan: RoutePlan) -> Self {
        Self::Plan(Box::new(plan))
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
    type Chunk = std::convert::Infallible;
}

/// The route a [`crate::Router`] over attempts `A` presents to its host.
pub type Routed<A> = Layered<Routing<A>, <A as Machine>::Route>;
