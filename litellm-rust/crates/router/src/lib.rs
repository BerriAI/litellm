//! The Python `Router` in Rust: retries, fallback groups, deployment selection and the
//! logical-call identity that spans every attempt of one user call.
//!
//! [`Router`] is a [`Machine`] that wraps the per-attempt machines a route produces. It
//! forwards each attempt's host ops unchanged, so the driver that already polls a route
//! machine polls a router the same way. Exactly one terminal outcome leaves `run`.
//!
//! Three seams, matching what litellm's routing options vary independently:
//! plan resolution (which model group, which fallback chain: produced by the host before the
//! loop, as a [`RoutePlan`]), selection within a group ([`DeploymentPicker`] over
//! [`Candidate`]s whose [`Load`] comes from [`Signals`]), and the loop itself ([`Router`]).
//! A new strategy is one picker impl plus, if it learns, one sink over the event stream.
//!
//! Plan resolution and selection can each stay in process or go to the host. The router
//! presents a [`Routed`] route: its own [`RoutingOp`]s beside the attempt's ops, so a host
//! that resolves plans with a model call or picks from shared usage answers them inline,
//! the way it answers any route op.

mod attempt;
mod clock;
mod layer;
mod pick;
mod plan;
mod report;
mod router;
mod routing;
mod signals;

pub use attempt::{Attempt, AttemptContext, AttemptDisposition, AttemptError, AttemptFactory};
pub use clock::{Clock, TokioClock};
pub use layer::{PerAttempt, RouterLayer};
pub use pick::{
    DeploymentPicker, LeastBusy, LowestLatency, RoundRobin, UsageBased, WeightedShuffle,
};
pub use plan::{Deployment, DeploymentId, LogicalCallId, RetryPolicy, RoutePlan};
pub use report::{AttemptRecord, CallFailure, CallReport};
pub use router::Router;
pub use routing::{Picker, PlanSource, Routed, Routing, RoutingOp, RoutingResult};
pub use signals::{Candidate, Load, NoSignals, Signals};

pub use litellm_callbacks::layer::{Layer, Stack};
pub use litellm_callbacks::machine::Machine;
pub use litellm_callbacks::route::{Layered, LayeredOp, LayeredResult};
