//! The Python `Router` in Rust: retries, fallback groups, deployment selection and the
//! trace identity that spans every attempt of one user call.
//!
//! [`Router`] is a [`Machine`] that wraps the per-attempt machines a route produces. It
//! forwards each attempt's host ops and chunks unchanged, so the driver that already polls
//! a route machine polls a router the same way. Exactly one terminal outcome leaves `run`.
//!
//! Everything the loop decides is data a test can state: a [`RoutePlan`] says what may be
//! tried, [`decide`] turns one failed attempt and its [`Situation`] into a [`Decision`],
//! and every effect of that decision (the wait, the next group, the next pick) is a
//! [`RoutingOp`] the host answers, so a trace shows it. The seams match what litellm's
//! routing options vary independently: plan resolution (which group, which fallback chain,
//! produced by the host before the loop or answered lazily through [`RoutingOp::NextGroup`]),
//! selection within a group ([`DeploymentPicker`] over [`Candidate`]s whose [`Load`] comes
//! from [`Signals`]), and the loop itself. A new strategy is one picker impl plus, if it
//! learns, one sink over the event stream.

mod attempt;
mod clock;
mod decide;
mod layer;
mod pick;
mod plan;
mod report;
mod router;
mod routing;
mod signals;

pub use attempt::{Attempt, AttemptContext, AttemptFactory};
pub use clock::{Clock, TokioClock};
pub use decide::{ChainsConfigured, Decision, Situation, decide, retries_allowed};
pub use layer::{PerAttempt, RouterLayer};
pub use pick::{
    DeploymentPicker, LeastBusy, LowestLatency, RoundRobin, UsageBased, WeightedShuffle,
};
pub use plan::{
    Deployment, DeploymentId, FallbackChains, Fallbacks, Retries, RetryPolicy, RoutePlan,
};
pub use report::{AttemptRecord, CallFailure, CallReport};
pub use router::Router;
pub use routing::{Picker, PlanSource, Routed, Routing, RoutingOp, RoutingResult};
pub use signals::{Candidate, Load, NoSignals, Signals};

pub use litellm_callbacks::failure::{Classified, FailureClass};
pub use litellm_callbacks::layer::{Layer, Stack};
pub use litellm_callbacks::machine::Machine;
pub use litellm_callbacks::route::{Layered, LayeredOp, LayeredResult};
