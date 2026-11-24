//! Whether a caller may call a route or model. Every check is pure: nothing here takes a host,
//! so the orchestrators run each one as soon as its input exists.

mod ask;
mod credential;
mod denial;
mod eligibility;
mod models;
mod policy;
mod principal;
mod team;

pub use ask::ModelAsk;
pub use credential::{check_admin_route, check_rbac_role, check_scope_based_access};
pub use denial::{AuthzDenial, DeniedList, Raised};
pub(crate) use eligibility::{TeamEligibility, Verdict};
pub use models::{Aliases, Candidates, Layer, ModelList, Names, authorize, candidates};
pub(crate) use policy::{RoutePolicy, Target};
