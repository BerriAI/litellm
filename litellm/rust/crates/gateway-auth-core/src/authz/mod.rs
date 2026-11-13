//! Whether a caller may call a route or model. Every check is pure: nothing here takes a host,
//! so the orchestrators run each one as soon as its input exists.

mod credential;
mod eligibility;
mod policy;
mod principal;
mod team;

pub use credential::{check_admin_route, check_rbac_role, check_scope_based_access};
pub(crate) use eligibility::{TeamEligibility, Verdict};
pub(crate) use policy::{RoutePolicy, Target};
