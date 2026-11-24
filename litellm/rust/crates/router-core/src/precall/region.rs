//! The region stage of `_pre_call_checks` (`is_region_allowed`): when the request names an
//! `allowed_model_region`, a deployment is dropped unless its `region_name` equals it. A
//! deployment without a `region_name` is dropped, and a null or absent region checks nothing.

use super::PreCallFacts;
use crate::registry::Deployment;

/// Whether `deployment` is outside the request's allowed region.
pub(super) fn outside(deployment: &Deployment, facts: &PreCallFacts) -> bool {
    facts
        .allowed_model_region
        .as_ref()
        .filter(|allowed| !allowed.is_null())
        .is_some_and(|allowed| deployment.raw_params.get("region_name") != Some(allowed))
}
