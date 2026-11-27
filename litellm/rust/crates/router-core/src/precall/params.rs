//! The params stage of `_pre_call_checks`: a deployment that lacks a requested param is dropped,
//! unless every deployment the earlier stages kept lacks it. Then only the ones that reject it
//! are dropped, those whose call settings neither drop nor allow it. A group whose every
//! deployment rejects it raises one `UnsupportedParams`.

use std::sync::Arc;

use super::PreCallFacts;
use crate::{error::Error, registry::Deployment};

/// Which of `left` the stage drops: the ones lacking a requested param, unless every one lacks
/// it, and then only the ones that reject it.
pub(super) fn failing(left: &[&Deployment], facts: &PreCallFacts) -> Vec<bool> {
    let checked = !facts.requested_params.is_empty();
    let lacking: Vec<bool> = left
        .iter()
        .map(|deployment| checked && facts.of(deployment).lacks_params)
        .collect();
    if lacking.is_empty() || lacking.contains(&false) {
        return lacking;
    }
    left.iter()
        .map(|deployment| facts.of(deployment).rejects_params)
        .collect()
}

/// The error for a group with nothing left, when every one of its deployments rejects a
/// requested param.
pub(super) fn unsupported(
    model: &str,
    pool: &[Arc<Deployment>],
    facts: &PreCallFacts,
) -> Option<Error> {
    let every_rejects = !facts.requested_params.is_empty()
        && !pool.is_empty()
        && pool
            .iter()
            .all(|deployment| facts.of(deployment).rejects_params);
    every_rejects.then(|| Error::UnsupportedParams {
        model: model.to_owned(),
        params: facts.requested_params.clone(),
    })
}
