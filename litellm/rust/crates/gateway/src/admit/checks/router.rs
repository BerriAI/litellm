//! The router settings router-core does not serve whole: the one it does not read that reaches
//! only the calls for some groups, and the `optional_pre_call_checks` entries it leaves to the host.
//! The settings it serves through a host step are not named here: `Router::reach` reports those
//! steps for each group a request reaches, and `model` names the ones the native host does not
//! answer.

use std::collections::BTreeMap;

use litellm_config::{OptionalPreCallCheck, RouterSettings};
use litellm_router_core::pre_call_checks;

/// Whether `optional_pre_call_checks` names a check router-core asks the host to run.
pub(super) fn checked_by_the_host(checks: &Option<Vec<OptionalPreCallCheck>>) -> bool {
    checks
        .iter()
        .flatten()
        .any(|check| !pre_call_checks::core_runs(check))
}

fn pins(affinity: &Option<BTreeMap<String, Vec<String>>>, group: &str) -> bool {
    affinity
        .as_ref()
        .is_some_and(|groups| groups.contains_key(group))
}

/// The setting router-core does not read that reaches a call for `group`.
pub(super) fn group(settings: &RouterSettings, group: &str) -> Option<Box<str>> {
    let RouterSettings {
        model_group_affinity_config,
        ..
    } = settings;
    set!(
        "router_settings.",
        [],
        by: [model_group_affinity_config => |affinity| pins(affinity, group)]
    )
    .next()
}
