//! The router setting router-core does not read that reaches only the calls for some groups.
//! The settings it serves through a host step are not named here: `Router::reach` reports those
//! steps for each group a request reaches, and `model` names the ones the native host does not
//! answer.

use std::collections::BTreeMap;

use litellm_config::{OptionalPreCallCheck, RouterSettings};

fn pins(affinity: &Option<BTreeMap<String, Vec<OptionalPreCallCheck>>>, group: &str) -> bool {
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
