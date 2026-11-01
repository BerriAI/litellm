use std::sync::Arc;

use indexmap::IndexMap;
use serde_json::Value;

use crate::catalog::Catalog;
use crate::register::Registrations;
use crate::wire::Fields;

/// One field of a fetched row over the layer below, RFC 7386. An absent field inherits, JSON `null` is `Remove`
#[derive(Clone, Debug, PartialEq)]
pub enum Patch {
    Set(Value),
    Remove,
}

/// The four layers served rows are composed from, lowest first
#[derive(Clone, Debug)]
#[expect(
    dead_code,
    reason = "02 and 07 add the functions that read these layers"
)]
pub struct Layers {
    /// 0, the wheel's embedded map
    bundled: Arc<Catalog>,
    /// 1, the last installed fetch, rejected rows left out
    fetched: Option<Arc<Catalog>>,
    /// 2, `register(persist=False)`, cleared by every install and refilled by the router replay
    transient: IndexMap<String, Fields>,
    /// 3, `register(persist=True)`, applied over every install
    persisted: Registrations,
}
