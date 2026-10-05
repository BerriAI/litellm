use std::collections::BTreeMap;

use schemars::Schema;

pub fn schemas() -> BTreeMap<&'static str, Schema> {
    BTreeMap::from([("TraceQueryHelp", crate::query::help_schema())])
}
