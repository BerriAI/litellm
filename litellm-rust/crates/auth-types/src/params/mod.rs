mod aws;
mod vertex;

pub use aws::{AwsKeys, AwsKeysPartial, AwsParams};
pub use vertex::{VertexKeys, VertexKeysPartial, VertexParams};

use serde::Serialize;
use serde_json::{Map, Value};

fn as_litellm_params(params: &impl Serialize) -> Map<String, Value> {
    match serde_json::to_value(params) {
        Ok(Value::Object(map)) => map,
        _ => Map::new(),
    }
}
