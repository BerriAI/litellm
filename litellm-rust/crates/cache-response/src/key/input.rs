use serde_json::Value;

#[derive(Clone, Debug, PartialEq)]
pub struct CacheKeyInput {
    pub(super) surface: String,
    pub(super) parameters: Value,
}

impl CacheKeyInput {
    pub fn new(surface: &str, parameters: Value) -> Self {
        Self {
            surface: surface.to_owned(),
            parameters,
        }
    }
}
