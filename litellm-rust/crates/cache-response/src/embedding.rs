use serde::Serialize;
use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PartialHits {
    pub values: Vec<Option<Value>>,
    pub missing_indices: Vec<usize>,
}

impl PartialHits {
    pub fn new(values: Vec<Option<Value>>) -> Self {
        let missing_indices = values
            .iter()
            .enumerate()
            .filter_map(|(index, value)| value.is_none().then_some(index))
            .collect();
        Self {
            values,
            missing_indices,
        }
    }
}
