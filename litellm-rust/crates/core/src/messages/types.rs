use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MessagesRequestData {
    pub body: Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MessagesResponseData {
    pub body: Value,
}

impl MessagesResponseData {
    pub fn into_json(self) -> Value {
        self.body
    }
}
