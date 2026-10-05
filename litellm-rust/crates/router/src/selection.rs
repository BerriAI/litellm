use serde::Serialize;

use crate::Error;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Candidate {
    pub deployment_id: String,
    pub model_name: String,
    pub model: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct SelectionContext {
    pub model: String,
    pub candidates: Vec<Candidate>,
    pub attempt: u32,
    pub session_id: Option<String>,
}

pub trait Selector {
    fn select(&self, context: &SelectionContext) -> Result<String, Error>;
}
