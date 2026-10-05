use serde::Serialize;

use crate::selection::Candidate;

#[derive(Clone, Debug, Serialize)]
pub struct Snapshot {
    pub generation: u64,
    pub closed: bool,
    pub deployments: Vec<Candidate>,
}
