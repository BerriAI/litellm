use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct CompilerProof {
    proc_macros: bool,
    dependency: String,
}

pub fn build_proof() -> CompilerProof {
    CompilerProof {
        proc_macros: true,
        dependency: "litellm-core".to_owned(),
    }
}
