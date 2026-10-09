#[derive(Debug, thiserror::Error)]
pub enum HookError {
    #[error("a hook rejected the call: {reason}")]
    Rejected { reason: String },
}
