#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("router is closed")]
    Closed,
    #[error("deployment ID {id} is duplicated")]
    DuplicateDeployment { id: String },
    #[error("model group {model} is not configured")]
    UnknownModel { model: String },
    #[error("deployment {id} is not a candidate for this request")]
    InvalidSelection { id: String },
    #[error("attempt authorization was denied")]
    Unauthorized,
    #[error("routed execution is not implemented in the API scaffold")]
    NotImplemented,
}
