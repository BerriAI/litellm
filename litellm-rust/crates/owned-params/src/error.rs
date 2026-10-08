#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("invalid request: extra_body must be an object")]
    ExtraBody,
    #[error("invalid request: body must be a JSON object")]
    Body,
}
