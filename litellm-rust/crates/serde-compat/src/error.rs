#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("unrecognized boolean token")]
pub struct InvalidBoolean;
