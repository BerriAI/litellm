use litellm_core_utils::Invalid;

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("invalid environment variable {name}: {invalid}")]
pub struct Error {
    pub name: &'static str,
    pub invalid: Invalid,
}
