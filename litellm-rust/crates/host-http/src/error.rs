#[derive(Debug, PartialEq, thiserror::Error)]
pub enum Error<E> {
    #[error(transparent)]
    Call(E),
    #[error("unexpected HTTP host operation")]
    Protocol,
}
