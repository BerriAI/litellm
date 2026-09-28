use crate::Effect;

#[derive(Debug, PartialEq, thiserror::Error)]
pub enum Error {
    #[error("accounting requires USD charges, received {currency}")]
    InvalidCurrency { currency: &'static str },
    #[error("USD charges must be nonnegative")]
    NegativeCost,
    #[error(transparent)]
    Money(#[from] rusty_money::MoneyError),
    #[error("the call already has a terminal outcome")]
    AlreadyTerminal,
    #[error("the call has no terminal outcome")]
    NotTerminal,
    #[error("{effect:?} can only be retried after a confirmed not-applied result")]
    UnsafeRetry { effect: Effect },
}
