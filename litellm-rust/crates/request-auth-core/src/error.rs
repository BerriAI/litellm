#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid auth effect answer")]
    InvalidAnswer,
    #[error("can't compare offset-naive and offset-aware datetimes")]
    IncompatibleDateTimes,
}
