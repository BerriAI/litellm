use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
#[error("expected a UTC window HH:MM-HH:MM")]
pub struct InvalidUtcWindow;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
#[error("expected an ISO weekday number 1-7 or an English day name")]
pub struct InvalidWeekday;
