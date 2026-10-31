use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Error, strum::VariantArray)]
pub enum Error {
    #[error("expected a UTC window HH:MM-HH:MM")]
    InvalidUtcWindow,
    #[error("expected an ISO weekday number 1-7 or an English day name")]
    InvalidWeekday,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::Error;
    use crate::schedule::{UtcWindow, Weekday};

    #[rstest]
    #[case::window("not-a-window".parse::<UtcWindow>().unwrap_err(), Error::InvalidUtcWindow)]
    #[case::weekday("not-a-weekday".parse::<Weekday>().unwrap_err(), Error::InvalidWeekday)]
    fn schedule_parsers_share_the_crate_error(#[case] error: Error, #[case] expected: Error) {
        assert_eq!(error, expected);
    }
}
