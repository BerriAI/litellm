use jiff::civil::Time;
use serde::{
    Deserialize, Deserializer,
    de::{Error, Unexpected},
};

/// `_DAILY_RESET_TIME_PATTERN`, which jiff alone is looser than: it reads a padded or a
/// one-digit number.
fn is_two_digits_each(text: &str) -> bool {
    matches!(
        text.as_bytes(),
        [h1, h2, b':', m1, m2] if [h1, h2, m1, m2].iter().all(|digit| digit.is_ascii_digit())
    )
}

/// `parse_daily_reset_time`: `HH:MM` in UTC, or the error Python's validator raises.
pub(crate) fn deserialize<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Time>, D::Error> {
    let Some(text) = Option::<String>::deserialize(deserializer)? else {
        return Ok(None);
    };
    is_two_digits_each(&text)
        .then(|| Time::strptime("%H:%M", &text).ok())
        .flatten()
        .map(Some)
        .ok_or_else(|| D::Error::invalid_value(Unexpected::Str(&text), &"HH:MM in UTC"))
}
