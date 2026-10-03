use base64::{Engine, engine::general_purpose::URL_SAFE};
use serde::{Deserialize, Serialize};

use crate::ReadError;

pub(super) fn encode_cursor<T: Serialize>(position: &T) -> String {
    URL_SAFE.encode(serde_json::to_vec(position).unwrap_or_default())
}

pub(super) fn decode_cursor<T: for<'de> Deserialize<'de>, E>(
    cursor: &str,
    kind: &'static str,
) -> Result<T, ReadError<E>> {
    URL_SAFE
        .decode(cursor)
        .ok()
        .and_then(|json| serde_json::from_slice(&json).ok())
        .ok_or(ReadError::InvalidCursor(kind))
}

pub(super) fn trace_position<E>(cursor: Option<&str>) -> Result<(i64, String), ReadError<E>> {
    let Some(cursor) = cursor.filter(|cursor| !cursor.is_empty()) else {
        return Ok((0, String::new()));
    };
    match decode_cursor::<(i64, String), E>(cursor, "trace")? {
        (start_ms, trace_ref) if start_ms > 0 && !trace_ref.is_empty() => Ok((start_ms, trace_ref)),
        _ => Err(ReadError::InvalidCursor("trace")),
    }
}

#[derive(Deserialize, Serialize)]
pub(super) struct ErrorPosition {
    pub(super) offset: u64,
    pub(super) version: String,
}

pub(super) fn error_position<E>(
    cursor: Option<&str>,
) -> Result<Option<ErrorPosition>, ReadError<E>> {
    let Some(cursor) = cursor else {
        return Ok(None);
    };
    let position: ErrorPosition = decode_cursor(cursor, "diagnostic")?;
    let valid_version = position.version.len() == 64
        && position
            .version
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'A'..=b'F').contains(&byte));
    if i64::try_from(position.offset).is_err() || !valid_version {
        return Err(ReadError::InvalidCursor("diagnostic"));
    }
    Ok(Some(position))
}

#[derive(Deserialize, Serialize)]
pub(super) struct SpanPosition {
    pub(super) trace_ref: String,
    pub(super) snapshot_ms: u64,
    pub(super) offset: usize,
    pub(super) version: String,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    fn trace_cursor_round_trips_the_last_listed_run() {
        let cursor = encode_cursor(&(1_790_742_989_377_i64, "4bad42b84e9de3ba46fc870185f8f023"));
        assert_eq!(
            trace_position::<std::io::Error>(Some(&cursor)).unwrap(),
            (
                1_790_742_989_377,
                "4bad42b84e9de3ba46fc870185f8f023".to_owned()
            )
        );
        assert_eq!(
            trace_position::<std::io::Error>(None).unwrap(),
            (0, String::new())
        );
        assert_eq!(
            trace_position::<std::io::Error>(Some("")).unwrap(),
            (0, String::new())
        );
    }

    #[rstest]
    #[case::not_base64("abc")]
    #[case::not_json("bm90LWpzb24=")]
    #[case::numeric_reference("WzEsIDJd")]
    #[case::zero_start("WzAsICJ0Il0=")]
    fn malformed_trace_cursors_are_rejected(#[case] cursor: &str) {
        let result: Result<(i64, String), ReadError<std::io::Error>> = trace_position(Some(cursor));
        assert!(matches!(result, Err(ReadError::InvalidCursor("trace"))));
    }

    #[rstest]
    #[case::not_base64("garbage")]
    #[case::missing_fields("e30=")]
    #[case::not_an_object("WzEsMl0=")]
    fn malformed_diagnostic_cursors_are_rejected(#[case] cursor: &str) {
        let result: Result<Option<ErrorPosition>, ReadError<std::io::Error>> =
            error_position(Some(cursor));
        assert!(matches!(
            result,
            Err(ReadError::InvalidCursor("diagnostic"))
        ));
    }

    #[rstest]
    #[case::lowercase_version("a".repeat(64))]
    #[case::short_version("A".repeat(63))]
    fn diagnostic_cursor_requires_a_content_version(#[case] version: String) {
        let cursor = encode_cursor(&ErrorPosition { offset: 1, version });
        let result: Result<Option<ErrorPosition>, ReadError<std::io::Error>> =
            error_position(Some(&cursor));
        assert!(matches!(
            result,
            Err(ReadError::InvalidCursor("diagnostic"))
        ));
    }
}
