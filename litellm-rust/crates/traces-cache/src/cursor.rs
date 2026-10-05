use base64::{Engine, engine::general_purpose::URL_SAFE};
use litellm_traces::store::{RunCursor, RunOrder, RunRow, SpanPart};
use serde::{Deserialize, Serialize};

use crate::ReadError;

#[derive(Deserialize, Serialize)]
#[serde(
    tag = "kind",
    content = "position",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub(super) enum Cursor {
    Run(RunPosition),
    Span(SpanPosition),
    Text(TextPosition),
}

impl Cursor {
    fn kind(&self) -> &'static str {
        match self {
            Self::Run(_) => "trace",
            Self::Span(_) => "span",
            Self::Text(_) => "diagnostic",
        }
    }

    pub(super) fn encode(&self) -> String {
        URL_SAFE.encode(serde_json::to_vec(self).unwrap_or_default())
    }

    fn decode<E>(cursor: &str, kind: &'static str) -> Result<Self, ReadError<E>> {
        URL_SAFE
            .decode(cursor)
            .ok()
            .and_then(|json| serde_json::from_slice::<Self>(&json).ok())
            .filter(|decoded| decoded.kind() == kind)
            .ok_or(ReadError::InvalidCursor(kind))
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RunPosition {
    order: RunOrder,
    value: i64,
    trace_ref: String,
}

impl RunPosition {
    pub(super) fn after(order: RunOrder, row: &RunRow) -> Self {
        let RunCursor { value, trace_ref } = order.cursor(row);
        Self {
            order,
            value,
            trace_ref,
        }
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SpanPosition {
    pub(super) trace_ref: String,
    pub(super) snapshot_ms: u64,
    pub(super) offset: usize,
    pub(super) version: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TextPosition {
    pub(super) part: SpanPart,
    pub(super) offset: u64,
    pub(super) version: String,
}

pub(super) fn run_position<E>(
    cursor: Option<&str>,
    order: RunOrder,
) -> Result<Option<RunCursor>, ReadError<E>> {
    let Some(cursor) = cursor.filter(|cursor| !cursor.is_empty()) else {
        return Ok(None);
    };
    match Cursor::decode(cursor, "trace")? {
        Cursor::Run(position) if position.order == order && !position.trace_ref.is_empty() => {
            Ok(Some(RunCursor {
                value: position.value,
                trace_ref: position.trace_ref,
            }))
        }
        _ => Err(ReadError::InvalidCursor("trace")),
    }
}

pub(super) fn span_position<E>(cursor: &str) -> Result<SpanPosition, ReadError<E>> {
    match Cursor::decode(cursor, "span")? {
        Cursor::Span(position) => Ok(position),
        _ => Err(ReadError::InvalidCursor("span")),
    }
}

pub(super) fn text_position<E>(
    cursor: Option<&str>,
    part: SpanPart,
) -> Result<Option<TextPosition>, ReadError<E>> {
    let Some(cursor) = cursor else {
        return Ok(None);
    };
    let Cursor::Text(position) = Cursor::decode(cursor, "diagnostic")? else {
        return Err(ReadError::InvalidCursor("diagnostic"));
    };
    let valid_version = position.version.len() == 64
        && position
            .version
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'A'..=b'F').contains(&byte));
    if position.part != part || i64::try_from(position.offset).is_err() || !valid_version {
        return Err(ReadError::InvalidCursor("diagnostic"));
    }
    Ok(Some(position))
}

#[cfg(test)]
mod tests {
    use litellm_traces::store::RunSortKey;
    use rstest::rstest;

    use super::*;

    fn run(order: RunOrder, value: i64, trace_ref: &str) -> String {
        Cursor::Run(RunPosition {
            order,
            value,
            trace_ref: trace_ref.into(),
        })
        .encode()
    }

    fn text(part: SpanPart, offset: u64, version: String) -> String {
        Cursor::Text(TextPosition {
            part,
            offset,
            version,
        })
        .encode()
    }

    fn span() -> String {
        Cursor::Span(SpanPosition {
            trace_ref: "ref".into(),
            snapshot_ms: 1,
            offset: 2,
            version: "A".repeat(64),
        })
        .encode()
    }

    fn json(value: serde_json::Value) -> String {
        URL_SAFE.encode(value.to_string())
    }

    const BY_ERRORS: RunOrder = RunOrder {
        key: RunSortKey::ErrorCount,
        descending: false,
    };

    #[rstest]
    #[case::newest(RunOrder::NEWEST, 1_790_742_989_377)]
    #[case::zero_value(BY_ERRORS, 0)]
    fn run_cursor_round_trips_under_its_own_order(#[case] order: RunOrder, #[case] value: i64) {
        let position = run_position::<std::io::Error>(Some(&run(order, value, "4BAD")), order)
            .unwrap()
            .unwrap();
        assert_eq!(
            (position.value, position.trace_ref.as_str()),
            (value, "4BAD")
        );
    }

    #[rstest]
    #[case::absent(None)]
    #[case::empty(Some(""))]
    fn missing_run_cursor_starts_from_the_first_page(#[case] cursor: Option<&str>) {
        assert!(
            run_position::<std::io::Error>(cursor, RunOrder::NEWEST)
                .unwrap()
                .is_none()
        );
    }

    #[rstest]
    #[case::not_base64("abc".into())]
    #[case::not_json(URL_SAFE.encode("not-json"))]
    #[case::untagged_tuple(json(serde_json::json!([1, "ref"])))]
    #[case::other_key(run(BY_ERRORS, 1, "ref"))]
    #[case::other_direction(run(RunOrder { descending: false, ..RunOrder::NEWEST }, 1, "ref"))]
    #[case::empty_ref(run(RunOrder::NEWEST, 1, ""))]
    #[case::span_cursor(span())]
    #[case::text_cursor(text(SpanPart::Error, 0, "A".repeat(64)))]
    #[case::without_order(json(serde_json::json!({"kind": "run", "position": {"value": 1, "trace_ref": "r"}})))]
    #[case::unknown_field(json(serde_json::json!({"kind": "run", "position": {"order": {"key": "start_ms", "descending": true}, "value": 1, "trace_ref": "r", "extra": 1}})))]
    fn run_cursors_not_minted_under_the_requested_order_are_rejected(#[case] cursor: String) {
        assert!(matches!(
            run_position::<std::io::Error>(Some(&cursor), RunOrder::NEWEST),
            Err(ReadError::InvalidCursor("trace"))
        ));
    }

    #[rstest]
    fn span_cursor_round_trips() {
        let position = span_position::<std::io::Error>(&span()).unwrap();
        assert_eq!(
            (
                position.trace_ref.as_str(),
                position.snapshot_ms,
                position.offset
            ),
            ("ref", 1, 2)
        );
    }

    #[rstest]
    #[case::run_cursor(run(RunOrder::NEWEST, 1, "ref"))]
    #[case::text_cursor(text(SpanPart::Error, 0, "a".repeat(64)))]
    fn other_kinds_are_not_span_cursors(#[case] cursor: String) {
        assert!(matches!(
            span_position::<std::io::Error>(&cursor),
            Err(ReadError::InvalidCursor("span"))
        ));
    }

    #[rstest]
    fn text_cursor_round_trips() {
        let position = text_position::<std::io::Error>(
            Some(&text(SpanPart::Error, 7, "A".repeat(64))),
            SpanPart::Error,
        )
        .unwrap()
        .unwrap();
        assert_eq!((position.offset, position.version), (7, "A".repeat(64)));
    }

    #[rstest]
    #[case::lowercase_version(text(SpanPart::Error, 1, "a".repeat(64)))]
    #[case::short_version(text(SpanPart::Error, 1, "A".repeat(63)))]
    #[case::offset_past_i64(text(SpanPart::Error, u64::MAX, "A".repeat(64)))]
    #[case::other_part(text(SpanPart::Output, 1, "A".repeat(64)))]
    #[case::span_cursor(span())]
    #[case::missing_fields(json(serde_json::json!({"kind": "text", "position": {}})))]
    fn malformed_text_cursors_are_rejected(#[case] cursor: String) {
        assert!(matches!(
            text_position::<std::io::Error>(Some(&cursor), SpanPart::Error),
            Err(ReadError::InvalidCursor("diagnostic"))
        ));
    }
}
