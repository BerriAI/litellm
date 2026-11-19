//! Serde helpers that write Prisma's `timestamp(3)` columns, which hold UTC wall time without a zone, as
//! RFC 3339 strings ending in `+00:00`, which Python's
//! `datetime.fromisoformat` reads on every supported version (3.10 rejects a `Z` suffix). They read
//! back what pydantic's JSON mode writes for such a column: RFC 3339 with any offset, or no offset
//! for a time already in UTC.

use chrono::{DateTime, NaiveDateTime, SecondsFormat};
use serde::{Deserialize, Deserializer, Serializer, de::Error};

/// Writes `value` as RFC 3339 in UTC with a `+00:00` offset.
pub(crate) fn serialize<S: Serializer>(
    value: &NaiveDateTime,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(
        &value
            .and_utc()
            .to_rfc3339_opts(SecondsFormat::AutoSi, false),
    )
}

/// Reads an RFC 3339 time as UTC wall time, taking a time without an offset as UTC.
pub(crate) fn deserialize<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<NaiveDateTime, D::Error> {
    let text = String::deserialize(deserializer)?;
    DateTime::parse_from_rfc3339(&text)
        .map(|at| at.naive_utc())
        .or_else(|_| text.parse())
        .map_err(D::Error::custom)
}

/// [`serialize`] and [`deserialize`] for nullable columns, with `null` for a missing value.
pub(crate) mod option {
    use chrono::NaiveDateTime;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(
        value: &Option<NaiveDateTime>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match value {
            Some(value) => super::serialize(value, serializer),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<NaiveDateTime>, D::Error> {
        #[derive(Deserialize)]
        struct At(#[serde(deserialize_with = "super::deserialize")] NaiveDateTime);

        Ok(Option::<At>::deserialize(deserializer)?.map(|At(at)| at))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde::Deserialize;

    use super::*;

    #[derive(Deserialize)]
    struct Row(#[serde(with = "option")] Option<NaiveDateTime>);

    fn at(text: &str) -> Option<NaiveDateTime> {
        Some(text.parse().unwrap())
    }

    #[rstest]
    #[case::utc_as_z(r#""2026-11-19T10:00:00.123000Z""#, at("2026-11-19T10:00:00.123"))]
    #[case::utc_as_an_offset(r#""2026-11-19T10:00:00+00:00""#, at("2026-11-19T10:00:00"))]
    #[case::another_offset(r#""2026-11-19T19:00:00+09:00""#, at("2026-11-19T10:00:00"))]
    #[case::no_offset(r#""2026-11-19T10:00:00.5""#, at("2026-11-19T10:00:00.5"))]
    #[case::null("null", None)]
    fn a_written_time_reads_back_as_utc_wall_time(
        #[case] json: &str,
        #[case] expected: Option<NaiveDateTime>,
    ) {
        assert_eq!(serde_json::from_str::<Row>(json).unwrap().0, expected);
    }

    #[rstest]
    #[case::not_a_time(r#""tomorrow""#)]
    #[case::a_number("1763546400")]
    fn anything_else_is_refused(#[case] json: &str) {
        assert!(serde_json::from_str::<Row>(json).is_err());
    }
}
