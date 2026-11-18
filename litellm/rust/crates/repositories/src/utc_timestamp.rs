//! Serde helpers that write Prisma's `timestamp(3)` columns, which hold UTC wall time without a zone, as
//! RFC 3339 strings ending in `+00:00`, which Python's
//! `datetime.fromisoformat` reads on every supported version (3.10 rejects a `Z` suffix).

use chrono::{NaiveDateTime, SecondsFormat};
use serde::Serializer;

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

/// [`serialize`] for nullable columns, writing `null` for a missing value.
pub(crate) mod option {
    use chrono::NaiveDateTime;
    use serde::Serializer;

    pub fn serialize<S: Serializer>(
        value: &Option<NaiveDateTime>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match value {
            Some(value) => super::serialize(value, serializer),
            None => serializer.serialize_none(),
        }
    }
}
