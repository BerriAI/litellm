//! One database column read as one field of the record the auth cache holds.

use chrono::{Datelike, NaiveDateTime};
use litellm_python_compat::{Value, datetime::IsoDateTime, json};
use litellm_redis_store::FromValue;
use litellm_repositories::RawJson;

use crate::Error;

/// A JSON column as the record's field `V`, or the column the model would refuse the row for.
pub(crate) fn column<V: FromValue>(name: &'static str, raw: &RawJson) -> Result<V, Error> {
    json::loads(raw.get())
        .ok()
        .and_then(|value| V::from_value(&value).ok())
        .ok_or(Error::row(name))
}

/// [`column`] for a field the model lets be unset: no value and a JSON `null` are both unset.
pub(crate) fn nullable<V: FromValue>(
    name: &'static str,
    raw: Option<&RawJson>,
) -> Result<Option<V>, Error> {
    match raw.map(|raw| json::loads(raw.get())) {
        None | Some(Ok(Value::None)) => Ok(None),
        Some(Ok(value)) => V::from_value(&value)
            .map(Some)
            .map_err(|_| Error::row(name)),
        Some(Err(_)) => Err(Error::row(name)),
    }
}

/// A timestamp column as Python's driver hands it out: an aware UTC `datetime`, or the column the
/// model refuses for a year a `datetime` cannot hold.
pub(crate) fn time(
    column: &'static str,
    at: Option<NaiveDateTime>,
) -> Result<Option<IsoDateTime>, Error> {
    at.map(|at| match at.year() {
        1..=9999 => Ok(IsoDateTime::utc(at.and_utc().timestamp_micros())),
        _ => Err(Error::row(column)),
    })
    .transpose()
}
