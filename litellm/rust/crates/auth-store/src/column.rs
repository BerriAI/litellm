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
    unset_or(name, parsed(name, raw)?)
}

/// [`nullable`] for a field the model's `set_model_info` validator runs `json.loads` on when it
/// holds a string, since older writers stored `json.dumps` text in the column.
pub(crate) fn nullable_or_text<V: FromValue>(
    name: &'static str,
    raw: Option<&RawJson>,
) -> Result<Option<V>, Error> {
    unset_or(name, unquoted(name, parsed(name, raw)?)?)
}

fn parsed(name: &'static str, raw: Option<&RawJson>) -> Result<Value, Error> {
    raw.map_or(Ok(Value::None), |raw| {
        json::loads(raw.get()).map_err(|_| Error::row(name))
    })
}

fn unquoted(name: &'static str, value: Value) -> Result<Value, Error> {
    match value {
        Value::Str(text) => json::loads(&text).map_err(|_| Error::row(name)),
        value => Ok(value),
    }
}

fn unset_or<V: FromValue>(name: &'static str, value: Value) -> Result<Option<V>, Error> {
    match value {
        Value::None => Ok(None),
        value => V::from_value(&value)
            .map(Some)
            .map_err(|_| Error::row(name)),
    }
}

/// A column the model requires that the table lets be NULL, or the column it refuses the row for.
pub(crate) fn required<T>(column: &'static str, value: Option<T>) -> Result<T, Error> {
    value.ok_or(Error::row(column))
}

/// A timestamp column as Python's driver hands it out: an aware UTC `datetime`, or the column the
/// model refuses for a year a `datetime` cannot hold.
pub(crate) fn instant(column: &'static str, at: NaiveDateTime) -> Result<IsoDateTime, Error> {
    match at.year() {
        1..=9999 => Ok(IsoDateTime::utc(at.and_utc().timestamp_micros())),
        _ => Err(Error::row(column)),
    }
}

/// [`instant`] for a column that may be NULL.
pub(crate) fn time(
    column: &'static str,
    at: Option<NaiveDateTime>,
) -> Result<Option<IsoDateTime>, Error> {
    at.map(|at| instant(column, at)).transpose()
}
