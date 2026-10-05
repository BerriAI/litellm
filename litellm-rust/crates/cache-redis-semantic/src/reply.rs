pub(crate) fn string_value(value: &redis::Value) -> Option<String> {
    match value {
        redis::Value::BulkString(bytes) => String::from_utf8(bytes.clone()).ok(),
        redis::Value::SimpleString(text) => Some(text.clone()),
        redis::Value::VerbatimString { text, .. } => Some(text.clone()),
        _ => None,
    }
}

pub(crate) fn number_value(value: &redis::Value) -> Option<f64> {
    match value {
        redis::Value::Int(number) => Some(*number as f64),
        redis::Value::Double(number) => Some(*number),
        _ => string_value(value).and_then(|text| text.parse().ok()),
    }
}

pub(crate) fn first_document(result: &redis::Value) -> Option<&[redis::Value]> {
    let redis::Value::Array(items) = result else {
        return None;
    };
    let [count, _document_id, fields, ..] = items.as_slice() else {
        return None;
    };
    if !matches!(count, redis::Value::Int(count) if *count > 0) {
        return None;
    }
    match fields {
        redis::Value::Array(fields) => Some(fields.as_slice()),
        _ => None,
    }
}

fn field_value<'a>(fields: &'a [redis::Value], name: &str) -> Option<&'a redis::Value> {
    fields
        .as_chunks::<2>()
        .0
        .iter()
        .find(|pair| string_value(&pair[0]).as_deref() == Some(name))
        .map(|pair| &pair[1])
}

pub(crate) fn string_field(fields: &[redis::Value], name: &str) -> Option<String> {
    field_value(fields, name).and_then(string_value)
}

pub(crate) fn number_field(fields: &[redis::Value], name: &str) -> Option<f64> {
    field_value(fields, name).and_then(number_value)
}

pub(crate) fn bytes_field(fields: &[redis::Value], name: &str) -> Option<Vec<u8>> {
    match field_value(fields, name)? {
        redis::Value::BulkString(bytes) => Some(bytes.clone()),
        redis::Value::SimpleString(text) => Some(text.clone().into_bytes()),
        _ => None,
    }
}
