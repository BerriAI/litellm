use std::sync::{Arc, Mutex};

use litellm_cache::Error;
use litellm_cache_redis::connection::ConnectionRef;

use crate::search::value_text;

/// The valkey-search index one cache writes to, with the dimension it was last ensured for.
#[derive(Clone)]
pub(crate) struct IndexState {
    pub(crate) name: String,
    pub(crate) prefix: String,
    pub(crate) dimension: Arc<Mutex<Option<usize>>>,
    pub(crate) similarity_threshold: f64,
}

/// `_ensure_index_sync` / `_ensure_index_async`: create the TAG + HNSW index once per dimension,
/// and accept an existing index unless it reports a different dimension.
pub(crate) fn ensure_index(
    connection: &mut ConnectionRef<'_>,
    index: &IndexState,
    dimension: usize,
) -> Result<(), Error> {
    if index
        .dimension
        .lock()
        .map_err(|_| Error::Unavailable)?
        .is_some_and(|existing| existing == dimension)
    {
        return Ok(());
    }
    let create = redis::cmd("FT.CREATE")
        .arg(&index.name)
        .arg("ON")
        .arg("HASH")
        .arg("PREFIX")
        .arg(1)
        .arg(&index.prefix)
        .arg("SCHEMA")
        .arg("litellm_cache_key")
        .arg("TAG")
        .arg("embedding")
        .arg("VECTOR")
        .arg("HNSW")
        .arg(6)
        .arg("TYPE")
        .arg("FLOAT32")
        .arg("DIM")
        .arg(dimension)
        .arg("DISTANCE_METRIC")
        .arg("COSINE")
        .query::<String>(connection)
        .map(|_| ())
        .map_err(|error| error.to_string());
    if let Err(message) = create {
        if !message.to_ascii_lowercase().contains("already exists") {
            return Err(Error::Unavailable);
        }
        let info = redis::cmd("FT.INFO")
            .arg(&index.name)
            .query::<redis::Value>(connection)
            .map_err(|_| Error::Unavailable)?;
        if index_dimension_from_info(&info).is_some_and(|existing| existing != dimension) {
            return Err(Error::Unavailable);
        }
    }
    *index.dimension.lock().map_err(|_| Error::Unavailable)? = Some(dimension);
    Ok(())
}

/// `_extract_index_dim`: flatten each attribute one level and read the value after
/// `dimensions`.
fn index_dimension_from_info(value: &redis::Value) -> Option<usize> {
    let redis::Value::Array(values) = value else {
        return None;
    };
    let attributes = values.windows(2).find_map(|pair| {
        (value_text(&pair[0]).as_deref() == Some("attributes")).then_some(&pair[1])
    })?;
    let redis::Value::Array(fields) = attributes else {
        return None;
    };
    fields.iter().find_map(|field| {
        let redis::Value::Array(values) = field else {
            return None;
        };
        let values = values
            .iter()
            .flat_map(|value| match value {
                redis::Value::Array(values) => values.as_slice(),
                _ => std::slice::from_ref(value),
            })
            .collect::<Vec<_>>();
        values.windows(2).find_map(|pair| {
            (value_text(pair[0]).as_deref() == Some("dimensions"))
                .then(|| value_text(pair[1]).and_then(|value| value.parse().ok()))
                .flatten()
        })
    })
}
