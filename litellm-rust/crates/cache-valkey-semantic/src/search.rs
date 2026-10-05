use std::time::Duration;

use litellm_cache::{Error, semantic::SemanticLookup};
use litellm_cache_redis::connection::ConnectionRef;
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::index::{IndexState, ensure_index};

/// `_scope_tag`: valkey-search TAG fields cannot match arbitrary keys verbatim, so scopes are
/// the key's lowercase SHA-256.
pub(crate) fn scope_tag(key: &str) -> String {
    let digest = Sha256::digest(key.as_bytes());
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub(crate) fn embedding_bytes(embedding: &[f32]) -> Vec<u8> {
    embedding
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// `HSET` a fresh `<prefix><scope>:<uuid4>` document, then `EXPIRE` it when a TTL is set.
pub(crate) fn write_document(
    connection: &mut ConnectionRef<'_>,
    index: &IndexState,
    scope: &str,
    prompt: &str,
    response: Vec<u8>,
    vector: Vec<u8>,
    ttl: Option<Duration>,
) -> Result<(), Error> {
    ensure_index(connection, index, vector.len() / size_of::<f32>())?;
    let document = format!("{}{scope}:{}", index.prefix, Uuid::new_v4());
    let mut pipeline = redis::pipe();
    pipeline
        .cmd("HSET")
        .arg(&document)
        .arg("litellm_cache_key")
        .arg(scope)
        .arg("prompt")
        .arg(prompt)
        .arg("response")
        .arg(response)
        .arg("embedding")
        .arg(vector)
        .ignore();
    if let Some(ttl) = ttl {
        pipeline
            .cmd("EXPIRE")
            .arg(&document)
            .arg(ttl.as_secs())
            .ignore();
    }
    pipeline
        .query::<()>(connection)
        .map_err(|_| Error::Unavailable)
}

/// The KNN-1 search within `scope`: the closest document's similarity, and its stored response
/// when that similarity reaches the threshold. No document reads as a similarity of `0.0`.
pub(crate) fn search_document(
    connection: &mut ConnectionRef<'_>,
    index: &IndexState,
    scope: &str,
    vector: Vec<u8>,
) -> Result<SemanticLookup<Vec<u8>>, Error> {
    ensure_index(connection, index, vector.len() / size_of::<f32>())?;
    let query =
        format!("(@litellm_cache_key:{{{scope}}})=>[KNN 1 @embedding $vec AS vector_distance]");
    let response = redis::cmd("FT.SEARCH")
        .arg(&index.name)
        .arg(query)
        .arg("PARAMS")
        .arg(2)
        .arg("vec")
        .arg(vector)
        .arg("RETURN")
        .arg(2)
        .arg("response")
        .arg("vector_distance")
        .arg("DIALECT")
        .arg(2)
        .query::<redis::Value>(connection)
        .map_err(|_| Error::Unavailable)?;
    let Some(fields) = search_fields(response)? else {
        return Ok(SemanticLookup::miss(Some(0.0)));
    };
    let field = |name: &str| {
        fields
            .iter()
            .find_map(|(field, value)| (field == name).then(|| value.clone()))
            .ok_or(Error::InvalidEntry)
    };
    let response = field("response")?;
    let similarity = 1.0 - parse_f64(&field("vector_distance")?)?;
    Ok(SemanticLookup {
        value: (similarity >= index.similarity_threshold).then_some(response),
        similarity: Some(similarity),
    })
}

type SearchFields = Vec<(String, Vec<u8>)>;

fn search_fields(value: redis::Value) -> Result<Option<SearchFields>, Error> {
    let redis::Value::Array(values) = value else {
        return Err(Error::InvalidEntry);
    };
    let total = parse_i64(values.first().ok_or(Error::InvalidEntry)?)?;
    if total <= 0 || values.len() < 3 {
        return Ok(None);
    }
    let redis::Value::Array(fields) = &values[2] else {
        return Err(Error::InvalidEntry);
    };
    let (pairs, remainder) = fields.as_chunks::<2>();
    if !remainder.is_empty() {
        return Err(Error::InvalidEntry);
    }
    pairs
        .iter()
        .map(|pair| {
            Ok((
                value_text(&pair[0]).ok_or(Error::InvalidEntry)?,
                value_bytes(&pair[1])?,
            ))
        })
        .collect::<Result<Vec<_>, Error>>()
        .map(Some)
}

fn parse_i64(value: &redis::Value) -> Result<i64, Error> {
    value_text(value)
        .ok_or(Error::InvalidEntry)?
        .parse()
        .map_err(|_| Error::InvalidEntry)
}

fn parse_f64(value: &[u8]) -> Result<f64, Error> {
    std::str::from_utf8(value)
        .map_err(|_| Error::InvalidEntry)?
        .parse()
        .map_err(|_| Error::InvalidEntry)
}

pub(crate) fn value_text(value: &redis::Value) -> Option<String> {
    match value {
        redis::Value::BulkString(bytes) => String::from_utf8(bytes.clone()).ok(),
        redis::Value::SimpleString(value) => Some(value.clone()),
        redis::Value::Int(value) => Some(value.to_string()),
        _ => None,
    }
}

fn value_bytes(value: &redis::Value) -> Result<Vec<u8>, Error> {
    match value {
        redis::Value::BulkString(bytes) => Ok(bytes.clone()),
        redis::Value::SimpleString(value) => Ok(value.as_bytes().to_vec()),
        redis::Value::Int(value) => Ok(value.to_string().into_bytes()),
        _ => Err(Error::InvalidEntry),
    }
}
