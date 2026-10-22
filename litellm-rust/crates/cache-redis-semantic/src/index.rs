use std::sync::OnceLock;

use litellm_cache::Error;
use litellm_cache_redis::connection::ConnectionRef;

use crate::reply::{number_value, string_value};

pub(crate) const CACHE_KEY_FIELD: &str = "litellm_cache_key";
pub(crate) const VECTOR_FIELD: &str = "prompt_vector";

/// The redisvl `SemanticCache` index, resolved once per cache: the configured name when its
/// schema fits, else `<name>_isolated`, recreated when that one is stale too.
pub(crate) struct Index {
    name: String,
    resolved: OnceLock<String>,
}

impl Index {
    pub(crate) fn new(name: String) -> Self {
        Self {
            name,
            resolved: OnceLock::new(),
        }
    }

    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    pub(crate) fn ensure(
        &self,
        connection: &mut ConnectionRef<'_>,
        dims: usize,
    ) -> Result<String, Error> {
        if let Some(name) = self.resolved.get() {
            return Ok(name.clone());
        }
        let name = match index_compatible(connection, &self.name, dims)? {
            Some(true) => self.name.clone(),
            Some(false) => self.isolated(connection, dims)?,
            None => match create_index(connection, &self.name, dims) {
                Ok(()) => self.name.clone(),
                Err(_) => match index_compatible(connection, &self.name, dims)? {
                    Some(true) => self.name.clone(),
                    Some(false) => self.isolated(connection, dims)?,
                    None => return Err(Error::Unavailable),
                },
            },
        };
        let _ = self.resolved.set(name.clone());
        Ok(name)
    }

    fn isolated(&self, connection: &mut ConnectionRef<'_>, dims: usize) -> Result<String, Error> {
        let name = format!("{}_isolated", self.name);
        match index_compatible(connection, &name, dims)? {
            Some(true) => Ok(name),
            Some(false) => {
                redis::cmd("FT.DROPINDEX")
                    .arg(&name)
                    .query::<()>(connection)
                    .map_err(|_| Error::Unavailable)?;
                create_index(connection, &name, dims)?;
                Ok(name)
            }
            None => {
                create_index(connection, &name, dims)?;
                Ok(name)
            }
        }
    }
}

fn create_index(connection: &mut ConnectionRef<'_>, name: &str, dims: usize) -> Result<(), Error> {
    redis::cmd("FT.CREATE")
        .arg(name)
        .arg("ON")
        .arg("HASH")
        .arg("PREFIX")
        .arg(1)
        .arg(name)
        .arg("SCORE")
        .arg(1.0)
        .arg("SCHEMA")
        .arg("prompt")
        .arg("TEXT")
        .arg("WEIGHT")
        .arg(1)
        .arg("response")
        .arg("TEXT")
        .arg("WEIGHT")
        .arg(1)
        .arg("inserted_at")
        .arg("NUMERIC")
        .arg("updated_at")
        .arg("NUMERIC")
        .arg(VECTOR_FIELD)
        .arg("VECTOR")
        .arg("FLAT")
        .arg(6)
        .arg("TYPE")
        .arg("FLOAT32")
        .arg("DIM")
        .arg(dims)
        .arg("DISTANCE_METRIC")
        .arg("COSINE")
        .arg(CACHE_KEY_FIELD)
        .arg("TAG")
        .arg("SEPARATOR")
        .arg(",")
        .query::<()>(connection)
        .map_err(|_| Error::Unavailable)
}

fn index_compatible(
    connection: &mut ConnectionRef<'_>,
    name: &str,
    dims: usize,
) -> Result<Option<bool>, Error> {
    let info = match redis::cmd("FT.INFO")
        .arg(name)
        .query::<redis::Value>(connection)
    {
        Ok(info) => info,
        Err(error) if unknown_index(&error) => return Ok(None),
        Err(_) => return Err(Error::Unavailable),
    };
    Ok(Some(schema_compatible(&info, dims)))
}

fn unknown_index(error: &redis::RedisError) -> bool {
    let message = error.to_string().to_lowercase();
    message.contains("unknown") && message.contains("index")
}

struct Attribute {
    name: Option<String>,
    field_type: Option<String>,
    dim: Option<f64>,
    data_type: Option<String>,
    distance_metric: Option<String>,
}

fn attribute(value: &redis::Value) -> Option<Attribute> {
    let redis::Value::Array(pairs) = value else {
        return None;
    };
    let mut attribute = Attribute {
        name: None,
        field_type: None,
        dim: None,
        data_type: None,
        distance_metric: None,
    };
    for pair in pairs.as_chunks::<2>().0 {
        match string_value(&pair[0]).as_deref() {
            Some("identifier") => attribute.name = string_value(&pair[1]),
            Some("type") => attribute.field_type = string_value(&pair[1]),
            Some("dim") => attribute.dim = number_value(&pair[1]),
            Some("data_type") => attribute.data_type = string_value(&pair[1]),
            Some("distance_metric") => attribute.distance_metric = string_value(&pair[1]),
            _ => {}
        }
    }
    Some(attribute)
}

fn schema_compatible(info: &redis::Value, dims: usize) -> bool {
    let redis::Value::Array(entries) = info else {
        return false;
    };
    let attributes = entries
        .as_chunks::<2>()
        .0
        .iter()
        .find(|pair| string_value(&pair[0]).as_deref() == Some("attributes"))
        .map(|pair| &pair[1]);
    let Some(redis::Value::Array(attributes)) = attributes else {
        return false;
    };
    let fields = attributes.iter().filter_map(attribute).collect::<Vec<_>>();
    let has_field = |name: &str, field_type: &str| {
        fields.iter().any(|field| {
            field.name.as_deref() == Some(name) && field.field_type.as_deref() == Some(field_type)
        })
    };
    has_field("prompt", "TEXT")
        && has_field("response", "TEXT")
        && has_field("inserted_at", "NUMERIC")
        && has_field("updated_at", "NUMERIC")
        && has_field(CACHE_KEY_FIELD, "TAG")
        && fields.iter().any(|field| {
            field.name.as_deref() == Some(VECTOR_FIELD)
                && field.field_type.as_deref() == Some("VECTOR")
                && field.dim == Some(dims as f64)
                && field
                    .data_type
                    .as_deref()
                    .is_some_and(|data| data.eq_ignore_ascii_case("float32"))
                && field
                    .distance_metric
                    .as_deref()
                    .is_some_and(|metric| metric.eq_ignore_ascii_case("cosine"))
        })
}
