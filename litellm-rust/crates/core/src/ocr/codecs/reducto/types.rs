use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(crate) struct ReductoV3Params {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub formatting: Option<Map<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retrieval: Option<Map<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub settings: Option<Map<String, Value>>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(crate) struct ReductoLegacyParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enhance: Option<Map<String, Value>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct ReductoV3Request {
    pub input: String,
    #[serde(flatten)]
    pub params: ReductoV3Params,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct ReductoLegacyRequest {
    pub document_url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<ReductoLegacyParams>,
}

#[derive(Deserialize)]
pub(crate) struct ReductoUploadResponse {
    pub file_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct ReductoResponse {
    #[serde(default, deserialize_with = "present_nullable")]
    pub result: Option<Option<ReductoResult>>,
    pub usage: Option<ReductoUsage>,
    #[serde(default)]
    pub chunks: Option<Vec<ReductoChunk>>,
}

fn present_nullable<'de, D: Deserializer<'de>, T: Deserialize<'de>>(
    deserializer: D,
) -> Result<Option<Option<T>>, D::Error> {
    Option::<T>::deserialize(deserializer).map(Some)
}

#[derive(Clone, Debug, Default, Deserialize)]
pub(crate) struct ReductoResult {
    pub chunks: Option<Vec<ReductoChunk>>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub(crate) struct ReductoUsage {
    #[serde(default, deserialize_with = "optional_i64")]
    pub num_pages: Option<i64>,
    #[serde(default, deserialize_with = "optional_f64")]
    pub credits: Option<f64>,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct ReductoChunk {
    pub content: Option<String>,
    pub blocks: Option<Vec<ReductoBlock>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct ReductoBlock {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bbox: Option<ReductoBoundingBox>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct ReductoBoundingBox {
    #[serde(default, deserialize_with = "optional_i64")]
    pub page: Option<i64>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

fn optional_i64<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<i64>, D::Error> {
    match Option::<Value>::deserialize(deserializer)? {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(number)) => number
            .as_i64()
            .or_else(|| number.as_f64().and_then(checked_truncated_i64))
            .map(Some)
            .ok_or_else(|| serde::de::Error::custom("expected an integer")),
        Some(Value::String(value)) => value
            .trim()
            .parse::<i64>()
            .map(Some)
            .map_err(|_| serde::de::Error::custom("expected an integer")),
        Some(Value::Bool(value)) => Ok(Some(i64::from(value))),
        Some(_) => Ok(None),
    }
}

fn optional_f64<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<f64>, D::Error> {
    match Option::<Value>::deserialize(deserializer)? {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(number)) => number
            .as_f64()
            .map(Some)
            .ok_or_else(|| serde::de::Error::custom("expected a number")),
        Some(Value::String(value)) => value
            .trim()
            .parse::<f64>()
            .map(Some)
            .map_err(|_| serde::de::Error::custom("expected a number")),
        Some(_) => Ok(None),
    }
}

fn checked_truncated_i64(value: f64) -> Option<i64> {
    (value.is_finite() && value >= i64::MIN as f64 && value <= i64::MAX as f64)
        .then(|| value.trunc() as i64)
}
