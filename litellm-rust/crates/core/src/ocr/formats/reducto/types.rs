use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ReductoV3Params {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub formatting: Option<Map<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retrieval: Option<Map<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub settings: Option<Map<String, Value>>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ReductoLegacyParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enhance: Option<Map<String, Value>>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReductoV3Request {
    pub input: String,
    #[serde(flatten)]
    pub params: ReductoV3Params,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReductoLegacyRequest {
    pub document_url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<ReductoLegacyParams>,
}
pub struct ReductoFileId(pub(crate) String);

impl TryFrom<ReductoUploadResponse> for ReductoFileId {
    type Error = crate::ocr::error::OcrError;

    fn try_from(response: ReductoUploadResponse) -> Result<Self, Self::Error> {
        if response.file_id.is_empty() {
            return Err(crate::ocr::error::OcrResponseError::ResponseField {
                path: "file_id".into(),
            }
            .into());
        }
        Ok(Self(response.file_id))
    }
}

#[derive(Deserialize)]
pub struct ReductoUploadResponse {
    pub file_id: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ReductoResponse {
    #[serde(default, deserialize_with = "present_nullable")]
    pub result: Option<Option<ReductoResult>>,
    pub usage: Option<ReductoUsage>,
    #[serde(default)]
    pub chunks: Option<Vec<ReductoChunk>>,
}

fn present_nullable<'de, D: serde::Deserializer<'de>, T: Deserialize<'de>>(
    deserializer: D,
) -> Result<Option<Option<T>>, D::Error> {
    Option::<T>::deserialize(deserializer).map(Some)
}
#[derive(Clone, Debug, Default, Deserialize)]
pub struct ReductoResult {
    pub chunks: Option<Vec<ReductoChunk>>,
}
#[derive(Clone, Debug, Default, Deserialize)]
pub struct ReductoUsage {
    #[serde(default, deserialize_with = "crate::ocr::wire::optional_i64")]
    pub num_pages: Option<i64>,
    #[serde(default, deserialize_with = "crate::ocr::wire::optional_f64")]
    pub credits: Option<f64>,
}
#[derive(Clone, Debug, Deserialize)]
pub struct ReductoChunk {
    pub content: Option<String>,
    pub blocks: Option<Vec<ReductoBlock>>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReductoBlock {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bbox: Option<ReductoBoundingBox>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReductoBoundingBox {
    #[serde(default, deserialize_with = "reducto_page")]
    pub page: Option<i64>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

fn reducto_page<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<i64>, D::Error> {
    let value = Value::deserialize(deserializer)?;
    Ok(match value {
        Value::Number(number) => number.as_i64().or_else(|| {
            number
                .as_f64()
                .and_then(crate::ocr::wire::checked_truncated_i64)
        }),
        Value::String(value) => value.trim().parse().ok(),
        Value::Bool(value) => Some(i64::from(value)),
        _ => None,
    })
}
