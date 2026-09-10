use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};

use crate::ocr::types::OcrResponseFormat;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub(crate) enum PagesInput {
    ZeroBasedIndices(Vec<i64>),
    NativeTokens(Vec<String>),
    NativeRange(String),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub(crate) enum FeaturesInput {
    Names(Vec<String>),
    CommaSeparated(String),
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct DocumentIntelligenceInputParams {
    pub pages: Option<PagesInput>,
    pub features: Option<FeaturesInput>,
    pub req_format: Option<OcrResponseFormat>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct DocumentIntelligenceParams {
    pub pages: Option<String>,
    pub features: Option<String>,
    #[serde(rename = "req_format")]
    pub response_format: OcrResponseFormat,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) enum DocumentIntelligenceRequest {
    #[serde(rename = "urlSource")]
    UrlSource(String),
    #[serde(rename = "base64Source")]
    Base64Source(String),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum OperationStatus {
    Succeeded,
    Running,
    NotStarted,
    Failed,
    Unknown(String),
}

impl<'de> Deserialize<'de> for OperationStatus {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(match String::deserialize(deserializer)?.as_str() {
            "succeeded" => Self::Succeeded,
            "running" => Self::Running,
            "notStarted" => Self::NotStarted,
            "failed" => Self::Failed,
            value => Self::Unknown(value.to_string()),
        })
    }
}

impl std::fmt::Display for OperationStatus {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Succeeded => "succeeded",
            Self::Running => "running",
            Self::NotStarted => "notStarted",
            Self::Failed => "failed",
            Self::Unknown(value) => value,
        })
    }
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct AzureDocumentIntelligenceOperation {
    pub status: Option<OperationStatus>,
    #[serde(rename = "analyzeResult")]
    pub analyze_result: Option<AzureDocumentIntelligenceAnalyzeResult>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub(crate) struct AzureDocumentIntelligenceAnalyzeResult {
    pub content: Option<String>,
    #[serde(default)]
    pub pages: Vec<AzureDocumentIntelligencePage>,
    pub tables: Option<Vec<Map<String, Value>>>,
    #[serde(rename = "keyValuePairs")]
    pub key_value_pairs: Option<Vec<Map<String, Value>>>,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct AzureDocumentIntelligencePage {
    #[serde(rename = "pageNumber", default, deserialize_with = "optional_i64")]
    pub page_number: Option<i64>,
    #[serde(default, deserialize_with = "optional_f64")]
    pub width: Option<f64>,
    #[serde(default, deserialize_with = "optional_f64")]
    pub height: Option<f64>,
    pub unit: Option<String>,
    #[serde(default)]
    pub lines: Vec<AzureDocumentIntelligenceLine>,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct AzureDocumentIntelligenceLine {
    pub content: Option<String>,
}

fn optional_i64<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<i64>, D::Error> {
    match Option::<Value>::deserialize(deserializer)? {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(number)) => number
            .as_i64()
            .map(Some)
            .ok_or_else(|| serde::de::Error::custom("expected an integer")),
        Some(Value::String(value)) => value
            .parse::<i64>()
            .map(Some)
            .map_err(|_| serde::de::Error::custom("expected an integer")),
        Some(_) => Err(serde::de::Error::custom("expected an integer")),
    }
}

fn optional_f64<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<f64>, D::Error> {
    match Option::<Value>::deserialize(deserializer)? {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(number)) => number
            .as_f64()
            .filter(|value| value.is_finite())
            .map(Some)
            .ok_or_else(|| serde::de::Error::custom("expected a finite number")),
        Some(Value::String(value)) => value
            .parse::<f64>()
            .ok()
            .filter(|value| value.is_finite())
            .map(Some)
            .ok_or_else(|| serde::de::Error::custom("expected a finite number")),
        Some(_) => Err(serde::de::Error::custom("expected a number")),
    }
}
