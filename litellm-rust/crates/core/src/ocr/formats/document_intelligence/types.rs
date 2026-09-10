use crate::ocr::types::OcrRequestFormat;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PagesInput {
    ZeroBasedIndices(Vec<i64>),
    NativeTokens(Vec<String>),
    NativeRange(String),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FeaturesInput {
    Names(Vec<String>),
    CommaSeparated(String),
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DocumentIntelligenceInputParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pages: Option<PagesInput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub features: Option<FeaturesInput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub req_format: Option<OcrRequestFormat>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NormalizedPages(pub(crate) String);
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NormalizedFeatures(pub(crate) String);

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DocumentIntelligenceParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) pages: Option<NormalizedPages>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) features: Option<NormalizedFeatures>,
    #[serde(rename = "req_format")]
    pub(crate) request_format: OcrRequestFormat,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum DocumentIntelligenceRequest {
    #[serde(rename = "urlSource")]
    UrlSource(String),
    #[serde(rename = "base64Source")]
    Base64Source(String),
}

#[derive(Clone, Debug, PartialEq)]
pub enum OperationStatus {
    Succeeded,
    Running,
    NotStarted,
    Failed,
    Unknown(String),
}

impl<'de> Deserialize<'de> for OperationStatus {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(match String::deserialize(deserializer)?.as_str() {
            "succeeded" => Self::Succeeded,
            "running" => Self::Running,
            "notStarted" => Self::NotStarted,
            "failed" => Self::Failed,
            other => Self::Unknown(other.to_string()),
        })
    }
}

impl std::fmt::Display for OperationStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Succeeded => "succeeded",
            Self::Running => "running",
            Self::NotStarted => "notStarted",
            Self::Failed => "failed",
            Self::Unknown(s) => s,
        })
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct AzureDocumentIntelligenceOperation {
    pub status: Option<OperationStatus>,
    #[serde(rename = "analyzeResult")]
    pub analyze_result: Option<AzureDocumentIntelligenceAnalyzeResult>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct AzureDocumentIntelligenceAnalyzeResult {
    pub content: Option<String>,
    #[serde(default)]
    pub pages: Vec<AzureDocumentIntelligencePage>,
    pub tables: Option<Vec<Map<String, Value>>>,
    #[serde(rename = "keyValuePairs")]
    pub key_value_pairs: Option<Vec<Map<String, Value>>>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct AzureDocumentIntelligencePage {
    #[serde(
        rename = "pageNumber",
        default,
        deserialize_with = "crate::ocr::wire::optional_i64"
    )]
    pub page_number: Option<i64>,
    #[serde(default, deserialize_with = "crate::ocr::wire::optional_f64")]
    pub width: Option<f64>,
    #[serde(default, deserialize_with = "crate::ocr::wire::optional_f64")]
    pub height: Option<f64>,
    pub unit: Option<String>,
    #[serde(default)]
    pub lines: Vec<AzureDocumentIntelligenceLine>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct AzureDocumentIntelligenceLine {
    pub content: Option<String>,
}
