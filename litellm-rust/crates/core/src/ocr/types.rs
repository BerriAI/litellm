use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Field<T> {
    Absent,
    Null,
    Value(T),
}

impl<T> Field<T> {
    pub fn as_ref(&self) -> Field<&T> {
        match self {
            Self::Absent => Field::Absent,
            Self::Null => Field::Null,
            Self::Value(value) => Field::Value(value),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OcrDialectId {
    Mistral,
    AzureFoundryMistral,
    AzureDocumentIntelligence,
    VertexMistral,
    VertexDeepSeek,
    ReductoV3,
    ReductoLegacy,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OcrRequestData {
    pub data: Value,
    pub files: Option<Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OcrResponseData {
    pub pages: Vec<Value>,
    pub model: String,
    pub document_annotation: Option<Value>,
    pub usage_info: Option<Value>,
    pub object: String,
}

impl OcrResponseData {
    pub fn into_json(self) -> Value {
        serde_json::json!({
            "pages": self.pages,
            "model": self.model,
            "document_annotation": self.document_annotation,
            "usage_info": self.usage_info,
            "object": self.object,
        })
    }
}
