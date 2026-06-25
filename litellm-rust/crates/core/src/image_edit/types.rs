use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImageEditInputFile {
    pub filename: String,
    pub content_type: String,
    pub data_base64: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImageEditMultipartPart {
    pub field_name: String,
    pub filename: String,
    pub content_type: String,
    pub data_base64: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ImageEditRequestFormat {
    Multipart,
    Json,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImageEditRequestData {
    pub data: Map<String, Value>,
    pub files: Vec<ImageEditMultipartPart>,
    pub format: ImageEditRequestFormat,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImageEditResponseData {
    pub data: Map<String, Value>,
}

impl ImageEditResponseData {
    pub fn into_json(self) -> Value {
        Value::Object(self.data)
    }
}
