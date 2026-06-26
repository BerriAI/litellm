use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

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
            "_hidden_params": Value::Object(litellm_rust_hidden_params()),
        })
    }
}

pub fn litellm_rust_hidden_params() -> Map<String, Value> {
    let mut hidden_params = Map::new();
    hidden_params.insert("litellm_rust".to_string(), json!(true));
    hidden_params
}

#[cfg(test)]
mod tests {
    use super::OcrResponseData;

    #[test]
    fn ocr_response_data_marks_litellm_rust_hidden_params() {
        let response = OcrResponseData {
            pages: vec![],
            model: "mistral-ocr-latest".to_string(),
            document_annotation: None,
            usage_info: None,
            object: "ocr".to_string(),
        }
        .into_json();

        assert_eq!(response["_hidden_params"]["litellm_rust"], true);
    }
}
