use serde::{Deserialize, Serialize};
use serde_json::Value;

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
    pub content: Option<Value>,
    pub tables: Option<Value>,
    pub key_value_pairs: Option<Value>,
    pub provider_native_response: Option<Value>,
    pub object: String,
}

impl OcrResponseData {
    pub fn into_json(self) -> Value {
        self.provider_native_response.unwrap_or_else(|| {
            serde_json::json!({
                "pages": self.pages,
                "model": self.model,
                "document_annotation": self.document_annotation,
                "usage_info": self.usage_info,
                "content": self.content,
                "tables": self.tables,
                "keyValuePairs": self.key_value_pairs,
                "object": self.object,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::OcrResponseData;

    #[test]
    fn into_json_returns_provider_native_response_when_present() {
        let native_response =
            json!({"status": "succeeded", "paragraphs": [{"content": "Invoice"}]});
        let response = OcrResponseData {
            pages: vec![],
            model: "prebuilt-layout".to_string(),
            document_annotation: None,
            usage_info: None,
            content: None,
            tables: None,
            key_value_pairs: None,
            provider_native_response: Some(native_response.clone()),
            object: "ocr".to_string(),
        };

        assert_eq!(response.into_json(), native_response);
    }
}
