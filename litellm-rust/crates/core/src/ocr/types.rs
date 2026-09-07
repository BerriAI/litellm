use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub struct OcrRequest {
    pub model: String,
    pub custom_llm_provider: Option<String>,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub extra_headers: Vec<(String, String)>,
    pub timeout_seconds: f64,
    pub request_format: Option<String>,
    pub document: OcrDocument,
    pub azure_ad_token: Option<String>,
    pub vertex_project: Option<String>,
    pub vertex_location: Option<String>,
    pub stream: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OcrDocument {
    DocumentUrl {
        document_url: String,
    },
    ImageUrl {
        image_url: String,
    },
    File,
    #[serde(other)]
    Unsupported,
}

impl OcrDocument {
    pub fn validate(&self, requires_data_uri: bool) -> Result<(), crate::Error> {
        let url = match self {
            Self::DocumentUrl { document_url } => document_url,
            Self::ImageUrl { image_url } => image_url,
            Self::File => return Err(crate::Error::Unsupported("OCR file document preparation")),
            Self::Unsupported => return Err(crate::Error::Unsupported("OCR document type")),
        };
        let parsed = reqwest::Url::parse(url)
            .map_err(|_| crate::Error::Unsupported("OCR local or non-HTTP document preparation"))?;
        match parsed.scheme() {
            "http" | "https" if requires_data_uri => Err(crate::Error::Unsupported(
                "OCR HTTP document URL to data URI conversion",
            )),
            "http" | "https" | "data" => Ok(()),
            _ => Err(crate::Error::Unsupported(
                "OCR local or non-HTTP document preparation",
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OcrDocumentProjection {
    RetainedDocument,
    ShallowCopyDocument,
    Transformed,
}

pub struct PreparedOcr {
    pub model: String,
    pub custom_llm_provider: String,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Map<String, Value>,
    pub document_projection: OcrDocumentProjection,
    pub parameter_fields: &'static [&'static str],
    pub timeout_seconds: f64,
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
    pub extra_fields: Map<String, Value>,
    pub provider_native_response: Option<Value>,
}

impl OcrResponseData {
    pub fn into_json(self) -> Value {
        let mut response = serde_json::json!({
            "pages": self.pages,
            "model": self.model,
            "document_annotation": self.document_annotation,
            "usage_info": self.usage_info,
            "object": self.object,
        });
        if let Value::Object(object) = &mut response {
            object.extend(self.extra_fields);
            if let Some(native_response) = self.provider_native_response {
                object.insert("provider_native_response".to_string(), native_response);
            }
        }
        response
    }
}
