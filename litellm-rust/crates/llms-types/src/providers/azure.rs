use litellm_python_compat::serde_compat::{FiniteF64, LaxI64};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};
use serde_with::serde_as;

#[macro_rules_attribute::apply(wire_type)]
#[serde(untagged)]
pub enum DocumentIntelligenceRequest {
    UrlSource {
        #[serde(rename = "urlSource")]
        url_source: String,
    },
    Base64Source {
        #[serde(rename = "base64Source")]
        base64_source: String,
    },
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "schema", schemars(with = "String"))]
pub enum OperationStatus {
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

#[macro_rules_attribute::apply(wire_type)]
pub struct AzureDocumentIntelligenceOperation {
    pub status: Option<OperationStatus>,
    #[serde(rename = "analyzeResult")]
    pub analyze_result: Option<AzureDocumentIntelligenceAnalyzeResult>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct AzureDocumentIntelligenceAnalyzeResult {
    pub content: Option<String>,
    #[serde(default)]
    pub pages: Vec<AzureDocumentIntelligencePage>,
    pub tables: Option<Vec<Map<String, Value>>>,
    #[serde(rename = "keyValuePairs")]
    pub key_value_pairs: Option<Vec<Map<String, Value>>>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[serde_as]
#[macro_rules_attribute::apply(wire_type)]
pub struct AzureDocumentIntelligencePage {
    #[serde(rename = "pageNumber")]
    #[serde_as(deserialize_as = "Option<LaxI64>")]
    pub page_number: Option<i64>,
    #[serde_as(deserialize_as = "Option<FiniteF64>")]
    pub width: Option<f64>,
    #[serde_as(deserialize_as = "Option<FiniteF64>")]
    pub height: Option<f64>,
    pub unit: Option<String>,
    #[serde(default)]
    pub lines: Vec<AzureDocumentIntelligenceLine>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
pub struct AzureDocumentIntelligenceLine {
    pub content: Option<String>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

impl Serialize for OperationStatus {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}
#[cfg(test)]
mod tests {
    use super::{AzureDocumentIntelligenceOperation, DocumentIntelligenceRequest, OperationStatus};
    use rstest::rstest;
    use serde_json::{Value, json};

    #[rstest]
    #[case::url(json!({"urlSource":"https://example.com/document.pdf"}))]
    #[case::base64(json!({"base64Source":"AA=="}))]
    fn document_sources_round_trip(#[case] wire: Value) {
        let request: DocumentIntelligenceRequest = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(serde_json::to_value(request).unwrap(), wire);
    }

    #[rstest]
    #[case::succeeded(OperationStatus::Succeeded)]
    #[case::running(OperationStatus::Running)]
    #[case::not_started(OperationStatus::NotStarted)]
    #[case::failed(OperationStatus::Failed)]
    #[case::unknown(OperationStatus::Unknown("future-status".into()))]
    fn operation_status_preserves_wire_value(#[case] status: OperationStatus) {
        let wire = serde_json::to_value(&status).unwrap();
        assert_eq!(wire, json!(status.to_string()));
        assert_eq!(
            serde_json::from_value::<OperationStatus>(wire).unwrap(),
            status
        );
    }

    #[rstest]
    fn operation_preserves_nested_extensions_and_numeric_coercion() {
        let wire = json!({
            "status":"succeeded", "createdDateTime":"timestamp", "analyzeResult":{
                "content":"text", "modelId":"layout", "pages":[{
                    "pageNumber":"2", "width":"8.5", "height":11, "unit":"inch",
                    "words":[{"content":"text","confidence":0.99}],
                    "lines":[{"content":"text","polygon":[0,0,1,1]}]
                }], "tables":[{"rowCount":1}], "keyValuePairs":[{"key":{"content":"key"}}]
            }
        });
        let operation: AzureDocumentIntelligenceOperation =
            serde_json::from_value(wire.clone()).unwrap();
        let page = &operation.analyze_result.as_ref().unwrap().pages[0];
        assert_eq!(page.page_number, Some(2));
        assert_eq!(page.width, Some(8.5));
        let encoded = serde_json::to_value(&operation).unwrap();
        assert_eq!(encoded["createdDateTime"], wire["createdDateTime"]);
        assert_eq!(
            encoded["analyzeResult"]["modelId"],
            wire["analyzeResult"]["modelId"]
        );
        assert_eq!(
            encoded["analyzeResult"]["pages"][0]["words"],
            wire["analyzeResult"]["pages"][0]["words"]
        );
        assert_eq!(
            encoded["analyzeResult"]["pages"][0]["lines"],
            wire["analyzeResult"]["pages"][0]["lines"]
        );
        assert_eq!(
            encoded["analyzeResult"]["keyValuePairs"],
            wire["analyzeResult"]["keyValuePairs"]
        );
        assert_eq!(
            serde_json::from_value::<AzureDocumentIntelligenceOperation>(encoded).unwrap(),
            operation
        );
    }

    #[rstest]
    #[case::status(json!({"status":false}))]
    #[case::page(json!({"analyzeResult":{"pages":[{"pageNumber":1.5}]}}))]
    #[case::line(json!({"analyzeResult":{"pages":[{"lines":[{"content":42}]}]}}))]
    fn operation_rejects_malformed_typed_fields(#[case] wire: Value) {
        assert!(serde_json::from_value::<AzureDocumentIntelligenceOperation>(wire).is_err());
    }
}
