use serde_json::{Map, Value};

use crate::recognized::Recognized;

#[macro_rules_attribute::apply(wire_type)]
#[derive(Copy, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum MistralFilePurpose {
    FineTune,
    Batch,
    Ocr,
}

#[macro_rules_attribute::apply(wire_type)]
pub struct MistralFileUploadFields {
    pub purpose: MistralFilePurpose,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub expiry: Option<Option<u64>>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
pub struct MistralFile {
    pub id: String,
    pub object: String,
    pub bytes: u64,
    pub created_at: i64,
    pub filename: String,
    pub purpose: Recognized<MistralFilePurpose>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub sample_type: Option<Option<String>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub source: Option<Option<String>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub num_lines: Option<Option<u64>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub mimetype: Option<Option<String>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub signature: Option<Option<String>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub expires_at: Option<Option<i64>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub visibility: Option<Option<String>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub deleted: Option<Option<bool>>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
pub struct MistralFileList {
    pub data: Vec<MistralFile>,
    pub object: String,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub total: Option<Option<u64>>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
pub struct MistralFileDeleted {
    pub id: String,
    pub object: String,
    pub deleted: bool,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
pub struct MistralSignedUrl {
    pub url: String,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::{
        MistralFile, MistralFileDeleted, MistralFileList, MistralFilePurpose,
        MistralFileUploadFields,
    };
    use crate::recognized::Recognized;
    use rstest::rstest;
    use serde_json::{Value, json};

    #[rstest]
    #[case::fine_tune(MistralFilePurpose::FineTune)]
    #[case::batch(MistralFilePurpose::Batch)]
    #[case::ocr(MistralFilePurpose::Ocr)]
    fn upload_fields_round_trip(#[case] purpose: MistralFilePurpose) {
        let wire = json!({"purpose":purpose,"expiry":24,"future_field":null});
        let fields: MistralFileUploadFields = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(fields.purpose, purpose);
        assert_eq!(serde_json::to_value(fields).unwrap(), wire);
    }

    #[rstest]
    #[case::known(json!("ocr"), true)]
    #[case::future(json!("future-purpose"), false)]
    fn file_list_preserves_purpose_and_metadata(#[case] purpose: Value, #[case] known: bool) {
        let wire = json!({"object":"list","total":1,"data":[{
            "id":"file-id","object":"file","bytes":10,"created_at":1,"filename":"document.pdf",
            "purpose":purpose,"sample_type":"batch_request","source":"upload","visibility":"workspace",
            "num_lines":null,"mimetype":"application/pdf","signature":null,"expires_at":null,
            "deleted":false,"future_field":{"nested":null}
        }]});
        let files: MistralFileList = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(files.data[0].purpose.known().is_some(), known);
        assert_eq!(serde_json::to_value(files).unwrap(), wire);
    }

    #[rstest]
    fn deletion_and_unknown_file_purpose_are_data_only() {
        let wire = json!({"id":"file-id","object":"file","deleted":true,"future_field":null});
        let deleted: MistralFileDeleted = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(serde_json::to_value(deleted).unwrap(), wire);
        let file: MistralFile = serde_json::from_value(json!({
            "id":"file-id","object":"file","bytes":10,"created_at":1,"filename":"document.pdf",
            "purpose":"future-purpose"
        }))
        .unwrap();
        assert_eq!(
            file.purpose,
            Recognized::Unrecognized(json!("future-purpose"))
        );
    }

    #[rstest]
    #[case::purpose(json!({"purpose":"unsupported"}))]
    #[case::expiry(json!({"purpose":"ocr","expiry":-1}))]
    fn upload_fields_reject_invalid_values(#[case] wire: Value) {
        assert!(serde_json::from_value::<MistralFileUploadFields>(wire).is_err());
    }
}
