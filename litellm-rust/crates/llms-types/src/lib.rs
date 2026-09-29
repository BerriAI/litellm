macro_rules_attribute::attribute_alias! {
    #[apply(wire_type)] =
        #[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
        #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))];
}

pub mod endpoint;
pub mod formats;
pub mod headers;
pub mod providers;
pub mod reasoning;
pub mod recognized;

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Value, json};

    #[macro_rules_attribute::apply(wire_type)]
    #[derive(Default)]
    struct Payload<T> {
        #[serde(rename = "wire_value")]
        value: T,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        optional: Option<String>,
    }

    #[macro_rules_attribute::apply(wire_type)]
    #[derive(Copy, Eq)]
    #[serde(rename_all = "snake_case")]
    enum Status {
        InProgress,
        Complete,
    }

    #[rstest]
    #[case::missing(json!({"wire_value": "in_progress"}))]
    #[case::present(json!({"wire_value": "complete", "optional": "text"}))]
    fn wire_type_preserves_serde_attributes_and_generic_values(#[case] wire: Value) {
        let payload: Payload<Status> = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(serde_json::to_value(payload.clone()).unwrap(), wire);
        assert_eq!(
            serde_json::from_value::<Payload<Status>>(wire).unwrap(),
            payload
        );
        assert_eq!(
            serde_json::to_value(Payload::<String>::default()).unwrap(),
            json!({"wire_value": ""})
        );
    }

    #[cfg(feature = "schema")]
    #[rstest]
    fn schema_uses_the_same_field_and_enum_names_as_serialization() {
        let schema = serde_json::to_value(schemars::schema_for!(Payload<Status>)).unwrap();
        let wire = serde_json::to_value(Payload {
            value: Status::InProgress,
            optional: Some("text".into()),
        })
        .unwrap();
        let properties = schema["properties"].as_object().unwrap();
        assert_eq!(
            properties.keys().collect::<Vec<_>>(),
            wire.as_object().unwrap().keys().collect::<Vec<_>>()
        );
        let status_schema = serde_json::to_value(schemars::schema_for!(Status)).unwrap();
        let statuses = [Status::InProgress, Status::Complete]
            .map(|value| serde_json::to_value(value).unwrap());
        assert_eq!(status_schema["enum"], json!(statuses));
    }
}
