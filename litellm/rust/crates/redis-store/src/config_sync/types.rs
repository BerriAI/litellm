use serde::Deserialize;

/// One message on the config change channel: a write to the table or the settings row
/// `object_type` names, with the model ids a model create, update or upsert wrote.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct ConfigChange {
    pub object_type: String,
    #[serde(default)]
    pub model_ids: Vec<String>,
}

impl ConfigChange {
    /// `parse_config_change`: anything but a JSON object with a string `object_type`, and a list of
    /// strings as `model_ids` when it is there, is ignored. Other fields are ignored.
    pub fn decode(payload: &[u8]) -> Option<Self> {
        serde_json::from_slice(payload).ok()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde::Deserialize;

    use super::ConfigChange;

    /// What `tests/unit/proxy/common_utils/test_config_sync_pubsub.py` checks Python's publisher
    /// writes and its subscriber reads, so a change on either side fails a test.
    #[derive(Deserialize)]
    struct Fixture {
        published: Vec<Case>,
        accepted: Vec<Case>,
        ignored: Vec<String>,
    }

    #[derive(Deserialize)]
    struct Case {
        payload: String,
        object_type: String,
        model_ids: Vec<String>,
    }

    fn fixture() -> Fixture {
        serde_json::from_str(include_str!("../../fixtures/config_change.json")).unwrap()
    }

    fn heard(index: usize) -> Case {
        let Fixture {
            published,
            accepted,
            ..
        } = fixture();
        published.into_iter().chain(accepted).nth(index).unwrap()
    }

    #[rstest]
    fn every_case_of_the_fixture_is_checked() {
        let fixture = fixture();
        assert_eq!(
            (
                fixture.published.len() + fixture.accepted.len(),
                fixture.ignored.len()
            ),
            (6, 7)
        );
    }

    #[rstest]
    #[case::a_written_model(0)]
    #[case::a_deleted_model(1)]
    #[case::a_credential(2)]
    #[case::a_settings_row(3)]
    #[case::several_models_and_unicode(4)]
    #[case::a_field_python_does_not_know(5)]
    fn a_message_python_reads_decodes_to_what_it_names(#[case] index: usize) {
        let case = heard(index);
        assert_eq!(
            ConfigChange::decode(case.payload.as_bytes()),
            Some(ConfigChange {
                object_type: case.object_type,
                model_ids: case.model_ids,
            })
        );
    }

    #[rstest]
    #[case::not_json(0)]
    #[case::not_an_object(1)]
    #[case::no_object_type(2)]
    #[case::an_object_type_that_is_not_a_string(3)]
    #[case::model_ids_that_are_a_string(4)]
    #[case::model_ids_that_are_null(5)]
    #[case::model_ids_that_are_not_strings(6)]
    fn a_message_python_ignores_is_ignored(#[case] index: usize) {
        let payload = fixture().ignored.swap_remove(index);
        assert_eq!(ConfigChange::decode(payload.as_bytes()), None);
    }
}
