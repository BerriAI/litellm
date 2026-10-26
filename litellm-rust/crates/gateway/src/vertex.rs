use litellm_auth_types::{VertexKeysPartial, VertexParams};
use litellm_config::{LiteLlmSettings, LitellmParams, Model};
use litellm_layered::merge;

pub(crate) fn with_written_settings(models: &[Model], settings: &LiteLlmSettings) -> Vec<Model> {
    let file = written(settings);
    models
        .iter()
        .map(|model| {
            let keys = merge([model.litellm_params.vertex.keys(), file.clone()]);
            Model {
                litellm_params: LitellmParams {
                    vertex: VertexParams {
                        vertex_project: keys.project,
                        vertex_location: keys.location,
                        ..model.litellm_params.vertex.clone()
                    },
                    ..model.litellm_params.clone()
                },
                ..model.clone()
            }
        })
        .collect()
}

fn written(settings: &LiteLlmSettings) -> VertexKeysPartial {
    VertexKeysPartial {
        credentials: None,
        project: settings.vertex_project.clone(),
        location: settings.vertex_location.clone(),
    }
}

#[cfg(test)]
mod tests {
    use litellm_auth_types::VertexKeys;
    use litellm_config::Config;
    use litellm_http::HttpKeys;
    use rstest::rstest;
    use serde_json::{Map, Value};

    use super::*;

    fn models(yaml: &str) -> Vec<Model> {
        let config = Config::from_yaml(yaml).unwrap();
        with_written_settings(&config.model_list, &config.litellm_settings)
    }

    #[rstest]
    #[case::file_fills_an_unset_deployment(
        "model_list:\n  - model_name: m\n    litellm_params:\n      model: vertex_ai/claude\nlitellm_settings:\n  vertex_project: file-project\n",
        Some("file-project")
    )]
    #[case::the_deployment_wins(
        "model_list:\n  - model_name: m\n    litellm_params:\n      model: vertex_ai/claude\n      vertex_ai_project: deployment\nlitellm_settings:\n  vertex_project: file-project\n",
        Some("deployment")
    )]
    #[case::a_blank_deployment_value_takes_the_file(
        "model_list:\n  - model_name: m\n    litellm_params:\n      model: vertex_ai/claude\n      vertex_project: ' '\nlitellm_settings:\n  vertex_project: file-project\n",
        Some("file-project")
    )]
    #[case::neither(
        "model_list:\n  - model_name: m\n    litellm_params:\n      model: vertex_ai/claude\n",
        None
    )]
    fn deployment_params_are_stacked_above_the_config_file(
        #[case] yaml: &str,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(
            models(yaml)[0].litellm_params.vertex.project().as_deref(),
            expected
        );
    }

    #[rstest]
    fn every_litellm_setting_a_group_declares_is_a_typed_config_field() {
        let names = HttpKeys::LITELLM_SETTINGS
            .iter()
            .chain(VertexKeys::LITELLM_SETTINGS)
            .map(|(name, _)| ((*name).to_owned(), Value::Null))
            .collect::<Map<_, _>>();
        let settings: LiteLlmSettings = serde_json::from_value(Value::Object(names)).unwrap();
        assert_eq!(
            settings.additional_fields.keys().collect::<Vec<_>>(),
            Vec::<&String>::new()
        );
    }
}
