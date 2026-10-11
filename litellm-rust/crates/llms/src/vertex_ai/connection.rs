use litellm_auth::VertexParams;
use litellm_auth_gcp::{VertexConfig, get_vertex_ai_project_from_credentials};
use litellm_core_utils::settings::{Selected, SettingSource, select_non_blank};
use litellm_llms_types::providers::vertex_ai::{DEFAULT_LOCATION, GLOBAL_API_BASE};

use super::common_utils::{get_vertex_base_url, invalid_location, validate_vertex_location};
use crate::base_llm::{
    connection::{InvalidApiBase, RuntimeDefaults, complete_url, validated_api_base},
    ocr::error::Error as OcrError,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum VertexConnectionError {
    #[error("vertex_location must be global or a lowercase alphanumeric token with hyphens")]
    InvalidLocation,
    #[error(transparent)]
    InvalidApiBase(#[from] InvalidApiBase),
}

impl From<VertexConnectionError> for crate::Error {
    fn from(error: VertexConnectionError) -> Self {
        match error {
            VertexConnectionError::InvalidLocation => invalid_location(),
            VertexConnectionError::InvalidApiBase(error) => error.into(),
        }
    }
}

impl From<VertexConnectionError> for OcrError {
    fn from(error: VertexConnectionError) -> Self {
        let path = match error {
            VertexConnectionError::InvalidLocation => "vertex_location",
            VertexConnectionError::InvalidApiBase(_) => "api_base",
        };
        Self::RequestField { path: path.into() }
    }
}

/// The project, location and host every Vertex AI operation sends to, chosen once per
/// deployment. Each comes from the explicit params, the host's runtime defaults, the
/// environment, then a derived or provider default: the project from the credentials, the
/// location `us-central1`, and the host from the location. `config` carries the same
/// project and location to authentication.
#[derive(Clone, Debug)]
pub struct VertexConnection {
    pub project: Option<Selected<String>>,
    pub location: Selected<String>,
    pub api_base: Selected<String>,
    pub config: VertexConfig,
}

impl VertexConnection {
    pub fn resolve(
        config: VertexConfig,
        api_base: Option<&str>,
        runtime_defaults: &RuntimeDefaults,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<Self, VertexConnectionError> {
        let location = select_non_blank(
            config.location(),
            runtime_defaults.vertex_location.as_deref(),
            env_lookup,
            VertexParams::LOCATION.env,
        )
        .unwrap_or_else(|| {
            Selected::new(DEFAULT_LOCATION.to_string(), SettingSource::ProviderDefault)
        });
        let derived_base = get_vertex_base_url(&location.value)
            .map_err(|_| VertexConnectionError::InvalidLocation)?;
        let project = select_non_blank(
            config.project_id(),
            runtime_defaults.vertex_project.as_deref(),
            env_lookup,
            VertexParams::PROJECT.env,
        )
        .or_else(|| {
            get_vertex_ai_project_from_credentials(&config, env_lookup)
                .map(|project| Selected::new(project, SettingSource::Derived))
        });
        let api_base = match select_non_blank(api_base, None, &|_| None, &[]) {
            Some(custom) => validated_api_base(custom)?,
            None => Selected::new(derived_base, SettingSource::Derived),
        };
        let config = config.with_project_and_location(
            project.as_ref().map(|project| project.value.clone()),
            location.value.clone(),
        );
        Ok(Self {
            project,
            location,
            api_base,
            config,
        })
    }

    pub fn location(&self) -> &str {
        &self.location.value
    }

    pub fn custom_api_base(&self) -> Option<&str> {
        (self.api_base.source != SettingSource::Derived).then_some(self.api_base.value.as_str())
    }

    /// `{api_base}/v1/projects/{project}/locations/{location}/{resource...}`.
    pub fn resource_url(
        &self,
        project: &str,
        resource: &[&str],
    ) -> Result<String, VertexConnectionError> {
        self.resource_url_on(&self.api_base.value, project, resource)
    }

    /// [`Self::resource_url`] for a publisher model Vertex serves only from the global host,
    /// whatever location the path names; a custom base still wins.
    pub fn global_resource_url(
        &self,
        project: &str,
        resource: &[&str],
    ) -> Result<String, VertexConnectionError> {
        let base = self.custom_api_base().unwrap_or(GLOBAL_API_BASE);
        self.resource_url_on(base, project, resource)
    }

    fn resource_url_on(
        &self,
        base: &str,
        project: &str,
        resource: &[&str],
    ) -> Result<String, VertexConnectionError> {
        validate_vertex_location(self.location())
            .map_err(|_| VertexConnectionError::InvalidLocation)?;
        let path = ["v1", "projects", project, "locations", self.location()]
            .into_iter()
            .chain(resource.iter().copied())
            .collect::<Vec<_>>();
        Ok(complete_url(base, &path)?)
    }
}

#[cfg(test)]
mod tests {
    use litellm_auth::{InputSource, SecretValue, Sourced};
    use rstest::rstest;

    use super::*;

    fn env(values: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |name| {
            values
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        }
    }

    fn explicit(project: Option<&str>, location: Option<&str>) -> VertexConfig {
        VertexConfig::new(None, project.map(Into::into), location.map(Into::into))
    }

    fn runtime(project: Option<&str>, location: Option<&str>) -> RuntimeDefaults {
        RuntimeDefaults {
            vertex_project: project.map(Into::into),
            vertex_location: location.map(Into::into),
            ..RuntimeDefaults::default()
        }
    }

    #[rstest]
    #[case::explicit_beats_runtime_and_environment(
        explicit(Some("arg"), Some("europe-west4")),
        runtime(Some("default"), Some("us-east5")),
        &[("VERTEXAI_PROJECT", "env"), ("VERTEXAI_LOCATION", "asia-east1")],
        ("arg", SettingSource::Explicit),
        ("europe-west4", SettingSource::Explicit)
    )]
    #[case::runtime_defaults_beat_the_environment(
        explicit(None, None),
        runtime(Some("default"), Some("us-east5")),
        &[("VERTEXAI_PROJECT", "env"), ("VERTEXAI_LOCATION", "asia-east1")],
        ("default", SettingSource::RuntimeDefault),
        ("us-east5", SettingSource::RuntimeDefault)
    )]
    #[case::environment_then_the_default_location(
        explicit(None, None),
        runtime(None, None),
        &[("VERTEXAI_PROJECT", "env")],
        ("env", SettingSource::Environment("VERTEXAI_PROJECT")),
        ("us-central1", SettingSource::ProviderDefault)
    )]
    #[case::second_location_alias(
        explicit(Some("arg"), None),
        runtime(None, None),
        &[("VERTEX_LOCATION", "europe-west1")],
        ("arg", SettingSource::Explicit),
        ("europe-west1", SettingSource::Environment("VERTEX_LOCATION"))
    )]
    fn project_and_location_follow_the_layer_precedence(
        #[case] config: VertexConfig,
        #[case] defaults: RuntimeDefaults,
        #[case] vars: &'static [(&'static str, &'static str)],
        #[case] project: (&str, SettingSource),
        #[case] location: (&str, SettingSource),
    ) {
        let connection = VertexConnection::resolve(config, None, &defaults, &env(vars)).unwrap();
        assert_eq!(
            connection.project,
            Some(Selected::new(project.0.to_string(), project.1))
        );
        assert_eq!(
            connection.location,
            Selected::new(location.0.to_string(), location.1)
        );
        assert_eq!(connection.config.project_id(), Some(project.0));
        assert_eq!(connection.config.location(), Some(location.0));
    }

    #[test]
    fn the_project_falls_back_to_the_credentials() {
        let config = VertexConfig::new(
            Some(Sourced::new(
                SecretValue::new(r#"{"type": "service_account", "project_id": "from-creds"}"#),
                InputSource::Deployment,
            )),
            None,
            Some("us-east5".into()),
        );
        let connection =
            VertexConnection::resolve(config, None, &RuntimeDefaults::default(), &env(&[]))
                .unwrap();
        assert_eq!(
            connection.project,
            Some(Selected::new("from-creds".into(), SettingSource::Derived))
        );
        assert_eq!(connection.config.project_id(), Some("from-creds"));
    }

    #[rstest]
    #[case::regional(
        None,
        "us-east5",
        "https://us-east5-aiplatform.googleapis.com/v1/projects/proj/locations/us-east5/endpoints/openapi"
    )]
    #[case::global(
        None,
        "global",
        "https://aiplatform.googleapis.com/v1/projects/proj/locations/global/endpoints/openapi"
    )]
    #[case::multi_region_geography(
        None,
        "eu",
        "https://aiplatform.eu.rep.googleapis.com/v1/projects/proj/locations/eu/endpoints/openapi"
    )]
    #[case::custom_base_keeps_project_and_location(
        Some("https://gateway.test/vertex"),
        "europe-west4",
        "https://gateway.test/vertex/v1/projects/proj/locations/europe-west4/endpoints/openapi"
    )]
    fn the_host_and_path_share_one_location(
        #[case] api_base: Option<&str>,
        #[case] location: &str,
        #[case] expected: &str,
    ) {
        let connection = VertexConnection::resolve(
            explicit(Some("proj"), Some(location)),
            api_base,
            &RuntimeDefaults::default(),
            &env(&[]),
        )
        .unwrap();
        assert_eq!(
            connection
                .resource_url("proj", &["endpoints", "openapi"])
                .unwrap(),
            expected
        );
    }

    #[rstest]
    #[case::host_injection(None, "attacker.example/", VertexConnectionError::InvalidLocation)]
    #[case::uppercase(None, "US-EAST5", VertexConnectionError::InvalidLocation)]
    #[case::malformed_custom_base(
        Some("not a url"),
        "us-east5",
        VertexConnectionError::InvalidApiBase(InvalidApiBase)
    )]
    fn an_unusable_connection_fails_before_sending(
        #[case] api_base: Option<&str>,
        #[case] location: &str,
        #[case] expected: VertexConnectionError,
    ) {
        assert_eq!(
            VertexConnection::resolve(
                explicit(Some("proj"), Some(location)),
                api_base,
                &RuntimeDefaults::default(),
                &env(&[]),
            )
            .unwrap_err(),
            expected
        );
    }

    #[rstest]
    #[case::regional_location_keeps_the_global_host(
        None,
        "us-central1",
        "https://aiplatform.googleapis.com/v1/projects/p/locations/us-central1/endpoints/openapi"
    )]
    #[case::global_location(
        None,
        "global",
        "https://aiplatform.googleapis.com/v1/projects/p/locations/global/endpoints/openapi"
    )]
    #[case::custom_base_wins(
        Some("https://gateway.test/vertex"),
        "us-central1",
        "https://gateway.test/vertex/v1/projects/p/locations/us-central1/endpoints/openapi"
    )]
    fn global_only_models_use_the_global_host_with_the_resolved_location(
        #[case] api_base: Option<&str>,
        #[case] location: &str,
        #[case] expected: &str,
    ) {
        let connection = VertexConnection::resolve(
            explicit(Some("p"), Some(location)),
            api_base,
            &RuntimeDefaults::default(),
            &env(&[]),
        )
        .unwrap();
        assert_eq!(
            connection
                .global_resource_url("p", &["endpoints", "openapi"])
                .unwrap(),
            expected
        );
    }

    #[test]
    fn global_only_models_still_reject_an_invalid_location() {
        let config = explicit(Some("p"), Some("evil.com#"));
        assert_eq!(
            VertexConnection::resolve(config, None, &RuntimeDefaults::default(), &env(&[]))
                .unwrap_err(),
            VertexConnectionError::InvalidLocation
        );
    }
}
