use litellm_auth_gcp::{VertexConfig, get_vertex_ai_location, get_vertex_ai_project};
use serde_json::Value;
use url::Url;

use crate::{Error, ErrorDetail};

pub const VERTEX_AI_API_KEY_ENV: &str = "VERTEX_AI_API_KEY";
pub const VERTEXAI_API_KEY_ENV: &str = "VERTEXAI_API_KEY";
pub const DEFAULT_VERTEX_LOCATION: &str = "us-central1";
const GLOBAL_LOCATION: &str = "global";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VertexLocation(String);

impl VertexLocation {
    pub fn parse(location: &str) -> Result<Self, Error> {
        let valid = location == GLOBAL_LOCATION
            || location
                .as_bytes()
                .first()
                .is_some_and(u8::is_ascii_lowercase)
                && location
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-');
        if valid {
            return Ok(Self(location.to_string()));
        }
        Err(Error::InvalidRequest(ErrorDetail::InvalidValue {
            field: "vertex_location",
            expected: "`global` or a lowercase region such as `us-central1`",
            actual: Value::String(location.to_string()),
        }))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn base_url(&self) -> Url {
        let location = self.as_str();
        let host = match location {
            GLOBAL_LOCATION => "aiplatform.googleapis.com".to_string(),
            geography if !geography.contains('-') => {
                format!("aiplatform.{geography}.rep.googleapis.com")
            }
            region => format!("{region}-aiplatform.googleapis.com"),
        };
        Url::parse(&format!("https://{host}")).expect("a validated location forms a valid host")
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VertexTarget {
    pub project: String,
    pub location: VertexLocation,
}

impl VertexTarget {
    pub fn resolve(
        config: &VertexConfig,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<Self, Error> {
        let project = get_vertex_ai_project(config, env_lookup).ok_or_else(|| {
            Error::Auth(litellm_auth::Error::InvalidConfiguration(
                "Vertex AI project is required: set `vertex_project` or VERTEXAI_PROJECT".into(),
            ))
        })?;
        let location = get_vertex_ai_location(config, env_lookup)
            .unwrap_or_else(|| DEFAULT_VERTEX_LOCATION.to_string());
        Ok(Self {
            project,
            location: VertexLocation::parse(&location)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::global("global", "https://aiplatform.googleapis.com/")]
    #[case::multi_region_geography("us", "https://aiplatform.us.rep.googleapis.com/")]
    #[case::region("europe-west4", "https://europe-west4-aiplatform.googleapis.com/")]
    fn the_location_picks_the_host(#[case] location: &str, #[case] expected: &str) {
        assert_eq!(
            VertexLocation::parse(location).unwrap().base_url().as_str(),
            expected
        );
    }

    #[rstest]
    #[case::host_injection("attacker.example/")]
    #[case::fragment("evil.com#")]
    #[case::uppercase("US-CENTRAL1")]
    #[case::leading_digit("1region")]
    #[case::leading_hyphen("-us")]
    #[case::empty("")]
    #[case::userinfo("us@evil")]
    fn a_location_that_could_leave_the_hostname_is_rejected(#[case] location: &str) {
        assert!(matches!(
            VertexLocation::parse(location),
            Err(Error::InvalidRequest(ErrorDetail::InvalidValue {
                field: "vertex_location",
                ..
            }))
        ));
    }

    fn env(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |name| {
            pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        }
    }

    #[test]
    fn the_target_comes_from_the_environment_and_defaults_the_location() {
        assert_eq!(
            VertexTarget::resolve(
                &VertexConfig::default(),
                &env(&[("VERTEXAI_PROJECT", "p1")])
            )
            .unwrap(),
            VertexTarget {
                project: "p1".into(),
                location: VertexLocation::parse(DEFAULT_VERTEX_LOCATION).unwrap(),
            }
        );
        assert_eq!(
            VertexTarget::resolve(
                &VertexConfig::default(),
                &env(&[("VERTEXAI_PROJECT", "p1"), ("VERTEX_LOCATION", "us-east5")])
            )
            .unwrap()
            .location
            .as_str(),
            "us-east5"
        );
    }

    #[test]
    fn a_missing_project_or_bad_location_fails_resolution() {
        assert!(matches!(
            VertexTarget::resolve(&VertexConfig::default(), &|_| None),
            Err(Error::Auth(litellm_auth::Error::InvalidConfiguration(_)))
        ));
        assert!(matches!(
            VertexTarget::resolve(
                &VertexConfig::default(),
                &env(&[
                    ("VERTEXAI_PROJECT", "p1"),
                    ("VERTEXAI_LOCATION", "evil.com#")
                ])
            ),
            Err(Error::InvalidRequest(_))
        ));
    }
}
