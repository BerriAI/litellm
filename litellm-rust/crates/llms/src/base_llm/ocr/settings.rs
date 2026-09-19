use std::{sync::Arc, time::Duration};

use litellm_core_utils::settings::Lookup;

pub type Secrets = Arc<dyn Lookup + Send + Sync>;

#[derive(Clone, Debug, PartialEq)]
pub struct OcrSettings {
    pub request_timeout: Duration,
    pub max_download_bytes: u64,
    pub poll_timeout: Duration,
    pub document_intelligence_api_version: String,
    pub document_intelligence_dpi: i64,
}

impl Default for OcrSettings {
    fn default() -> Self {
        Self {
            request_timeout: Duration::from_secs(6000),
            max_download_bytes: megabytes(50.0),
            poll_timeout: Duration::from_secs(120),
            document_intelligence_api_version: "2024-11-30".into(),
            document_intelligence_dpi: 96,
        }
    }
}

impl OcrSettings {
    pub fn from_environment(env: &impl Lookup) -> Self {
        let defaults = Self::default();
        Self {
            request_timeout: env
                .parsed::<f64>("REQUEST_TIMEOUT")
                .and_then(|seconds| Duration::try_from_secs_f64(seconds).ok())
                .unwrap_or(defaults.request_timeout),
            max_download_bytes: env
                .parsed::<f64>("MAX_IMAGE_URL_DOWNLOAD_SIZE_MB")
                .filter(|size| size.is_finite())
                .map_or(defaults.max_download_bytes, megabytes),
            poll_timeout: env
                .parsed::<i64>("AZURE_OPERATION_POLLING_TIMEOUT")
                .map_or(defaults.poll_timeout, |seconds| {
                    Duration::from_secs(seconds.max(0).unsigned_abs())
                }),
            document_intelligence_api_version: env
                .get("AZURE_DOCUMENT_INTELLIGENCE_API_VERSION")
                .unwrap_or(defaults.document_intelligence_api_version),
            document_intelligence_dpi: env
                .parsed("AZURE_DOCUMENT_INTELLIGENCE_DEFAULT_DPI")
                .unwrap_or(defaults.document_intelligence_dpi),
        }
    }
}

fn megabytes(size: f64) -> u64 {
    (size * 1024.0 * 1024.0) as u64
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn env_of(values: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |name| {
            values
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        }
    }

    #[test]
    fn an_empty_environment_keeps_the_python_defaults() {
        assert_eq!(
            OcrSettings::from_environment(&env_of(&[])),
            OcrSettings::default()
        );
    }

    #[test]
    fn every_setting_follows_its_environment_variable() {
        let settings = OcrSettings::from_environment(&env_of(&[
            ("REQUEST_TIMEOUT", "30.5"),
            ("MAX_IMAGE_URL_DOWNLOAD_SIZE_MB", "0.5"),
            ("AZURE_OPERATION_POLLING_TIMEOUT", " 600 "),
            ("AZURE_DOCUMENT_INTELLIGENCE_API_VERSION", "2025-01-01"),
            ("AZURE_DOCUMENT_INTELLIGENCE_DEFAULT_DPI", "72"),
        ]));
        assert_eq!(
            settings,
            OcrSettings {
                request_timeout: Duration::from_millis(30_500),
                max_download_bytes: 512 * 1024,
                poll_timeout: Duration::from_secs(600),
                document_intelligence_api_version: "2025-01-01".into(),
                document_intelligence_dpi: 72,
            }
        );
    }

    #[rstest]
    #[case::zero_disables_downloads("0", 0)]
    #[case::negative_rejects_every_download("-1", 0)]
    #[case::fraction_truncates_like_int("0.0000001", 0)]
    #[case::unparsable_keeps_the_default("big", 50 * 1024 * 1024)]
    fn download_size_converts_megabytes_like_python(
        #[case] value: &'static str,
        #[case] bytes: u64,
    ) {
        let env =
            move |name: &str| (name == "MAX_IMAGE_URL_DOWNLOAD_SIZE_MB").then(|| value.to_string());
        assert_eq!(
            OcrSettings::from_environment(&env).max_download_bytes,
            bytes
        );
    }

    #[test]
    fn a_negative_polling_timeout_expires_immediately() {
        let env =
            |name: &str| (name == "AZURE_OPERATION_POLLING_TIMEOUT").then(|| "-5".to_string());
        assert_eq!(
            OcrSettings::from_environment(&env).poll_timeout,
            Duration::ZERO
        );
    }

    #[test]
    fn an_empty_api_version_is_sent_as_is_like_python_str_of_getenv() {
        let env =
            |name: &str| (name == "AZURE_DOCUMENT_INTELLIGENCE_API_VERSION").then(String::new);
        assert_eq!(
            OcrSettings::from_environment(&env).document_intelligence_api_version,
            ""
        );
    }
}
