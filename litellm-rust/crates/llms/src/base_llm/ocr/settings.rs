use std::time::Duration;

use litellm_layered::{Invalid, Layered, read};

#[derive(Clone, Debug, PartialEq)]
pub struct OcrSettings {
    pub request_timeout: Duration,
    pub max_download_bytes: u64,
    pub poll_timeout: Duration,
    pub document_intelligence_api_version: String,
    pub document_intelligence_dpi: i64,
    pub enable_azure_ad_token_refresh: bool,
}

impl Default for OcrSettings {
    fn default() -> Self {
        Self {
            request_timeout: Duration::from_secs(6000),
            max_download_bytes: megabytes(50.0),
            poll_timeout: Duration::from_secs(120),
            document_intelligence_api_version: "2024-11-30".into(),
            document_intelligence_dpi: 96,
            enable_azure_ad_token_refresh: false,
        }
    }
}

#[derive(Layered)]
pub struct OcrKeys {
    #[layer(env = "REQUEST_TIMEOUT", read = timeout_seconds)]
    pub request_timeout: Duration,
    #[layer(env = "MAX_IMAGE_URL_DOWNLOAD_SIZE_MB", read = download_megabytes)]
    pub max_download_bytes: u64,
    #[layer(env = "AZURE_OPERATION_POLLING_TIMEOUT", read = polling_seconds)]
    pub poll_timeout: Duration,
    #[layer(env = "AZURE_DOCUMENT_INTELLIGENCE_API_VERSION", read = read::text)]
    pub document_intelligence_api_version: String,
    #[layer(env = "AZURE_DOCUMENT_INTELLIGENCE_DEFAULT_DPI")]
    pub document_intelligence_dpi: i64,
    pub enable_azure_ad_token_refresh: bool,
}

impl From<OcrKeysPartial> for OcrSettings {
    fn from(merged: OcrKeysPartial) -> Self {
        let defaults = Self::default();
        Self {
            request_timeout: merged.request_timeout.unwrap_or(defaults.request_timeout),
            max_download_bytes: merged
                .max_download_bytes
                .unwrap_or(defaults.max_download_bytes),
            poll_timeout: merged.poll_timeout.unwrap_or(defaults.poll_timeout),
            document_intelligence_api_version: merged
                .document_intelligence_api_version
                .unwrap_or(defaults.document_intelligence_api_version),
            document_intelligence_dpi: merged
                .document_intelligence_dpi
                .unwrap_or(defaults.document_intelligence_dpi),
            enable_azure_ad_token_refresh: merged
                .enable_azure_ad_token_refresh
                .unwrap_or(defaults.enable_azure_ad_token_refresh),
        }
    }
}

fn timeout_seconds(value: &str) -> Result<Option<Duration>, Invalid> {
    Ok(read::parsed::<f64>(value)?.and_then(|seconds| Duration::try_from_secs_f64(seconds).ok()))
}

fn download_megabytes(value: &str) -> Result<Option<u64>, Invalid> {
    Ok(read::parsed::<f64>(value)?
        .filter(|size| size.is_finite())
        .map(megabytes))
}

fn polling_seconds(value: &str) -> Result<Option<Duration>, Invalid> {
    Ok(read::parsed::<i64>(value)?
        .map(|seconds| Duration::from_secs(seconds.max(0).unsigned_abs())))
}

fn megabytes(size: f64) -> u64 {
    (size * 1024.0 * 1024.0) as u64
}

#[cfg(test)]
mod tests {
    use litellm_layered::Env;
    use rstest::rstest;

    use super::*;

    fn from_env(values: &[(&str, &str)]) -> OcrSettings {
        OcrSettings::from(OcrKeysPartial::from_env(&Env::of(values)).unwrap())
    }

    #[test]
    fn an_empty_environment_keeps_the_python_defaults() {
        assert_eq!(from_env(&[]), OcrSettings::default());
    }

    #[test]
    fn every_setting_follows_its_environment_variable() {
        let settings = from_env(&[
            ("REQUEST_TIMEOUT", "30.5"),
            ("MAX_IMAGE_URL_DOWNLOAD_SIZE_MB", "0.5"),
            ("AZURE_OPERATION_POLLING_TIMEOUT", " 600 "),
            ("AZURE_DOCUMENT_INTELLIGENCE_API_VERSION", "2025-01-01"),
            ("AZURE_DOCUMENT_INTELLIGENCE_DEFAULT_DPI", "72"),
        ]);
        assert_eq!(
            settings,
            OcrSettings {
                request_timeout: Duration::from_millis(30_500),
                max_download_bytes: 512 * 1024,
                poll_timeout: Duration::from_secs(600),
                document_intelligence_api_version: "2025-01-01".into(),
                document_intelligence_dpi: 72,
                ..OcrSettings::default()
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
        assert_eq!(
            from_env(&[("MAX_IMAGE_URL_DOWNLOAD_SIZE_MB", value)]).max_download_bytes,
            bytes
        );
    }

    #[test]
    fn a_negative_polling_timeout_expires_immediately() {
        assert_eq!(
            from_env(&[("AZURE_OPERATION_POLLING_TIMEOUT", "-5")]).poll_timeout,
            Duration::ZERO
        );
    }

    #[test]
    fn an_empty_api_version_falls_back_to_the_default() {
        assert_eq!(
            from_env(&[("AZURE_DOCUMENT_INTELLIGENCE_API_VERSION", "")])
                .document_intelligence_api_version,
            OcrSettings::default().document_intelligence_api_version
        );
    }
}
