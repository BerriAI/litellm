use std::sync::OnceLock;
use std::time::Duration;

const AUDIO_TRANSCRIPTION_TIMEOUT_SECS: u64 = 600;

pub(super) fn http_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(Duration::from_secs(AUDIO_TRANSCRIPTION_TIMEOUT_SECS))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new())
    })
}
