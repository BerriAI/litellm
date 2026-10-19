//! Object keys, download names and URLs, laid out as the proxy's `get_s3_object_key` and
//! `S3Logger._build_object_url` lay them out, length bounds included.

use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

use super::config::Settings;
use crate::spend::civil;

/// S3's limit on a key, in UTF-8 bytes.
const MAX_KEY_BYTES: usize = 1024;
/// The proxy's limit on the last segment of a key.
const MAX_FILE_NAME_BYTES: usize = 255;
/// The proxy's limit on the `Content-Disposition` file name.
const MAX_DOWNLOAD_NAME_BYTES: usize = 1024;
/// How much of an over-long name is kept before its digest.
const BOUNDED_HEAD_BYTES: i64 = 64;
const MIN_BOUNDED_FILE_NAME_BYTES: usize = 64;
const PREFIX_DIGEST_CHARS: usize = 16;
const EXTENSION: &str = ".json";

/// `s3_partition_granularity`: a folder per day, or per hour within it.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) enum Granularity {
    #[default]
    Day,
    Hour,
}

/// A wall-clock time, UTC, split the way the key names it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Utc {
    year: i64,
    month: u32,
    day: u32,
    hour: u64,
    minute: u64,
    second: u64,
    micros: u32,
}

impl Utc {
    pub fn new(time: SystemTime) -> Self {
        let elapsed = time.duration_since(UNIX_EPOCH).unwrap_or_default();
        let seconds = elapsed.as_secs();
        let (year, month, day) = civil((seconds / 86_400) as i64);
        let of_day = seconds % 86_400;
        Self {
            year,
            month,
            day,
            hour: of_day / 3600,
            minute: of_day / 60 % 60,
            second: of_day % 60,
            micros: elapsed.subsec_micros(),
        }
    }

    fn date(&self) -> String {
        format!("{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }

    /// `%H-%M-%S`.
    pub fn clock(&self) -> String {
        format!("{:02}-{:02}-{:02}", self.hour, self.minute, self.second)
    }

    /// `%H-%M-%S-%f`.
    fn clock_micros(&self) -> String {
        format!("{}-{:06}", self.clock(), self.micros)
    }
}

/// The proxy's `get_logging_id`: the request's time and the payload id.
pub(super) fn logging_id(start: &Utc, id: &str) -> String {
    format!("time-{}_{id}", start.clock_micros())
}

/// `{s3_path}/[prefix]YYYY-MM-DD[/HH]/{file_name}.json`, shortened as the proxy shortens it when
/// it would pass S3's limits: the file name first, the configured prefix only if that is what
/// does not fit.
pub(super) fn object_key(
    path: &str,
    prefix: &str,
    start: &Utc,
    file_name: &str,
    granularity: Granularity,
) -> String {
    let sanitized = file_name.replace(['/', ':'], "_");
    let configured = match path.is_empty() {
        true => prefix.to_owned(),
        false => format!("{}/{prefix}", path.trim_end_matches('/')),
    };
    let date = match granularity {
        Granularity::Day => format!("{}/", start.date()),
        Granularity::Hour => format!("{}/{:02}/", start.date(), start.hour),
    };
    let file_name_budget = MAX_FILE_NAME_BYTES - EXTENSION.len();
    let key = format!("{configured}{date}{sanitized}{EXTENSION}");
    if key.len() <= MAX_KEY_BYTES && sanitized.len() <= file_name_budget {
        return key;
    }
    let budget = MAX_KEY_BYTES - date.len() - EXTENSION.len();
    if configured.len() + MIN_BOUNDED_FILE_NAME_BYTES <= budget {
        let bounded = bounded_name(
            file_name,
            &sanitized,
            file_name_budget.min(budget - configured.len()),
        );
        return format!("{configured}{date}{bounded}{EXTENSION}");
    }
    let shortest = bounded_name(file_name, &sanitized, MIN_BOUNDED_FILE_NAME_BYTES);
    let prefix = bounded_prefix(&configured, budget.saturating_sub(shortest.len()));
    format!("{prefix}{date}{shortest}{EXTENSION}")
}

/// The proxy's `get_s3_object_download_filename`.
pub(super) fn download_name(start: &Utc, id: &str) -> String {
    let stamp = format!("time-{}T{}", start.date(), start.clock_micros());
    let name = format!("{stamp}_{id}");
    let sanitized = format!("{stamp}_{}", id.replace(['/', '"'], "_"));
    let budget = MAX_DOWNLOAD_NAME_BYTES - EXTENSION.len();
    match sanitized.len() <= budget {
        true => format!("{sanitized}{EXTENSION}"),
        false => format!("{}{EXTENSION}", bounded_name(&name, &sanitized, budget)),
    }
}

/// As much of `sanitized` as fits, then the sha256 of the unsanitized name.
fn bounded_name(name: &str, sanitized: &str, max_bytes: usize) -> String {
    let digest = hex_sha256(name);
    let head_budget = BOUNDED_HEAD_BYTES.min(max_bytes as i64 - digest.len() as i64 - 1);
    let head = truncate(sanitized, head_budget);
    match head.is_empty() {
        true => digest,
        false => format!("{head}_{digest}"),
    }
}

/// As much of the configured prefix as fits, then a digest segment naming all of it.
fn bounded_prefix(prefix: &str, max_bytes: usize) -> String {
    let digest = format!("{}/", &hex_sha256(prefix)[..PREFIX_DIGEST_CHARS]);
    if max_bytes < digest.len() {
        return String::new();
    }
    let head = truncate(prefix, max_bytes as i64 - digest.len() as i64 - 1).trim_end_matches('/');
    match head.is_empty() {
        true => digest,
        false => format!("{head}/{digest}"),
    }
}

/// `value` cut to at most `max_bytes` of UTF-8, never inside a character.
fn truncate(value: &str, max_bytes: i64) -> &str {
    let Ok(max_bytes) = usize::try_from(max_bytes) else {
        return "";
    };
    if value.len() <= max_bytes {
        return value;
    }
    let end = (0..=max_bytes)
        .rev()
        .find(|end| value.is_char_boundary(*end))
        .unwrap_or(0);
    &value[..end]
}

fn hex_sha256(value: &str) -> String {
    Sha256::digest(value.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Everything before the last `/`, the folder batch files are grouped by.
pub(super) fn parent(key: &str) -> &str {
    key.rsplit_once('/').map_or("", |(parent, _)| parent)
}

/// The URL that is signed and sent, with the key percent-encoded once, as the proxy builds it.
pub(super) fn object_url(settings: &Settings, key: &str) -> String {
    let key = encode(key);
    let bucket = &settings.bucket;
    match &settings.endpoint_url {
        Some(endpoint) if settings.virtual_hosted_style => {
            let (scheme, host) = match endpoint.strip_prefix("https://") {
                Some(host) => ("https://", host),
                None => ("http://", endpoint.trim_start_matches("http://")),
            };
            format!("{scheme}{bucket}.{host}/{key}")
        }
        Some(endpoint) => format!("{endpoint}/{bucket}/{key}"),
        None => format!(
            "https://{bucket}.s3.{region}.{suffix}/{key}",
            region = settings.region,
            suffix = dns_suffix(&settings.region)
        ),
    }
}

/// Python's `quote(key, safe="/")`.
fn encode(key: &str) -> String {
    key.bytes()
        .map(|byte| match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' | b'/' => {
                char::from(byte).to_string()
            }
            byte => format!("%{byte:02X}"),
        })
        .collect()
}

/// The AWS partition's domain for a region.
fn dns_suffix(region: &str) -> &'static str {
    [
        ("cn-", "amazonaws.com.cn"),
        ("us-isob-", "sc2s.sgov.gov"),
        ("us-isof-", "csp.hci.ic.gov"),
        ("us-iso-", "c2s.ic.gov"),
        ("eu-isoe-", "cloud.adc-e.uk"),
    ]
    .into_iter()
    .find(|(prefix, _)| region.starts_with(prefix))
    .map_or("amazonaws.com", |(_, suffix)| suffix)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use rstest::rstest;

    use super::*;

    /// 2026-03-04 05:06:07.000089 UTC.
    fn start() -> Utc {
        Utc::new(UNIX_EPOCH + Duration::from_micros(1_772_600_767_000_089))
    }

    #[rstest]
    #[case::bare(
        "",
        "",
        Granularity::Day,
        "2026-03-04/time-05-06-07-000089_chatcmpl-1.json"
    )]
    #[case::path(
        "logs/",
        "",
        Granularity::Day,
        "logs/2026-03-04/time-05-06-07-000089_chatcmpl-1.json"
    )]
    #[case::prefixes(
        "logs",
        "research/team-key/",
        Granularity::Day,
        "logs/research/team-key/2026-03-04/time-05-06-07-000089_chatcmpl-1.json"
    )]
    #[case::hour(
        "logs",
        "",
        Granularity::Hour,
        "logs/2026-03-04/05/time-05-06-07-000089_chatcmpl-1.json"
    )]
    fn keys_are_laid_out_like_the_proxy(
        #[case] path: &str,
        #[case] prefix: &str,
        #[case] granularity: Granularity,
        #[case] expected: &str,
    ) {
        let file_name = logging_id(&start(), "chatcmpl-1");
        assert_eq!(
            object_key(path, prefix, &start(), &file_name, granularity),
            expected
        );
    }

    #[test]
    fn slashes_and_colons_in_the_id_stay_in_one_segment() {
        let file_name = logging_id(&start(), "a/b:c");
        assert_eq!(
            object_key("", "", &start(), &file_name, Granularity::Day),
            "2026-03-04/time-05-06-07-000089_a_b_c.json"
        );
        assert_eq!(
            download_name(&start(), "a/b\"c"),
            "time-2026-03-04T05-06-07-000089_a_b_c.json"
        );
    }

    #[test]
    fn a_long_id_is_shortened_to_a_digest_within_s3_limits() {
        let id = "é".repeat(400);
        let file_name = logging_id(&start(), &id);
        let key = object_key("logs", "", &start(), &file_name, Granularity::Day);
        let (folder, last) = key.rsplit_once('/').unwrap();
        assert_eq!(folder, "logs/2026-03-04");
        assert!(last.len() <= MAX_FILE_NAME_BYTES, "{last}");
        assert!(last.ends_with(&format!("_{}.json", hex_sha256(&file_name))));
        assert!(last.starts_with("time-05-06-07-000089_é"));
    }

    #[test]
    fn a_long_prefix_keeps_its_head_and_a_digest_of_all_of_it() {
        let path = "p".repeat(2000);
        let file_name = logging_id(&start(), "chatcmpl-1");
        let key = object_key(&path, "", &start(), &file_name, Granularity::Day);
        assert!(key.len() <= MAX_KEY_BYTES, "{}", key.len());
        let digest = &hex_sha256(&format!("{path}/"))[..PREFIX_DIGEST_CHARS];
        assert!(key.starts_with("ppp"), "{key}");
        assert!(key.contains(&format!("/{digest}/2026-03-04/")), "{key}");
    }

    fn settings(endpoint: Option<&str>, virtual_hosted: bool, region: &str) -> Settings {
        Settings {
            bucket: "logs".into(),
            path: String::new(),
            region: region.into(),
            endpoint_url: endpoint.map(str::to_owned),
            virtual_hosted_style: virtual_hosted,
            verify: None,
            auth: Default::default(),
            team_prefix: false,
            key_prefix: false,
            strip_base64_files: false,
            prompts_only: false,
            granularity: Granularity::Day,
            server_side_encryption: None,
            sse_kms_key_id: None,
            max_concurrent_uploads: 1,
            max_queue_size: 1,
            max_retry_age: None,
            drop_on_terminal_error: true,
            batch_file_upload: false,
            flush_interval: Duration::from_secs(1),
            attempt_timeout: Duration::from_secs(1),
            log_success: true,
            log_failure: true,
            turn_off_message_logging: false,
            redact_user_api_key_info: false,
            user_agent_tags: true,
        }
    }

    #[rstest]
    #[case::aws(
        None,
        false,
        "eu-west-1",
        "https://logs.s3.eu-west-1.amazonaws.com/a/b%20c%3D%2B.json"
    )]
    #[case::china(
        None,
        false,
        "cn-north-1",
        "https://logs.s3.cn-north-1.amazonaws.com.cn/a/b%20c%3D%2B.json"
    )]
    #[case::path_style(
        Some("http://minio:9000"),
        false,
        "us-east-1",
        "http://minio:9000/logs/a/b%20c%3D%2B.json"
    )]
    #[case::virtual_hosted(
        Some("https://s3.example"),
        true,
        "us-east-1",
        "https://logs.s3.example/a/b%20c%3D%2B.json"
    )]
    fn urls_are_built_like_the_proxy(
        #[case] endpoint: Option<&str>,
        #[case] virtual_hosted: bool,
        #[case] region: &str,
        #[case] expected: &str,
    ) {
        assert_eq!(
            object_url(&settings(endpoint, virtual_hosted, region), "a/b c=+.json"),
            expected
        );
    }

    #[rstest]
    #[case("a/b/c.json", "a/b")]
    #[case("c.json", "")]
    fn parents(#[case] key: &str, #[case] expected: &str) {
        assert_eq!(parent(key), expected);
    }
}
