use std::time::{SystemTime, UNIX_EPOCH};

use litellm_callbacks::Moment;
use sha2::{Digest, Sha256};

use crate::settings::{Granularity, Settings};

const MAX_KEY_BYTES: usize = 1024;
const MAX_FILE_NAME_BYTES: usize = 255;
const MAX_DOWNLOAD_NAME_BYTES: usize = 1024;
const BOUNDED_HEAD_BYTES: i64 = 64;
const MIN_BOUNDED_FILE_NAME_BYTES: usize = 64;
const PREFIX_DIGEST_CHARS: usize = 16;
const EXTENSION: &str = ".json";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stamp {
    pub year: i64,
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
    pub micros: u32,
}

impl From<[i64; 7]> for Stamp {
    fn from(parts: [i64; 7]) -> Self {
        let [year, month, day, hour, minute, second, micros] = parts;
        let part = |value: i64| u32::try_from(value).unwrap_or_default();
        Self {
            year,
            month: part(month),
            day: part(day),
            hour: part(hour),
            minute: part(minute),
            second: part(second),
            micros: part(micros),
        }
    }
}

impl From<&Moment> for Stamp {
    fn from(moment: &Moment) -> Self {
        Self {
            year: i64::from(moment.year),
            month: u32::from(moment.month),
            day: u32::from(moment.day),
            hour: u32::from(moment.hour),
            minute: u32::from(moment.minute),
            second: u32::from(moment.second),
            micros: moment.microsecond,
        }
    }
}

impl Stamp {
    pub fn utc(time: SystemTime) -> Self {
        let elapsed = time.duration_since(UNIX_EPOCH).unwrap_or_default();
        let seconds = elapsed.as_secs();
        let (year, month, day) = civil_from_days((seconds / 86_400) as i64);
        let of_day = seconds % 86_400;
        Self {
            year,
            month,
            day,
            hour: (of_day / 3600) as u32,
            minute: (of_day / 60 % 60) as u32,
            second: (of_day % 60) as u32,
            micros: elapsed.subsec_micros(),
        }
    }

    pub fn date(&self) -> String {
        format!("{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }

    pub fn clock(&self) -> String {
        format!("{:02}-{:02}-{:02}", self.hour, self.minute, self.second)
    }

    fn clock_micros(&self) -> String {
        format!("{}-{:06}", self.clock(), self.micros)
    }
}

fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = (day_of_year - (153 * month_index + 2) / 5 + 1) as u32;
    let month = if month_index < 10 {
        month_index + 3
    } else {
        month_index - 9
    } as u32;
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

pub fn logging_id(start: &Stamp, id: &str) -> String {
    format!("time-{}_{id}", start.clock_micros())
}

pub fn object_key(
    path: &str,
    prefix: &str,
    start: &Stamp,
    file_name: &str,
    granularity: Granularity,
) -> String {
    let sanitized = file_name.replace(['/', ':'], "_");
    let configured = if path.is_empty() {
        prefix.to_owned()
    } else {
        format!("{}/{prefix}", path.trim_end_matches('/'))
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
            file_name_budget.min(budget - configured.len()) as i64,
        );
        return format!("{configured}{date}{bounded}{EXTENSION}");
    }
    let shortest = bounded_name(file_name, &sanitized, MIN_BOUNDED_FILE_NAME_BYTES as i64);
    let prefix = bounded_prefix(&configured, budget as i64 - shortest.len() as i64);
    format!("{prefix}{date}{shortest}{EXTENSION}")
}

pub fn download_name(start: &Stamp, id: &str) -> String {
    let stamp = format!("time-{}T{}", start.date(), start.clock_micros());
    let name = format!("{stamp}_{id}");
    let sanitized = format!("{stamp}_{}", id.replace(['/', '"'], "_"));
    let budget = MAX_DOWNLOAD_NAME_BYTES - EXTENSION.len();
    if sanitized.len() <= budget {
        return format!("{sanitized}{EXTENSION}");
    }
    format!(
        "{}{EXTENSION}",
        bounded_name(&name, &sanitized, budget as i64)
    )
}

fn bounded_name(name: &str, sanitized: &str, max_bytes: i64) -> String {
    let digest = hex_sha256(name);
    let head = truncate(
        sanitized,
        BOUNDED_HEAD_BYTES.min(max_bytes - digest.len() as i64 - 1),
    );
    if head.is_empty() {
        digest
    } else {
        format!("{head}_{digest}")
    }
}

fn bounded_prefix(prefix: &str, max_bytes: i64) -> String {
    let digest = format!("{}/", &hex_sha256(prefix)[..PREFIX_DIGEST_CHARS]);
    if max_bytes < digest.len() as i64 {
        return String::new();
    }
    let head = truncate(prefix, max_bytes - digest.len() as i64 - 1).trim_end_matches('/');
    if head.is_empty() {
        digest
    } else {
        format!("{head}/{digest}")
    }
}

fn truncate(value: &str, max_bytes: i64) -> &str {
    let Ok(max_bytes) = usize::try_from(max_bytes) else {
        return "";
    };
    if max_bytes == 0 {
        return "";
    }
    if value.len() <= max_bytes {
        return value;
    }
    let end = (0..=max_bytes)
        .rev()
        .find(|end| value.is_char_boundary(*end))
        .unwrap_or(0);
    &value[..end]
}

pub fn hex_sha256(value: impl AsRef<[u8]>) -> String {
    hex(&Sha256::digest(value.as_ref()))
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub fn parent(key: &str) -> &str {
    key.rsplit_once('/').map_or("", |(parent, _)| parent)
}

pub fn quote(key: &str) -> String {
    let mut encoded = String::with_capacity(key.len());
    for byte in key.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'-' | b'~' | b'/') {
            encoded.push(byte as char);
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    encoded
}

pub fn dns_suffix(region: Option<&str>) -> &'static str {
    let Some(region) = region.filter(|region| !region.is_empty()) else {
        return "amazonaws.com";
    };
    [
        ("cn-", "amazonaws.com.cn"),
        ("us-gov-", "amazonaws.com"),
        ("us-isob-", "sc2s.sgov.gov"),
        ("us-isof-", "csp.hci.ic.gov"),
        ("us-iso-", "c2s.ic.gov"),
        ("eu-isoe-", "cloud.adc-e.uk"),
    ]
    .into_iter()
    .find(|(prefix, _)| region.starts_with(prefix))
    .map_or("amazonaws.com", |(_, suffix)| suffix)
}

pub fn object_url(settings: &Settings, key: &str) -> String {
    let encoded = quote(key);
    match (&settings.endpoint_url, &settings.bucket_name) {
        (Some(endpoint), Some(bucket)) if settings.use_virtual_hosted_style => {
            let host = endpoint.replace("https://", "").replace("http://", "");
            let scheme = if endpoint.starts_with("https://") {
                "https://"
            } else {
                "http://"
            };
            format!("{scheme}{bucket}.{host}/{encoded}")
        }
        (Some(endpoint), Some(bucket)) => format!("{endpoint}/{bucket}/{encoded}"),
        _ => {
            let region = settings.url_region();
            let suffix = dns_suffix(
                settings
                    .region_name
                    .as_ref()
                    .and_then(serde_json::Value::as_str),
            );
            format!(
                "https://{}.s3.{region}.{suffix}/{encoded}",
                settings.bucket_name.as_deref().unwrap_or("None")
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn stamp() -> Stamp {
        Stamp::from([2026, 11, 13, 9, 5, 7, 42])
    }

    #[rstest]
    #[case("", "", Granularity::Day, "2026-11-13/time-09-05-07-000042_abc.json")]
    #[case(
        "logs/",
        "team/",
        Granularity::Hour,
        "logs/team/2026-11-13/09/time-09-05-07-000042_abc.json"
    )]
    #[case(
        "a/b",
        "",
        Granularity::Day,
        "a/b/2026-11-13/time-09-05-07-000042_abc.json"
    )]
    fn keys_are_laid_out_as_python_lays_them_out(
        #[case] path: &str,
        #[case] prefix: &str,
        #[case] granularity: Granularity,
        #[case] expected: &str,
    ) {
        let name = logging_id(&stamp(), "abc");
        assert_eq!(
            object_key(path, prefix, &stamp(), &name, granularity),
            expected
        );
    }

    #[rstest]
    fn a_long_id_is_bounded_with_its_digest() {
        let name = logging_id(&stamp(), &"x".repeat(400));
        let key = object_key("p", "", &stamp(), &name, Granularity::Day);
        let file = key.rsplit('/').next().unwrap();
        assert!(file.len() <= MAX_FILE_NAME_BYTES);
        assert!(file.ends_with(&format!("_{}.json", hex_sha256(&name))));
    }

    #[rstest]
    fn utc_stamps_split_the_clock() {
        let time = UNIX_EPOCH + std::time::Duration::from_micros(1_763_024_707_000_042);
        assert_eq!(Stamp::utc(time), Stamp::from([2025, 11, 13, 9, 5, 7, 42]));
    }

    #[rstest]
    fn keys_are_quoted_like_urllib() {
        assert_eq!(quote("a b/c=d+e~f.json"), "a%20b/c%3Dd%2Be~f.json");
        assert_eq!(quote("é"), "%C3%A9");
    }

    #[rstest]
    #[case(
        "cn-north-1",
        "https://audit-bucket.s3.cn-north-1.amazonaws.com.cn/2025-01-01/key.json"
    )]
    #[case(
        "cn-northwest-1",
        "https://audit-bucket.s3.cn-northwest-1.amazonaws.com.cn/2025-01-01/key.json"
    )]
    #[case(
        "us-east-1",
        "https://audit-bucket.s3.us-east-1.amazonaws.com/2025-01-01/key.json"
    )]
    #[case(
        "us-gov-west-1",
        "https://audit-bucket.s3.us-gov-west-1.amazonaws.com/2025-01-01/key.json"
    )]
    fn object_urls_follow_the_partition_of_the_region(
        #[case] region: &str,
        #[case] expected: &str,
    ) {
        let params =
            serde_json::json!({"s3_bucket_name": "audit-bucket", "s3_region_name": region});
        let settings = Settings::from_params(
            params.as_object().unwrap(),
            &crate::settings::Constants::default(),
        );
        assert_eq!(object_url(&settings, "2025-01-01/key.json"), expected);
    }

    #[rstest]
    fn download_names_replace_slashes_and_quotes() {
        assert_eq!(
            download_name(&stamp(), "a/b\"c"),
            "time-2026-11-13T09-05-07-000042_a_b_c.json"
        );
    }
}
