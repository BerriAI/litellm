use std::path::PathBuf;

use crate::constants::AWS_DEFAULT_REGION;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AwsPartition {
    pub partition: &'static str,
    pub dns_suffix: &'static str,
}

const COMMERCIAL_PARTITION: AwsPartition = AwsPartition {
    partition: "aws",
    dns_suffix: "amazonaws.com",
};

const PARTITIONS_BY_REGION_PREFIX: [(&str, AwsPartition); 6] = [
    (
        "cn-",
        AwsPartition {
            partition: "aws-cn",
            dns_suffix: "amazonaws.com.cn",
        },
    ),
    (
        "us-gov-",
        AwsPartition {
            partition: "aws-us-gov",
            dns_suffix: "amazonaws.com",
        },
    ),
    (
        "us-isob-",
        AwsPartition {
            partition: "aws-iso-b",
            dns_suffix: "sc2s.sgov.gov",
        },
    ),
    (
        "us-isof-",
        AwsPartition {
            partition: "aws-iso-f",
            dns_suffix: "csp.hci.ic.gov",
        },
    ),
    (
        "us-iso-",
        AwsPartition {
            partition: "aws-iso",
            dns_suffix: "c2s.ic.gov",
        },
    ),
    (
        "eu-isoe-",
        AwsPartition {
            partition: "aws-iso-e",
            dns_suffix: "cloud.adc-e.uk",
        },
    ),
];

pub fn aws_partition(region: &str) -> AwsPartition {
    PARTITIONS_BY_REGION_PREFIX
        .iter()
        .find(|(prefix, _)| region.starts_with(prefix))
        .map_or(COMMERCIAL_PARTITION, |(_, partition)| *partition)
}

pub fn bedrock_runtime_endpoint(region: &str) -> String {
    format!(
        "https://bedrock-runtime.{region}.{}",
        aws_partition(region).dns_suffix
    )
}

pub(crate) fn session_region(
    env_lookup: &dyn Fn(&str) -> Option<String>,
    process_env: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    env_lookup(AWS_DEFAULT_REGION)
        .filter(|region| !region.is_empty())
        .or_else(|| profile_region(process_env))
}

fn profile_region(process_env: &dyn Fn(&str) -> Option<String>) -> Option<String> {
    let profile = process_env("AWS_PROFILE")
        .or_else(|| process_env("AWS_DEFAULT_PROFILE"))
        .filter(|profile| !profile.is_empty())
        .unwrap_or_else(|| "default".to_string());
    let home = process_env("HOME");
    let path = match process_env("AWS_CONFIG_FILE").filter(|path| !path.is_empty()) {
        Some(path) => match (path.strip_prefix("~/"), &home) {
            (Some(rest), Some(home)) => PathBuf::from(home).join(rest),
            _ => PathBuf::from(path),
        },
        None => PathBuf::from(home?).join(".aws").join("config"),
    };
    config_file_region(&std::fs::read_to_string(path).ok()?, &profile)
}

fn config_file_region(contents: &str, profile: &str) -> Option<String> {
    let named = format!("profile {profile}");
    let selects = |section: &str| section == named || (profile == "default" && section == profile);
    let mut selected = false;
    for line in contents.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
            continue;
        }
        if let Some(section) = line
            .strip_prefix('[')
            .and_then(|line| line.strip_suffix(']'))
        {
            selected = selects(&section.split_whitespace().collect::<Vec<_>>().join(" "));
            continue;
        }
        if selected
            && let Some((key, value)) = line.split_once('=')
            && key.trim() == "region"
        {
            return Some(value.trim().to_string()).filter(|region| !region.is_empty());
        }
    }
    None
}

pub(crate) fn sts_endpoint_region(endpoint: &str) -> Option<String> {
    let url = url::Url::parse(endpoint).ok()?;
    let labels: Vec<&str> = url.host_str()?.split('.').collect();
    (0..labels.len()).find_map(|index| {
        let ["sts" | "sts-fips", region, rest @ ..] = &labels[index..] else {
            return None;
        };
        let rest = rest.join(".");
        let valid_region = !region.is_empty()
            && region
                .chars()
                .all(|char| char.is_ascii_lowercase() || char.is_ascii_digit() || char == '-');
        (valid_region
            && (rest.starts_with("amazonaws.com") || rest.starts_with("vpce.amazonaws.com")))
        .then(|| region.to_string())
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::commercial("us-east-1", "https://bedrock-runtime.us-east-1.amazonaws.com")]
    #[case::china("cn-north-1", "https://bedrock-runtime.cn-north-1.amazonaws.com.cn")]
    #[case::gov_cloud("us-gov-west-1", "https://bedrock-runtime.us-gov-west-1.amazonaws.com")]
    #[case::iso("us-iso-east-1", "https://bedrock-runtime.us-iso-east-1.c2s.ic.gov")]
    #[case::iso_b(
        "us-isob-east-1",
        "https://bedrock-runtime.us-isob-east-1.sc2s.sgov.gov"
    )]
    #[case::iso_f(
        "us-isof-south-1",
        "https://bedrock-runtime.us-isof-south-1.csp.hci.ic.gov"
    )]
    #[case::iso_e(
        "eu-isoe-west-1",
        "https://bedrock-runtime.eu-isoe-west-1.cloud.adc-e.uk"
    )]
    fn the_runtime_endpoint_uses_the_partition_dns_suffix(
        #[case] region: &str,
        #[case] expected: &str,
    ) {
        assert_eq!(bedrock_runtime_endpoint(region), expected);
    }

    #[rstest]
    #[case::china("cn-northwest-1", "aws-cn")]
    #[case::gov_cloud("us-gov-east-1", "aws-us-gov")]
    #[case::commercial("eu-central-1", "aws")]
    fn a_region_names_its_partition(#[case] region: &str, #[case] expected: &str) {
        assert_eq!(aws_partition(region).partition, expected);
    }

    #[rstest]
    #[case::regional("https://sts.eu-west-1.amazonaws.com", Some("eu-west-1"))]
    #[case::fips("https://sts-fips.us-east-2.amazonaws.com/", Some("us-east-2"))]
    #[case::china("https://sts.cn-north-1.amazonaws.com.cn", Some("cn-north-1"))]
    #[case::vpc_endpoint(
        "https://vpce-0abc-1234.sts.ap-south-1.vpce.amazonaws.com",
        Some("ap-south-1")
    )]
    #[case::global("https://sts.amazonaws.com", None)]
    #[case::custom_host("https://sts.internal.example.com", None)]
    #[case::no_scheme("sts.eu-west-1.amazonaws.com", None)]
    fn the_sts_region_is_read_from_a_regional_endpoint(
        #[case] endpoint: &str,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(sts_endpoint_region(endpoint).as_deref(), expected);
    }

    const CONFIG: &str = "[default]\nregion = eu-west-3\n\n[profile work]\noutput = json\nregion=ap-northeast-1\n\n[profile empty]\nregion =\n";

    #[rstest]
    #[case::default_profile("default", Some("eu-west-3"))]
    #[case::named_profile("work", Some("ap-northeast-1"))]
    #[case::blank_region("empty", None)]
    #[case::unknown_profile("missing", None)]
    fn the_profile_region_comes_from_its_config_section(
        #[case] profile: &str,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(config_file_region(CONFIG, profile).as_deref(), expected);
    }

    #[rstest]
    #[case::default_region_wins(Some("sa-east-1"), None, Some("sa-east-1"))]
    #[case::blank_default_region_reads_the_profile(Some(""), Some("work"), Some("ap-northeast-1"))]
    #[case::profile_from_the_environment(None, Some("work"), Some("ap-northeast-1"))]
    #[case::default_profile(None, None, Some("eu-west-3"))]
    fn the_session_region_reads_the_default_region_then_the_profile(
        #[case] default_region: Option<&str>,
        #[case] profile: Option<&str>,
        #[case] expected: Option<&str>,
    ) {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), CONFIG).unwrap();
        let config = file.path().to_string_lossy().into_owned();
        let env = |name: &str| {
            (name == AWS_DEFAULT_REGION)
                .then(|| default_region.map(str::to_string))
                .flatten()
        };
        let process_env = |name: &str| match name {
            "AWS_CONFIG_FILE" => Some(config.clone()),
            "AWS_PROFILE" => profile.map(str::to_string),
            _ => None,
        };

        assert_eq!(session_region(&env, &process_env).as_deref(), expected);
    }
}
