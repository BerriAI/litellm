pub const AWS_ACCESS_KEY_ID: &str = "AWS_ACCESS_KEY_ID";
pub const AWS_SECRET_ACCESS_KEY: &str = "AWS_SECRET_ACCESS_KEY";
pub const AWS_SESSION_TOKEN: &str = "AWS_SESSION_TOKEN";
pub const AWS_REGION_NAME: &str = "AWS_REGION_NAME";
pub const AWS_REGION: &str = "AWS_REGION";
pub const AWS_DEFAULT_REGION: &str = "AWS_DEFAULT_REGION";
pub const AWS_BEDROCK_RUNTIME_ENDPOINT: &str = "AWS_BEDROCK_RUNTIME_ENDPOINT";
pub const AWS_SESSION_NAME: &str = "AWS_SESSION_NAME";
pub const AWS_PROFILE_NAME: &str = "AWS_PROFILE_NAME";
pub const AWS_ROLE_NAME: &str = "AWS_ROLE_NAME";
pub const AWS_WEB_IDENTITY_TOKEN: &str = "AWS_WEB_IDENTITY_TOKEN";
pub const AWS_ROLE_ARN: &str = "AWS_ROLE_ARN";
pub const AWS_WEB_IDENTITY_TOKEN_FILE: &str = "AWS_WEB_IDENTITY_TOKEN_FILE";
pub const AWS_STS_ENDPOINT: &str = "AWS_STS_ENDPOINT";
pub const AWS_EXTERNAL_ID: &str = "AWS_EXTERNAL_ID";
pub const AWS_BEARER_TOKEN_BEDROCK: &str = "AWS_BEARER_TOKEN_BEDROCK";
pub const SECRET_NAMES: &[&str] = &[
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_SESSION_TOKEN,
    AWS_REGION_NAME,
    AWS_REGION,
    AWS_SESSION_NAME,
    AWS_PROFILE_NAME,
    AWS_ROLE_NAME,
    AWS_WEB_IDENTITY_TOKEN,
    AWS_STS_ENDPOINT,
    AWS_EXTERNAL_ID,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SigV4Header {
    Authorization,
    Host,
    ContentType,
    Date,
    AmzDate,
    AmzSecurityToken,
}

impl SigV4Header {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Authorization => "Authorization",
            Self::Host => "Host",
            Self::ContentType => "Content-Type",
            Self::Date => "Date",
            Self::AmzDate => "X-Amz-Date",
            Self::AmzSecurityToken => "X-Amz-Security-Token",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        [
            Self::Authorization,
            Self::Host,
            Self::ContentType,
            Self::Date,
            Self::AmzDate,
            Self::AmzSecurityToken,
        ]
        .into_iter()
        .find(|header| name.eq_ignore_ascii_case(header.as_str()))
    }

    pub const fn is_computed(self) -> bool {
        matches!(
            self,
            Self::Authorization | Self::Date | Self::AmzDate | Self::AmzSecurityToken
        )
    }
}

pub const BEDROCK_SERVICE: &str = "bedrock";
pub const DEFAULT_SESSION_NAME_PREFIX: &str = "litellm-session";
pub const DEFAULT_BEDROCK_REGION: &str = "us-west-2";
pub const BEDROCK_RUNTIME_ENDPOINT_TEMPLATE: &str =
    "https://bedrock-runtime.{region}.amazonaws.com";
