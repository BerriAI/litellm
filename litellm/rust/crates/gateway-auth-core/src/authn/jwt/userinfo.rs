use jiff::Timestamp;
use serde_json::{Map, Value};

use super::{
    Claims, JwtSettings,
    verify::{self, ttl},
};
use crate::{AuthHost, Error, Token, VerifyDenial};

/// `get_oidc_userinfo`
pub async fn oidc_userinfo(
    host: &impl AuthHost,
    token: &Token,
    settings: &JwtSettings,
) -> Result<Value, Error> {
    let Some(endpoint) = settings
        .config
        .oidc_userinfo_endpoint
        .as_deref()
        .filter(|endpoint| !endpoint.is_empty())
    else {
        return Err(VerifyDenial::UserinfoNotConfigured.into());
    };
    let cache_key = format!("oidc_userinfo_{}", token.hash());
    if let Some(cached) = host.cached_json(&cache_key).await? {
        return Ok(cached);
    }
    let userinfo = host
        .fetch_userinfo(endpoint, token)
        .await?
        .unwrap_or(Value::Null);
    host.cache_json(
        &cache_key,
        &userinfo,
        ttl(settings.config.oidc_userinfo_cache_ttl),
    )
    .await?;
    Ok(userinfo)
}

/// `auth_jwt`, or `get_oidc_userinfo` for a token that is not a JWT when UserInfo is enabled.
pub(crate) async fn decode(
    host: &impl AuthHost,
    token: &Token,
    settings: &JwtSettings,
    now: Timestamp,
) -> Result<Claims, Error> {
    if settings.config.oidc_userinfo_enabled && !token.is_jwt() {
        let userinfo = oidc_userinfo(host, token, settings).await?;
        return Ok(Claims(match userinfo {
            Value::Object(claims) => claims,
            _ => Map::new(),
        }));
    }
    verify::verify(host, token, settings, now).await
}

#[cfg(test)]
mod tests {
    use litellm_config::JwtAuthSettings;
    use rstest::{fixture, rstest};
    use serde_json::json;

    use super::*;
    use crate::testing::ScriptedHost;

    const USERINFO: &str = "https://idp.example/userinfo";
    const USERINFO_KEY: &str =
        "oidc_userinfo_84d3f23da9b5f51b3269566eff05d3fb23607eeef89567f9cd280b90ca0dbc5c";

    #[fixture]
    fn settings() -> JwtSettings {
        JwtSettings::default()
    }

    fn userinfo_settings(settings: JwtSettings, endpoint: Option<&str>) -> JwtSettings {
        JwtSettings {
            config: JwtAuthSettings {
                oidc_userinfo_endpoint: endpoint.map(str::to_owned),
                ..settings.config
            },
            ..settings
        }
    }

    async fn userinfo(
        host: &ScriptedHost,
        token: &str,
        settings: &JwtSettings,
    ) -> Result<Value, Error> {
        oidc_userinfo(host, &Token::new(token), settings).await
    }

    #[rstest]
    #[case::unset(None)]
    #[case::empty(Some(""))]
    #[tokio::test]
    async fn userinfo_needs_an_endpoint(settings: JwtSettings, #[case] endpoint: Option<&str>) {
        let host = ScriptedHost::default();

        let userinfo = userinfo(
            &host,
            "opaque-token",
            &userinfo_settings(settings, endpoint),
        )
        .await;

        assert_eq!(userinfo, Err(VerifyDenial::UserinfoNotConfigured.into()));
        assert!(host.calls().is_empty());
    }

    #[rstest]
    #[tokio::test]
    async fn userinfo_is_fetched_once_then_cached(settings: JwtSettings) {
        let settings = userinfo_settings(settings, Some(USERINFO));
        let host = ScriptedHost {
            userinfo: Some(json!({"sub": "u1"})),
            ..ScriptedHost::default()
        };

        let first = userinfo(&host, "opaque-token", &settings).await;
        let second = userinfo(&host, "opaque-token", &settings).await;

        assert_eq!(
            (first, second),
            (Ok(json!({"sub": "u1"})), Ok(json!({"sub": "u1"})))
        );
        assert_eq!(
            host.calls(),
            [
                format!("get {USERINFO_KEY}"),
                format!("userinfo {USERINFO} opaque-token"),
                format!("set {USERINFO_KEY} 300"),
                format!("get {USERINFO_KEY}"),
            ]
        );
    }
}
