use std::{collections::HashSet, iter, time::Duration};

use jiff::{SignedDuration, Timestamp};
use jsonwebtoken::{
    Algorithm, AlgorithmFamily, DecodingKey, Header, Validation, dangerous::insecure_decode_claims,
};
use litellm_config::{JwtIssuerConfig, OneOrMany};
use serde_json::{Map, Value, json};
use strum::VariantArray;

use super::{
    AssertionRejection, JwtDenial, JwtSettings, KeyError, KeyFailure, Searched, TokenError,
    VerifyDenial,
    claims::{InternalClaim, nested_value},
};
use crate::{AuthHost, Error, OperatorHooks, credential::key::Token, value};

pub const ID_JAG_TYP: &str = "oauth-id-jag+jwt";
const JWKS_FETCH_ATTEMPTS: u32 = 3;
const JWKS_UNREACHABLE_BACKOFF: Duration = Duration::from_secs(30);
const STALE_CACHE_KEY_PREFIX: &str = "litellm_stale_";
const STALE_WRITTEN_AT_CACHE_KEY_PREFIX: &str = "litellm_stale_written_at_";
const UNREACHABLE_CACHE_KEY_PREFIX: &str = "litellm_jwks_unreachable_";
const OIDC_DISCOVERY_PATH: &str = ".well-known/openid-configuration";

/// The JWK members the verifier reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr, strum::VariantArray)]
enum JwkField {
    #[strum(serialize = "kty")]
    Kty,
    #[strum(serialize = "kid")]
    Kid,
    #[strum(serialize = "n")]
    N,
    #[strum(serialize = "e")]
    E,
    #[strum(serialize = "x")]
    X,
    #[strum(serialize = "y")]
    Y,
    #[strum(serialize = "crv")]
    Crv,
}

/// A signing key lookup: the host failed, or the lookup found the key or why there is none.
type Lookup<T> = Result<Result<T, KeyFailure>, Error>;

#[derive(Clone, Debug, PartialEq)]
pub struct Claims(pub Map<String, Value>);

/// What the token endpoint asks of an assertion on top of the trusted issuer.
#[derive(Clone, Debug, Default)]
pub struct Expected {
    pub audience: Vec<String>,
    pub required_claims: Vec<String>,
}

impl KeyFailure {
    fn unreachable(url: &str) -> Self {
        Self::Unreachable {
            error: format!("TransportError fetching {url} after {JWKS_FETCH_ATTEMPTS} attempts"),
        }
    }
}

pub fn is_jwt(token: &str) -> bool {
    token.split('.').count() == 3
}

/// `get_unverified_claims`: the payload of a well-formed token, signature unchecked.
pub fn unverified_claims(token: &str) -> Option<Claims> {
    if !is_jwt(token) {
        return None;
    }
    insecure_decode_claims(token).ok().map(Claims)
}

/// Whether `token` carries every claim of `verified` that the verifier did not write itself.
/// A claim the token lacks reads as `null`, like `dict.get`.
pub fn presents_claims(token: &str, verified: &Claims) -> bool {
    let Some(presented) = unverified_claims(token) else {
        return false;
    };
    verified
        .0
        .iter()
        .filter(|(claim, _)| !is_internal_claim(claim))
        .all(|(claim, value)| presented.0.get(claim).unwrap_or(&Value::Null) == value)
}

fn is_internal_claim(claim: &str) -> bool {
    claim.parse::<InternalClaim>().is_ok()
}

/// `managed_issuer_is_trusted`
pub fn managed_issuer_is_trusted(issuer: &Value, settings: &JwtSettings) -> bool {
    let Some(issuer) = issuer.as_str() else {
        return false;
    };
    match settings
        .config
        .issuers
        .iter()
        .flatten()
        .find(|config| config.issuer == issuer)
    {
        Some(config) => {
            config.audience.as_ref().is_some_and(names_an_audience)
                && !config.disable_audience_validation
        }
        None => {
            settings.issuer.as_deref() == Some(issuer)
                && settings
                    .audience
                    .as_deref()
                    .is_some_and(|aud| !aud.is_empty())
        }
    }
}

fn names_an_audience(audience: &OneOrMany) -> bool {
    match audience {
        OneOrMany::One(audience) => !audience.is_empty(),
        OneOrMany::Many(audiences) => !audiences.is_empty(),
    }
}

/// `auth_jwt`
pub async fn verify(
    host: &impl AuthHost,
    token: &Token,
    settings: &JwtSettings,
    now: Timestamp,
) -> Result<Claims, Error> {
    let token = token.expose();
    let header = match header(token) {
        Ok(header) => header,
        Err(error) => return Err(VerifyDenial::Malformed(error).into()),
    };
    if header.typ.as_deref() == Some(ID_JAG_TYP) {
        return Err(VerifyDenial::IdJagNotBearer.into());
    }
    let leeway = leeway(settings);

    if let Some(issuer) = configured_issuer(token, settings) {
        let key = match issuer_key(host, issuer, header.kid.as_deref(), settings, now).await? {
            Ok(key) => key,
            Err(failure) => return Err(VerifyDenial::Key(failure).into()),
        };
        let audience = issuer.audience.as_ref().map(OneOrMany::values);
        if audience.is_none() && !issuer.disable_audience_validation {
            return Err(VerifyDenial::Invalid(TokenError::AudienceRequired).into());
        }
        let checks = Checks {
            audience,
            issuer: Some(&issuer.issuer),
            require: &[],
            leeway,
            now,
        };
        return decode(token, &key, &checks)
            .map(|payload| Claims(apply_issuer_claim_mappings(&payload, issuer)))
            .map_err(rejected);
    }

    let key = match global_key(host, header.kid.as_deref(), settings, now).await? {
        Ok(key) => key,
        Err(failure) => return Err(VerifyDenial::Key(failure).into()),
    };
    let checks = Checks {
        audience: settings.audience.as_ref().map(std::slice::from_ref),
        issuer: settings.issuer.as_deref(),
        require: &[],
        leeway,
        now,
    };
    decode(token, &key, &checks)
        .map(|payload| Claims(without_internal_claims(&payload)))
        .map_err(rejected)
}

fn rejected(error: TokenError) -> Error {
    match error {
        TokenError::Expired => VerifyDenial::Expired.into(),
        error => VerifyDenial::Invalid(error).into(),
    }
}

/// `decode_trusted_assertion`. A JWKS outage is the 503 on either issuer path.
pub async fn verify_assertion(
    host: &impl AuthHost,
    token: &Token,
    expected: &Expected,
    settings: &JwtSettings,
    now: Timestamp,
) -> Result<Claims, Error> {
    let token = token.expose();
    let reject = |rejection| Err(VerifyDenial::Assertion(rejection).into());
    let header = match header(token) {
        Ok(header) => header,
        Err(error) => return reject(AssertionRejection::Malformed(error)),
    };
    let issuer_config = configured_issuer(token, settings);
    let claimed_issuer = unverified_claims(token).and_then(|claims| claims.0.get("iss").cloned());
    let trusted_issuer = issuer_config
        .map(|config| config.issuer.as_str())
        .or(settings.issuer.as_deref())
        .filter(|issuer| {
            !issuer.is_empty() && claimed_issuer.as_ref().and_then(Value::as_str) == Some(*issuer)
        });
    let Some(trusted_issuer) = trusted_issuer else {
        return reject(AssertionRejection::UntrustedIssuer);
    };

    let key = match issuer_config {
        Some(config) => issuer_key(host, config, header.kid.as_deref(), settings, now).await?,
        None => global_key(host, header.kid.as_deref(), settings, now).await?,
    };
    let key = match key {
        Ok(key) => key,
        Err(failure @ KeyFailure::Unreachable { .. }) => {
            return Err(VerifyDenial::Key(failure).into());
        }
        Err(failure) => return reject(AssertionRejection::NoSigningKey(failure)),
    };
    let checks = Checks {
        audience: Some(&expected.audience),
        issuer: Some(trusted_issuer),
        require: &expected.required_claims,
        leeway: leeway(settings),
        now,
    };
    let payload = match decode(token, &key, &checks) {
        Ok(payload) => payload,
        Err(error) => return reject(AssertionRejection::Invalid(error)),
    };
    let verified = match issuer_config {
        Some(config) => apply_issuer_claim_mappings(&payload, config),
        None => without_internal_claims(&payload),
    };
    if !operator_accepts(host, &verified, settings).await? {
        return reject(AssertionRejection::CustomValidate);
    }
    Ok(Claims(verified))
}

/// Whether the operator's `custom_validate` accepts `claims`, true when none is configured. A host
/// with no operator hooks cannot say, so the request is refused.
pub(super) async fn operator_accepts(
    host: &impl AuthHost,
    claims: &Map<String, Value>,
    settings: &JwtSettings,
) -> Result<bool, Error> {
    if settings.config.custom_validate.is_none() {
        return Ok(true);
    }
    let hooks = host
        .operator_hooks()
        .ok_or(JwtDenial::CustomValidateUnavailable)?;
    hooks.custom_validate(&Value::Object(claims.clone())).await
}

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

fn configured_issuer<'s>(token: &str, settings: &'s JwtSettings) -> Option<&'s JwtIssuerConfig> {
    let issuers = settings
        .config
        .issuers
        .as_deref()
        .filter(|issuers| !issuers.is_empty())?;
    let claims = unverified_claims(token)?;
    let issuer = claims
        .0
        .get("iss")?
        .as_str()
        .filter(|issuer| !issuer.is_empty())?;
    issuers.iter().find(|config| config.issuer == issuer)
}

fn jwks_url_for_issuer(issuer: &JwtIssuerConfig) -> String {
    match issuer.jwks_url.as_deref().filter(|url| !url.is_empty()) {
        Some(url) => url.to_owned(),
        None => format!(
            "{}/{OIDC_DISCOVERY_PATH}",
            issuer.issuer.trim_end_matches('/')
        ),
    }
}

fn without_internal_claims(payload: &Map<String, Value>) -> Map<String, Value> {
    payload
        .iter()
        .filter(|(claim, _)| !is_internal_claim(claim))
        .map(|(claim, value)| (claim.clone(), value.clone()))
        .collect()
}

/// `_apply_issuer_claim_mappings`: the issuer's own claim names normalized into `_litellm_*`.
fn apply_issuer_claim_mappings(
    payload: &Map<String, Value>,
    issuer: &JwtIssuerConfig,
) -> Map<String, Value> {
    let mappings = [
        (&issuer.user_id_jwt_field, InternalClaim::UserId),
        (&issuer.user_email_jwt_field, InternalClaim::UserEmail),
        (&issuer.team_id_jwt_field, InternalClaim::TeamId),
        (&issuer.team_ids_jwt_field, InternalClaim::TeamIds),
        (&issuer.org_id_jwt_field, InternalClaim::OrgId),
        (&issuer.end_user_id_jwt_field, InternalClaim::EndUserId),
    ];
    let mapped = mappings.into_iter().filter_map(|(field, claim)| {
        let value = nested_value(payload, field.as_ref()?)?;
        let present = !value.is_null() && value.as_str() != Some("");
        present.then(|| (<&str>::from(claim).to_owned(), value.clone()))
    });
    without_internal_claims(payload)
        .into_iter()
        .chain(iter::once((
            <&str>::from(InternalClaim::Issuer).to_owned(),
            Value::String(issuer.issuer.clone()),
        )))
        .chain(mapped)
        .collect()
}

async fn issuer_key(
    host: &impl AuthHost,
    issuer: &JwtIssuerConfig,
    kid: Option<&str>,
    settings: &JwtSettings,
    now: Timestamp,
) -> Lookup<Map<String, Value>> {
    key_from_jwks_url(host, &jwks_url_for_issuer(issuer), kid, settings, now).await
}

/// `get_public_key`: the first of the `JWT_PUBLIC_KEY_URL` endpoints that holds the kid.
async fn global_key(
    host: &impl AuthHost,
    kid: Option<&str>,
    settings: &JwtSettings,
    now: Timestamp,
) -> Lookup<Map<String, Value>> {
    let urls: Vec<&str> = settings
        .public_key_urls
        .iter()
        .map(|url| url.trim())
        .filter(|url| !url.is_empty())
        .collect();
    if urls.is_empty() {
        return Ok(Err(KeyFailure::MissingUrl));
    }
    for url in &urls {
        match key_from_jwks_url(host, url, kid, settings, now).await? {
            Err(KeyFailure::NoMatch { .. }) => continue,
            found => return Ok(found),
        }
    }
    Ok(Err(KeyFailure::NoMatch {
        keys: Searched::Urls(urls.into_iter().map(str::to_owned).collect()),
        kid: kid.map(str::to_owned),
    }))
}

/// `_get_public_key_from_jwks_url`
async fn key_from_jwks_url(
    host: &impl AuthHost,
    url: &str,
    kid: Option<&str>,
    settings: &JwtSettings,
    now: Timestamp,
) -> Lookup<Map<String, Value>> {
    let jwks_url = if url.contains(OIDC_DISCOVERY_PATH) {
        let cache_key = format!("litellm_oidc_discovery_{url}");
        match cached_with_stale_fallback(host, &cache_key, Source::Discovery(url), settings, now)
            .await?
        {
            Ok(jwks_uri) => value::display(&jwks_uri),
            Err(failure) => return Ok(Err(failure)),
        }
    } else {
        url.to_owned()
    };
    let cache_key = format!("litellm_jwt_auth_keys_{jwks_url}");
    let keys =
        match cached_with_stale_fallback(host, &cache_key, Source::Jwks(&jwks_url), settings, now)
            .await?
        {
            Ok(keys) => keys,
            Err(failure) => return Ok(Err(failure)),
        };
    Ok(parse_keys(&keys, kid)
        .cloned()
        .ok_or_else(|| KeyFailure::NoMatch {
            keys: Searched::Url(jwks_url),
            kid: kid.map(str::to_owned),
        }))
}

/// `parse_keys`: a lone key matches when its kid does or the token names none; among several,
/// the last whose kid equals the token's. Not `JwkSet::find`, which takes the first, and only after every key in
/// the set has parsed.
fn parse_keys<'k>(keys: &'k Value, kid: Option<&str>) -> Option<&'k Map<String, Value>> {
    let kid_matches =
        |key: &Map<String, Value>| kid.is_none() || key.get("kid").and_then(Value::as_str) == kid;
    match keys {
        Value::Object(key) if key.len() == 1 => kid_matches(key).then_some(key),
        Value::Array(keys) if keys.len() == 1 => {
            keys.first()?.as_object().filter(|key| kid_matches(key))
        }
        Value::Array(keys) => {
            let kid = kid?;
            keys.iter()
                .filter_map(Value::as_object)
                .rfind(|key| key.get("kid").and_then(Value::as_str) == Some(kid))
        }
        _ => None,
    }
}

#[derive(Clone, Copy)]
enum Source<'a> {
    Discovery(&'a str),
    Jwks(&'a str),
}

/// `_cached_with_stale_fallback` and `_refresh_or_serve_stale`. A copy taken more than
/// `public_key_ttl + public_key_stale_ttl` ago is never served, whatever its cache expiry says.
async fn cached_with_stale_fallback(
    host: &impl AuthHost,
    cache_key: &str,
    source: Source<'_>,
    settings: &JwtSettings,
    now: Timestamp,
) -> Lookup<Value> {
    if let Some(cached) = host.cached_json(cache_key).await? {
        return Ok(Ok(cached));
    }
    let (ttl, stale_ttl) = (
        settings.config.public_key_ttl,
        settings.config.public_key_stale_ttl,
    );
    let error = match refresh_or_record_outage(host, cache_key, source, ttl, stale_ttl, now).await?
    {
        Err(KeyFailure::Unreachable { error }) => error,
        refreshed => return Ok(refreshed),
    };
    let unreachable = KeyFailure::Unreachable { error };
    if stale_ttl <= 0.0 {
        return Ok(Err(unreachable));
    }
    let stale = host
        .cached_json(&format!("{STALE_CACHE_KEY_PREFIX}{cache_key}"))
        .await?;
    let written_at = host
        .cached_json(&format!("{STALE_WRITTEN_AT_CACHE_KEY_PREFIX}{cache_key}"))
        .await?;
    let age = written_at
        .as_ref()
        .and_then(Value::as_f64)
        .map(|written_at| epoch_seconds(now) - written_at);
    Ok(match (stale, age) {
        (Some(stale), Some(age)) if age <= ttl + stale_ttl => Ok(stale),
        _ => Err(unreachable),
    })
}

/// `_refresh_or_record_outage`: one fetch per backoff window while the IdP stays unreachable.
async fn refresh_or_record_outage(
    host: &impl AuthHost,
    cache_key: &str,
    source: Source<'_>,
    ttl: f64,
    stale_ttl: f64,
    now: Timestamp,
) -> Lookup<Value> {
    let unreachable_key = format!("{UNREACHABLE_CACHE_KEY_PREFIX}{cache_key}");
    if let Some(recent_failure) = host.cached_json(&unreachable_key).await? {
        return Ok(Err(KeyFailure::Unreachable {
            error: value::display(&recent_failure),
        }));
    }
    let refreshed = match refresh(host, source).await? {
        Ok(refreshed) => refreshed,
        Err(KeyFailure::Unreachable { error }) => {
            host.cache_json(
                &unreachable_key,
                &Value::String(error.clone()),
                JWKS_UNREACHABLE_BACKOFF,
            )
            .await?;
            return Ok(Err(KeyFailure::Unreachable { error }));
        }
        Err(failure) => return Ok(Err(failure)),
    };
    host.cache_json(cache_key, &refreshed, self::ttl(ttl))
        .await?;
    if stale_ttl > 0.0 {
        let lifetime = self::ttl(ttl + stale_ttl);
        host.cache_json(
            &format!("{STALE_CACHE_KEY_PREFIX}{cache_key}"),
            &refreshed,
            lifetime,
        )
        .await?;
        host.cache_json(
            &format!("{STALE_WRITTEN_AT_CACHE_KEY_PREFIX}{cache_key}"),
            &json!(epoch_seconds(now)),
            lifetime,
        )
        .await?;
    }
    Ok(Ok(refreshed))
}

/// `_fetch_jwks_uri_from_discovery` and `_fetch_jwks_keys`, past the host's fetch.
async fn refresh(host: &impl AuthHost, source: Source<'_>) -> Lookup<Value> {
    let url = match source {
        Source::Discovery(url) | Source::Jwks(url) => url,
    };
    let Some(body) = host.fetch_json(url).await? else {
        return Ok(Err(KeyFailure::unreachable(url)));
    };
    Ok(match source {
        Source::Discovery(url) => match body.get("jwks_uri") {
            Some(Value::String(jwks_uri)) if !jwks_uri.is_empty() => {
                Ok(Value::String(jwks_uri.clone()))
            }
            _ => Err(KeyFailure::NoJwksUri {
                url: url.to_owned(),
            }),
        },
        Source::Jwks(_) => Ok(match body.get("keys") {
            Some(keys) => keys.clone(),
            None => body,
        }),
    })
}

fn ttl(seconds: f64) -> Duration {
    Duration::try_from_secs_f64(seconds).unwrap_or(Duration::ZERO)
}

/// `jwt.get_unverified_header`, nothing verified.
fn header(token: &str) -> Result<Header, TokenError> {
    let header = jsonwebtoken::decode_header(token)?;
    reject_unprocessed_extensions(&header)?;
    Ok(header)
}

/// jsonwebtoken ignores `crit` and `b64`, but RFC 7515 4.1.11 makes a recipient refuse a critical extension it does
/// not process. PyJWT only processes `b64`, and the gateway only while the payload stays encoded.
fn reject_unprocessed_extensions(header: &Header) -> Result<(), TokenError> {
    let b64 = header.extras.inner().get("b64");
    if b64 == Some(&Value::Bool(false)) {
        return Err(TokenError::UnencodedPayload);
    }
    let Some(crit) = &header.crit else {
        return Ok(());
    };
    match crit
        .iter()
        .find(|extension| *extension != "b64" || b64.is_none())
    {
        Some(extension) => Err(TokenError::CritUnsupported(extension.clone())),
        None if crit.is_empty() => Err(TokenError::CritEmpty),
        None => Ok(()),
    }
}

fn leeway(settings: &JwtSettings) -> SignedDuration {
    SignedDuration::from_secs(i64::try_from(settings.leeway_seconds).unwrap_or(i64::MAX))
}

/// The cached `written_at` stays the float of epoch seconds `time.time()` writes.
fn epoch_seconds(time: Timestamp) -> f64 {
    time.as_duration().as_secs_f64()
}

/// The `jwt.decode` options and arguments `_decode_jwt_with_public_key` passes.
struct Checks<'a> {
    /// `None` turns `verify_aud` off.
    audience: Option<&'a [String]>,
    /// `None` turns `verify_iss` off.
    issuer: Option<&'a str>,
    require: &'a [String],
    leeway: SignedDuration,
    now: Timestamp,
}

/// `_decode_jwt_with_public_key`: `PyJWK` from the JWK, then `jwt.decode`.
fn decode(
    token: &str,
    jwk: &Map<String, Value>,
    checks: &Checks<'_>,
) -> Result<Map<String, Value>, TokenError> {
    let (key, validation) = verifier(jwk)?;
    let payload = jsonwebtoken::decode::<Map<String, Value>>(token, &key, &validation)?.claims;
    validate_claims(&payload, checks)?;
    Ok(payload)
}

/// `PyJWT._validate_claims`, in its order. Not `Validation`, which never checks `iat`, and counts an empty `aud` or a
/// `null` required claim as present.
fn validate_claims(payload: &Map<String, Value>, checks: &Checks<'_>) -> Result<(), TokenError> {
    if let Some(claim) = checks
        .require
        .iter()
        .find(|claim| payload.get(*claim).is_none_or(Value::is_null))
    {
        return Err(TokenError::MissingClaim(claim.clone()));
    }
    let date = |claim: &Value, malformed| value::numeric_date(claim).ok_or(malformed);
    if let Some(iat) = payload.get("iat")
        && date(iat, TokenError::IatNotInteger)?.duration_since(checks.now) > checks.leeway
    {
        return Err(TokenError::IssuedInFuture);
    }
    if let Some(nbf) = payload.get("nbf")
        && date(nbf, TokenError::NbfNotInteger)?.duration_since(checks.now) > checks.leeway
    {
        return Err(TokenError::NotYetValid);
    }
    if let Some(exp) = payload.get("exp")
        && checks
            .now
            .duration_since(date(exp, TokenError::ExpNotInteger)?)
            >= checks.leeway
    {
        return Err(TokenError::Expired);
    }
    if let Some(issuer) = checks.issuer {
        match payload.get("iss") {
            None => return Err(TokenError::MissingClaim("iss".to_owned())),
            Some(Value::String(iss)) if iss == issuer => {}
            Some(Value::String(_)) => return Err(TokenError::WrongIssuer),
            Some(_) => return Err(TokenError::IssuerNotString),
        }
    }
    if let Some(audience) = checks.audience {
        validate_audience(payload, audience)?;
    }
    if payload.get("sub").is_some_and(|sub| !sub.is_string()) {
        return Err(TokenError::SubjectNotString);
    }
    if payload.get("jti").is_some_and(|jti| !jti.is_string()) {
        return Err(TokenError::JtiNotString);
    }
    Ok(())
}

fn validate_audience(payload: &Map<String, Value>, audience: &[String]) -> Result<(), TokenError> {
    let claim = payload
        .get("aud")
        .filter(|aud| value::is_set(aud))
        .ok_or_else(|| TokenError::MissingClaim("aud".to_owned()))?;
    let claimed: Vec<&str> = match claim {
        Value::String(aud) => vec![aud],
        Value::Array(auds) => auds
            .iter()
            .map(Value::as_str)
            .collect::<Option<_>>()
            .ok_or(TokenError::AudienceNotStrings)?,
        _ => return Err(TokenError::AudienceNotStrings),
    };
    if audience.iter().any(|aud| claimed.contains(&aud.as_str())) {
        Ok(())
    } else {
        Err(TokenError::WrongAudience)
    }
}

/// jsonwebtoken checks the signature and algorithm, `validate_claims` every claim.
fn signature_only(validation: Validation) -> Validation {
    Validation {
        required_spec_claims: HashSet::new(),
        validate_exp: false,
        validate_aud: false,
        ..validation
    }
}

/// `PyJWK.from_dict(_get_jwk_from_public_key(key))`. Not `Jwk` and `DecodingKey::from_jwk`, which refuse the EC key
/// without `crv` that PyJWT reads as P-256.
fn verifier(key: &Map<String, Value>) -> Result<(DecodingKey, Validation), KeyError> {
    let jwk: Map<String, Value> = JwkField::VARIANTS
        .iter()
        .filter_map(|field| {
            let name: &str = field.into();
            key.get(name).map(|value| (name.to_owned(), value.clone()))
        })
        .collect();
    let member = |field: JwkField| jwk.get(<&str>::from(field));
    let field = |field: JwkField| member(field).and_then(Value::as_str);
    let malformed = |reason: &str| KeyError::Malformed {
        reason: reason.to_owned(),
    };
    let verifying = |key: jsonwebtoken::errors::Result<DecodingKey>, validation| {
        key.map(|key| (key, signature_only(validation)))
            .map_err(|error| KeyError::Malformed {
                reason: error.to_string(),
            })
    };
    let unverifiable = |algorithm| Err(KeyError::Unverifiable { algorithm });
    let kty = member(JwkField::Kty)
        .filter(|kty| value::is_set(kty))
        .ok_or_else(|| KeyError::MissingKty {
            jwk: value::json(&Value::Object(jwk.clone())),
        })?;
    let crv = member(JwkField::Crv).filter(|crv| value::is_set(crv));
    match kty.as_str() {
        Some("RSA") => {
            let (Some(n), Some(e)) = (field(JwkField::N), field(JwkField::E)) else {
                return Err(malformed("Not a public or private key"));
            };
            verifying(
                DecodingKey::from_rsa_components(n, e),
                Validation::new_for_family(AlgorithmFamily::Rsa),
            )
        }
        Some("EC") => {
            let algorithm = match crv.map(Value::as_str) {
                None | Some(Some("P-256")) => Algorithm::ES256,
                Some(Some("P-384")) => Algorithm::ES384,
                Some(Some("P-521")) => return unverifiable("ES512"),
                Some(Some("secp256k1")) => return unverifiable("ES256K"),
                Some(_) => {
                    return Err(KeyError::UnsupportedCrv {
                        crv: crv.map(value::display).unwrap_or_default(),
                    });
                }
            };
            let (Some(x), Some(y)) = (field(JwkField::X), field(JwkField::Y)) else {
                return Err(malformed("Not an Elliptic curve key"));
            };
            verifying(
                DecodingKey::from_ec_components(x, y),
                Validation::new(algorithm),
            )
        }
        Some("OKP") => match crv {
            None => Err(KeyError::MissingCrv {
                jwk: value::json(&Value::Object(jwk.clone())),
            }),
            Some(Value::String(crv)) if crv == "Ed25519" => {
                let x = field(JwkField::X).ok_or_else(|| malformed("Not an Octet Key Pair"))?;
                verifying(
                    DecodingKey::from_ed_components(x),
                    Validation::new(Algorithm::EdDSA),
                )
            }
            Some(crv) => Err(KeyError::UnsupportedCrv {
                crv: value::display(crv),
            }),
        },
        Some("oct") => unverifiable("HS256"),
        _ => Err(KeyError::UnsupportedKty {
            kty: value::display(kty),
        }),
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Mutex};

    use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
    use jsonwebtoken::{EncodingKey, Header, crypto::sign, errors::ErrorKind};
    use litellm_config::{JwtAuthSettings, JwtIssuerConfig, OneOrMany};
    use rstest::{fixture, rstest};
    use serde_json::{Value, json};

    use super::*;
    use crate::{ErrorType, PublicDenial, credential::jwt::JwtDenial, testing::ScriptedHost};

    const NOW: f64 = 1_700_000_000.0;
    const AT: Timestamp = Timestamp::constant(1_700_000_000, 0);
    const JWKS: &str = "https://idp.example/jwks";

    async fn verify(
        host: &impl AuthHost,
        token: &str,
        settings: &JwtSettings,
        now: Timestamp,
    ) -> Result<Claims, Error> {
        super::verify(host, &Token::new(token), settings, now).await
    }

    async fn verify_assertion(
        host: &impl AuthHost,
        token: &str,
        expected: &Expected,
        settings: &JwtSettings,
        now: Timestamp,
    ) -> Result<Claims, Error> {
        super::verify_assertion(host, &Token::new(token), expected, settings, now).await
    }

    async fn oidc_userinfo(
        host: &impl AuthHost,
        token: &str,
        settings: &JwtSettings,
    ) -> Result<Value, Error> {
        super::oidc_userinfo(host, &Token::new(token), settings).await
    }
    const OTHER_JWKS: &str = "https://other.example/jwks";
    const DISCOVERY: &str = "https://idp.example/.well-known/openid-configuration";
    const ISSUER: &str = "https://issuer.example/";
    const ISSUER_DISCOVERY: &str = "https://issuer.example/.well-known/openid-configuration";
    const ISSUER_JWKS: &str = "https://issuer.example/keys";
    const UNREACHABLE: &str = "TransportError fetching https://idp.example/jwks after 3 attempts";

    const RSA_A_PEM: &str = "-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC1F5RKK1x3q7zo
8TBAF+pdG7uCPxyxXrdqAI74SlvxHviSb9D6FBt/SlFQAWNYbPqWEj8lJDMTTdB5
b2YUdnbdD/+ltckqOP7KPuF9GRKQfHBGQ0kXJKf25+qDkAi0KJZxfXD1YctMIxNv
xHhNMheBVYGNIgcDvGeX7uJ9rrrvTcBwaIoLTovnIiSGjWloYYmOx8dCOkxW9Uo8
s7EZ2yKgReyKOcfPxFiNNGZIqn9fs6YdP4uNEzilKzFwPd4r14oA1AfpOYCkargu
fQr148PsbNuNgtUsQWlCBKJJBtDa79Tx9npR7KWqFwTPThgsbzOpg09jidCzA+Xf
+lLkt6fbAgMBAAECggEABNr8KF2aQ0EmYFkwfH9yUrXEJMgGv8C03+biQ8j78cPt
UUEl6WvxmCDze4GAsDsXyunz/yMbKs4qzp63Lh7xApiuYjs0s2jyIpUQsax4QXR8
rtLuBOyq2OVl9coTrv+UQmMPzMcm+W8f/clW77xDQhNQqmqPpUDUxuTqS163Ig8N
Z4lWsMOWsU+2cWZV1MNIQN0d5/53gfrUl0FvOPfRPR+sDGZ+pZGAVlPoArEoc1X3
qXm77z/DG3V+X3J51CSkFd7KEIumcIaYDQ43adeFFfwZLsBD2VjKyrKVOAfPwVmh
4q+VY1BpHC4oprDAxnru7vAYQHIktiClzmdaQrwAGQKBgQD/+0GRwaOzpoRXaLJB
ahB/vFeIiMe3wDD6F4+Mw0nMIVQFqMmpTBiFNcfeXt6kE9Cj6vT+/2eoEZ3VqZMO
BdtKGv0KL9jqanNa1tTb7/2UgthIO0YL7zNephqT+frOSgiUDjKLRIrazhbLgObx
Km0n6YBlKgUMdbXnBTVzWBIzcwKBgQC1Gu9t5HhMBFbWf0gS4OUnOt4tOb7MwoV8
ix6uidgve+w4HbLKFA1BudNi/bT2ca5E+prDk+f5S2vf4EgwoDLZhnrzCkCDW7yq
PpVxYDxxlDEeSDFJTqNs5vApqFNbJU3VNBGftxcvtifHWc4P+rFxoTfk1AmD1Gef
Rmj+CMiv+QKBgAeu0937EN3AfrJc6xS5iTs8POhVKqFiPFYVRPK1nc5pparx0POq
3AWLgq8y673C9n0k/St2IPdTaAhUqZw/IsDLpJKcswb+IO42Cnd7cYH2l2U7mjOb
f1A07hFI7HUqla1T68JwUTaV/usmuEe+vT34bSkcJMCl+8quRldq6kjvAoGBAKo5
EfipeQSCQ5jC89yCtABmjXPWxeaxo0QHgyQsfh1LjiDEkUrtA/TWE8EElDXbbiB5
qEPR3oTsBg3cQh4qRQE+LIwKA88zx6YAwvhYqxdaYbyDwIeVuXffrLaK9yZBbuHJ
WhNTxSBcRxFR8KQGTwhtrxqOzg6x81+6j27OlMsJAoGAfrZt5NWHFr3hyerakbDc
4f0DSfPF+9truEVUSOYBgoZBZYc0IxLdyVM0kRnG8raXjVYKDVQsjLT7F3yPHFWo
0RvZVzesIpAccefMK4Fm2FJwDLwkUujZjQOeQfEf5VNAiS0+Xn2T1yw/PvGsEYos
SDrEV7bM3GyrToGQsCY72+o=
-----END PRIVATE KEY-----
";
    const RSA_A_N: &str = "tReUSitcd6u86PEwQBfqXRu7gj8csV63agCO-Epb8R74km_Q-hQbf0pRUAFjWGz6lhI_JSQzE03QeW9mFHZ23Q__pbXJKjj-yj7hfRkSkHxwRkNJFySn9ufqg5AItCiWcX1w9WHLTCMTb8R4TTIXgVWBjSIHA7xnl-7ifa66703AcGiKC06L5yIkho1paGGJjsfHQjpMVvVKPLOxGdsioEXsijnHz8RYjTRmSKp_X7OmHT-LjRM4pSsxcD3eK9eKANQH6TmApGq4Ln0K9ePD7GzbjYLVLEFpQgSiSQbQ2u_U8fZ6UeylqhcEz04YLG8zqYNPY4nQswPl3_pS5Len2w";

    const RSA_B_PEM: &str = "-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQCPW9Bb38fcKzlK
x4wmg5v6FYFXdPiwkwCTRpFzlnsmGOV+Sb+gcXgELtiuH2yt1sm73SfFc8twQQpm
FhXmmQliDitqgWeJlKmaUvuPMrDiQsmCatsgvT82IHrGxD8jvpBDpzu8X9J+j+YY
L0FJnEGtbeTqLN+xPHjhncEJTHoSRe38w0f+DmHbvCsi+botgkUMZV3E/MnIXzyJ
jAYvvzf67WcXv/FMbxEkbnRSOTkdA+cEMQpILXYUFOaieVq1QMQOVDSj2X8RduEo
p+nBSIDsrhhmjJMUjfiK/N/hYeyS9G6xo/0Gg8PBErTS5OmgXRx8I3ETrs3HqSiG
NJO32l49AgMBAAECggEAIf+Vyfwn42yy0SvC7q1es/AetRWOFx490C+3thB1k0Lr
RkiKNCxE2MUdaSoVtaKPREj+MiM0l6ItK6nINuCLa6BNppwa6HODx5AvjbXpYjeR
n6bk4f5Pu+/2/iWzjbAXeZvzHEkjXBWXbZvjYWg34UADPpA22ssrJeKqUaC8AJIP
MVdwS6F/UAzKn21e6xOR248oPwfdxBFY+1i4e3B2dXzAUG7pEFRqkrH4USkfayUt
W59fSuU27p1w+znI/K7jTB2zeQSFsa4v57yvCA8iQGT0sSGRcDQL5jodCwok5ifv
vcDNorx3zOG1xA8xokG0UroOCesKj3ykHfUt8Mb4pwKBgQDAap8GQbqOnSGeyuCh
weU4LVQIiRl/V/uEfxlaI5kS+Fhj10RaGzjEEKb5mG2Nfi9xJfZXrjb8plqvvjEL
Ct4zSEND0jEgOu1e+JT3EYlNne60e2IrOC08aGgx6VNupRznSGyO6jZByYgPSDDK
Tt6bi8ZnUT1EhQJvjZzcWovVXwKBgQC+uyssHcog+n6aNYaMKafbN65nPel0Svcy
+hbEUrHvcoPrSrUhyIGqZEWpzDHUi97JpIkzHrXeq5P1genYwQKMYDnpKAgST4KW
3ktQwaXpkQ5qpUROHtjB1FwlkgQRGbcMxfa75vwKq5PLORL70v/ymy+S6eoz7H8d
y+TPcAC14wKBgQCtTls1M3ZohTaG44w3pYBjfNA3b+S7Qe+UgzW+blmtMGnMY1N5
JROX5LfAD3589q6cZujeSFfWb682o0NhAsTnw6pyJ5t/wBOUeqkRb4YPlrQtcwQU
kWCNDCzs547yFV1XI3BDuMhtsKsPuKHZVfhCeT6WSCG62WI8P3yNLDDOWQKBgQCZ
c061rLXtuHtZKE06wVzaIz8+svH7bUDbE5hzlTCxPAnkqPN33bBs3UWq6a0BmxLA
Wo9DwKaEIrjkogP5LrgVJmoRKvLrgGrJZMlljJcmhlz58kM1Mt4je+4zD/oTpAPA
1BjBboJDN22t6Bpc3pKrpq0s3uNdqR1QZLWPK0Z3DwKBgEpnleHPy3hSKJ9/c/5B
hMIP8GHpMD974tkx6/BezFczMKc6MbdM4/N5jZ1F5kEBvvGEdM9a4jmumJtVOXb/
MDx8YuRrVg5GKD34zymqjv+GcGbY3sQe8DFVNjKqMWj6dHTfb4YUHlfGbf1VnF4d
QaFtuUT+1uBjA03vqHEQAf0I
-----END PRIVATE KEY-----
";
    const RSA_B_N: &str = "j1vQW9_H3Cs5SseMJoOb-hWBV3T4sJMAk0aRc5Z7Jhjlfkm_oHF4BC7Yrh9srdbJu90nxXPLcEEKZhYV5pkJYg4raoFniZSpmlL7jzKw4kLJgmrbIL0_NiB6xsQ_I76QQ6c7vF_Sfo_mGC9BSZxBrW3k6izfsTx44Z3BCUx6EkXt_MNH_g5h27wrIvm6LYJFDGVdxPzJyF88iYwGL783-u1nF7_xTG8RJG50Ujk5HQPnBDEKSC12FBTmonlatUDEDlQ0o9l_EXbhKKfpwUiA7K4YZoyTFI34ivzf4WHskvRusaP9BoPDwRK00uTpoF0cfCNxE67Nx6kohjSTt9pePQ";

    const EC_PEM: &str = "-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQg6BLo9PryhP9wlgmd
E6XFeYjf7sopcfkLeWqlbU3HdyWhRANCAAR8yVF7bW8YjX/nL8/Gu3B4sbOpYpLs
qXKGoygdhdAWm7F6OHrZrW/OaoIbxnWhBf7gIdgsTFq++4On5fYvjTx8
-----END PRIVATE KEY-----
";
    const EC_X: &str = "fMlRe21vGI1_5y_PxrtweLGzqWKS7KlyhqMoHYXQFps";
    const EC_Y: &str = "sXo4etmtb85qghvGdaEF_uAh2CxMWr77g6fl9i-NPHw";

    const ED_PEM: &str = "-----BEGIN PRIVATE KEY-----
MC4CAQAwBQYDK2VwBCIEIEjkuYWkj6THWjxVRFL36QT1XS2a1KjdV89JMvLev7r2
-----END PRIVATE KEY-----
";
    const ED_X: &str = "iO5qGwVLMzVNKsLll1PGbMPQzI_aTO9tZuoe8AEWCo4";

    #[derive(Clone, Copy)]
    enum Signer {
        RsaA,
        RsaB,
        Ec,
        Ed,
        Secret,
    }

    fn b64(value: &Value) -> String {
        URL_SAFE_NO_PAD.encode(serde_json::to_vec(value).unwrap())
    }

    fn signed(algorithm: Algorithm, signer: Signer, header: Value, claims: Value) -> String {
        let key = match signer {
            Signer::RsaA => EncodingKey::from_rsa_pem(RSA_A_PEM.as_bytes()),
            Signer::RsaB => EncodingKey::from_rsa_pem(RSA_B_PEM.as_bytes()),
            Signer::Ec => EncodingKey::from_ec_pem(EC_PEM.as_bytes()),
            Signer::Ed => EncodingKey::from_ed_pem(ED_PEM.as_bytes()),
            Signer::Secret => Ok(EncodingKey::from_secret(b"shared-secret")),
        }
        .unwrap();
        let input = format!("{}.{}", b64(&header), b64(&claims));
        let signature = sign(input.as_bytes(), &key, algorithm).unwrap();
        format!("{input}.{signature}")
    }

    fn token(algorithm: Algorithm, signer: Signer, kid: Option<&str>, claims: Value) -> String {
        let header = Header {
            kid: kid.map(str::to_owned),
            ..Header::new(algorithm)
        };
        signed(
            algorithm,
            signer,
            serde_json::to_value(header).unwrap(),
            claims,
        )
    }

    fn rs256(kid: Option<&str>, claims: Value) -> String {
        token(Algorithm::RS256, Signer::RsaA, kid, claims)
    }

    fn rsa_jwk(kid: &str, n: &str) -> Value {
        json!({"kty": "RSA", "kid": kid, "n": n, "e": "AQAB", "alg": "RS256", "use": "sig"})
    }

    fn ec_jwk(kid: &str) -> Value {
        json!({"kty": "EC", "kid": kid, "crv": "P-256", "x": EC_X, "y": EC_Y})
    }

    fn ed_jwk(kid: &str) -> Value {
        json!({"kty": "OKP", "kid": kid, "crv": "Ed25519", "x": ED_X})
    }

    fn jwks(keys: &[Value]) -> Option<Value> {
        Some(json!({"keys": keys}))
    }

    fn host(fetched: &[(&str, Option<Value>)]) -> ScriptedHost {
        ScriptedHost {
            fetched: fetched
                .iter()
                .map(|(url, body)| ((*url).to_owned(), body.clone()))
                .collect(),
            ..ScriptedHost::default()
        }
    }

    fn with_cache(host: ScriptedHost, entries: &[(String, Value)]) -> ScriptedHost {
        ScriptedHost {
            json_cache: Mutex::new(entries.iter().cloned().collect::<HashMap<_, _>>()),
            ..host
        }
    }

    fn object(value: Value) -> Map<String, Value> {
        match value {
            Value::Object(object) => object,
            other => panic!("not an object: {other}"),
        }
    }

    fn fetches(host: &ScriptedHost) -> Vec<String> {
        host.calls()
            .into_iter()
            .filter(|call| call.starts_with("fetch "))
            .collect()
    }

    #[fixture]
    fn settings() -> JwtSettings {
        JwtSettings {
            public_key_urls: vec![JWKS.to_owned()],
            ..JwtSettings::default()
        }
    }

    fn issuer_config() -> JwtIssuerConfig {
        JwtIssuerConfig {
            issuer: ISSUER.to_owned(),
            jwks_url: Some(ISSUER_JWKS.to_owned()),
            audience: Some(OneOrMany::One("proxy".to_owned())),
            user_id_jwt_field: Some("profile.id".into()),
            team_ids_jwt_field: Some("groups".into()),
            org_id_jwt_field: Some("org".into()),
            user_email_jwt_field: Some("absent".into()),
            ..JwtIssuerConfig::default()
        }
    }

    fn with_issuer(settings: JwtSettings, issuer: JwtIssuerConfig) -> JwtSettings {
        JwtSettings {
            config: JwtAuthSettings {
                issuers: Some(vec![issuer].into_boxed_slice()),
                ..settings.config
            },
            ..settings
        }
    }

    #[rstest]
    #[case::rs256(Algorithm::RS256, Signer::RsaA, rsa_jwk("k", RSA_A_N), None)]
    #[case::ps384_on_rsa_key(Algorithm::PS384, Signer::RsaA, rsa_jwk("k", RSA_A_N), None)]
    #[case::es256(Algorithm::ES256, Signer::Ec, ec_jwk("k"), None)]
    #[case::eddsa(Algorithm::EdDSA, Signer::Ed, ed_jwk("k"), None)]
    #[case::signed_by_another_key(
        Algorithm::RS256,
        Signer::RsaB,
        rsa_jwk("k", RSA_A_N),
        Some(TokenError::Jwt(ErrorKind::InvalidSignature.into()))
    )]
    #[case::symmetric_algorithm_refused(
        Algorithm::HS256,
        Signer::Secret,
        rsa_jwk("k", RSA_A_N),
        Some(TokenError::Jwt(ErrorKind::InvalidAlgorithm.into()))
    )]
    #[case::ec_token_on_rsa_key(
        Algorithm::ES256,
        Signer::Ec,
        rsa_jwk("k", RSA_A_N),
        Some(TokenError::Jwt(ErrorKind::InvalidAlgorithm.into()))
    )]
    #[tokio::test]
    async fn signature_is_checked_with_the_selected_jwk(
        settings: JwtSettings,
        #[case] algorithm: Algorithm,
        #[case] signer: Signer,
        #[case] jwk: Value,
        #[case] rejection: Option<TokenError>,
    ) {
        let claims = json!({"sub": "user", "exp": NOW + 60.0});
        let token = token(algorithm, signer, Some("k"), claims.clone());
        let host = host(&[(JWKS, jwks(&[jwk]))]);

        let verified = verify(&host, &token, &settings, AT).await;

        let expected = match rejection {
            None => Ok(Claims(object(claims))),
            Some(error) => Err(VerifyDenial::Invalid(error).into()),
        };
        assert_eq!(verified, expected);
    }

    #[rstest]
    #[case::ec_without_crv_is_p256(
        Algorithm::ES256,
        Signer::Ec,
        json!({"kty": "EC", "kid": "k", "x": EC_X, "y": EC_Y}),
        None
    )]
    #[case::p384_key_refuses_es256(
        Algorithm::ES256,
        Signer::Ec,
        json!({"kty": "EC", "kid": "k", "crv": "P-384", "x": EC_X, "y": EC_Y}),
        Some(TokenError::Jwt(ErrorKind::InvalidAlgorithm.into()))
    )]
    #[case::ed_key_refuses_rs256(
        Algorithm::RS256,
        Signer::RsaA,
        ed_jwk("k"),
        Some(TokenError::Jwt(ErrorKind::InvalidAlgorithm.into()))
    )]
    #[case::p521_key(
        Algorithm::ES256,
        Signer::Ec,
        json!({"kty": "EC", "kid": "k", "crv": "P-521", "x": EC_X, "y": EC_Y}),
        Some(KeyError::Unverifiable { algorithm: "ES512" }.into())
    )]
    #[case::secp256k1_key(
        Algorithm::ES256,
        Signer::Ec,
        json!({"kty": "EC", "kid": "k", "crv": "secp256k1", "x": EC_X, "y": EC_Y}),
        Some(KeyError::Unverifiable { algorithm: "ES256K" }.into())
    )]
    #[case::symmetric_key(
        Algorithm::RS256,
        Signer::RsaA,
        json!({"kty": "oct", "kid": "k", "k": "c2VjcmV0"}),
        Some(KeyError::Unverifiable { algorithm: "HS256" }.into())
    )]
    #[case::unknown_curve(
        Algorithm::ES256,
        Signer::Ec,
        json!({"kty": "EC", "kid": "k", "crv": "P-999", "x": EC_X, "y": EC_Y}),
        Some(KeyError::UnsupportedCrv { crv: "P-999".to_owned() }.into())
    )]
    #[case::okp_without_crv(
        Algorithm::EdDSA,
        Signer::Ed,
        json!({"kty": "OKP", "kid": "k", "x": ED_X}),
        Some(KeyError::MissingCrv { jwk: format!(r#"{{"kid":"k","kty":"OKP","x":"{ED_X}"}}"#) }.into())
    )]
    #[case::missing_kty(
        Algorithm::RS256,
        Signer::RsaA,
        json!({"kid": "k", "n": RSA_A_N, "e": "AQAB"}),
        Some(KeyError::MissingKty { jwk: format!(r#"{{"e":"AQAB","kid":"k","n":"{RSA_A_N}"}}"#) }.into())
    )]
    #[case::unknown_kty(
        Algorithm::RS256,
        Signer::RsaA,
        json!({"kty": "AKP", "kid": "k"}),
        Some(KeyError::UnsupportedKty { kty: "AKP".to_owned() }.into())
    )]
    #[case::rsa_without_modulus(
        Algorithm::RS256,
        Signer::RsaA,
        json!({"kty": "RSA", "kid": "k", "e": "AQAB"}),
        Some(KeyError::Malformed { reason: "Not a public or private key".to_owned() }.into())
    )]
    #[tokio::test]
    async fn the_jwk_decides_which_algorithms_verify(
        settings: JwtSettings,
        #[case] algorithm: Algorithm,
        #[case] signer: Signer,
        #[case] jwk: Value,
        #[case] rejection: Option<TokenError>,
    ) {
        let claims = json!({"sub": "user"});
        let host = host(&[(JWKS, jwks(&[jwk]))]);

        let verified = verify(
            &host,
            &token(algorithm, signer, Some("k"), claims.clone()),
            &settings,
            AT,
        )
        .await;

        let expected = match rejection {
            None => Ok(Claims(object(claims))),
            Some(error) => Err(VerifyDenial::Invalid(error).into()),
        };
        assert_eq!(verified, expected);
    }

    #[rstest]
    #[case::expired(json!({"exp": NOW - 1.0}), 0, Err(VerifyDenial::Expired.into()))]
    #[case::expires_this_second(json!({"exp": NOW}), 0, Err(VerifyDenial::Expired.into()))]
    #[case::fraction_truncated_to_now(json!({"exp": NOW + 0.5}), 0, Err(VerifyDenial::Expired.into()))]
    #[case::expired_within_leeway(json!({"exp": NOW - 5.0}), 10, Ok(()))]
    #[case::exp_as_numeric_string(
        json!({"exp": "1700000100"}),
        0,
        Err(VerifyDenial::Invalid(TokenError::ExpNotInteger).into())
    )]
    #[case::exp_as_bool(json!({"exp": true}), 0, Err(VerifyDenial::Invalid(TokenError::ExpNotInteger).into()))]
    #[case::exp_not_integer(json!({"exp": "soon"}), 0, Err(VerifyDenial::Invalid(TokenError::ExpNotInteger).into()))]
    #[case::not_yet_valid(json!({"nbf": NOW + 5.0}), 0, Err(VerifyDenial::Invalid(TokenError::NotYetValid).into()))]
    #[case::not_before_within_leeway(json!({"nbf": NOW + 5.0}), 10, Ok(()))]
    #[case::issued_in_future(json!({"iat": NOW + 5.0}), 0, Err(VerifyDenial::Invalid(TokenError::IssuedInFuture).into()))]
    #[case::iat_not_integer(json!({"iat": [1]}), 0, Err(VerifyDenial::Invalid(TokenError::IatNotInteger).into()))]
    #[tokio::test]
    async fn time_claims_honor_the_leeway(
        settings: JwtSettings,
        #[case] claims: Value,
        #[case] leeway_seconds: u64,
        #[case] expected: Result<(), Error>,
    ) {
        let settings = JwtSettings {
            leeway_seconds,
            ..settings
        };
        let host = host(&[(JWKS, jwks(&[rsa_jwk("k", RSA_A_N)]))]);

        let verified = verify(&host, &rs256(Some("k"), claims), &settings, AT).await;

        assert_eq!(verified.map(|_| ()), expected);
    }

    #[rstest]
    #[case::unscoped_ignores_aud(None, None, json!({"aud": "anyone"}), Ok(()))]
    #[case::audience_in_list(Some("proxy"), None, json!({"aud": ["other", "proxy"]}), Ok(()))]
    #[case::audience_missing(Some("proxy"), None, json!({"sub": "u"}), Err(TokenError::MissingClaim("aud".to_owned())))]
    #[case::audience_empty(Some("proxy"), None, json!({"aud": ""}), Err(TokenError::MissingClaim("aud".to_owned())))]
    #[case::audience_other(Some("proxy"), None, json!({"aud": "other"}), Err(TokenError::WrongAudience))]
    #[case::audience_not_strings(Some("proxy"), None, json!({"aud": [1]}), Err(TokenError::AudienceNotStrings))]
    #[case::issuer_matches(None, Some(ISSUER), json!({"iss": ISSUER}), Ok(()))]
    #[case::issuer_other(None, Some(ISSUER), json!({"iss": "https://evil.example"}), Err(TokenError::WrongIssuer))]
    #[case::issuer_missing(None, Some(ISSUER), json!({}), Err(TokenError::MissingClaim("iss".to_owned())))]
    #[case::issuer_not_string(None, Some(ISSUER), json!({"iss": 7}), Err(TokenError::IssuerNotString))]
    #[case::subject_not_string(None, None, json!({"sub": 1}), Err(TokenError::SubjectNotString))]
    #[case::jti_not_string(None, None, json!({"jti": 1}), Err(TokenError::JtiNotString))]
    #[tokio::test]
    async fn global_audience_and_issuer_come_from_the_environment(
        settings: JwtSettings,
        #[case] audience: Option<&str>,
        #[case] issuer: Option<&str>,
        #[case] claims: Value,
        #[case] expected: Result<(), TokenError>,
    ) {
        let settings = JwtSettings {
            audience: audience.map(str::to_owned),
            issuer: issuer.map(str::to_owned),
            ..settings
        };
        let host = host(&[(JWKS, jwks(&[rsa_jwk("k", RSA_A_N)]))]);

        let verified = verify(&host, &rs256(Some("k"), claims), &settings, AT).await;

        assert_eq!(
            verified.map(|_| ()),
            expected.map_err(|error| VerifyDenial::Invalid(error).into())
        );
    }

    #[rstest]
    #[tokio::test]
    async fn global_path_drops_internal_claims(settings: JwtSettings) {
        let host = host(&[(JWKS, jwks(&[rsa_jwk("k", RSA_A_N)]))]);
        let token = rs256(
            Some("k"),
            json!({"sub": "u", "_litellm_team_id": "forged", "_litellm_jwt_issuer": ISSUER}),
        );

        let verified = verify(&host, &token, &settings, AT).await;

        assert_eq!(verified, Ok(Claims(object(json!({"sub": "u"})))));
    }

    #[rstest]
    #[case::matching_kid_among_many(
        json!({"keys": [rsa_jwk("a", RSA_A_N), rsa_jwk("b", RSA_B_N)]}), Some("b"), Signer::RsaB, Ok(())
    )]
    #[case::bare_key_list(json!([rsa_jwk("a", RSA_A_N), rsa_jwk("b", RSA_B_N)]), Some("a"), Signer::RsaA, Ok(()))]
    #[case::unknown_kid(
        json!({"keys": [rsa_jwk("a", RSA_A_N), rsa_jwk("b", RSA_B_N)]}), Some("c"), Signer::RsaA, Err(Some("c"))
    )]
    #[case::many_keys_no_kid(
        json!({"keys": [rsa_jwk("a", RSA_A_N), rsa_jwk("b", RSA_B_N)]}), None, Signer::RsaA, Err(None)
    )]
    #[case::lone_key_no_kid(json!({"keys": [rsa_jwk("a", RSA_A_N)]}), None, Signer::RsaA, Ok(()))]
    #[case::lone_key_other_kid(json!({"keys": [rsa_jwk("a", RSA_A_N)]}), Some("z"), Signer::RsaA, Err(Some("z")))]
    #[case::single_jwk_object_is_not_a_set(json!(rsa_jwk("a", RSA_A_N)), Some("a"), Signer::RsaA, Err(Some("a")))]
    #[tokio::test]
    async fn the_kid_selects_the_key(
        settings: JwtSettings,
        #[case] body: Value,
        #[case] kid: Option<&str>,
        #[case] signer: Signer,
        #[case] expected: Result<(), Option<&str>>,
    ) {
        let host = host(&[(JWKS, Some(body))]);

        let verified = verify(
            &host,
            &token(Algorithm::RS256, signer, kid, json!({})),
            &settings,
            AT,
        )
        .await;

        let expected = expected.map_err(|kid| {
            VerifyDenial::Key(KeyFailure::NoMatch {
                keys: Searched::Urls(vec![JWKS.to_owned()]),
                kid: kid.map(str::to_owned),
            })
            .into()
        });
        assert_eq!(verified.map(|_| ()), expected);
    }

    #[rstest]
    #[tokio::test]
    async fn later_urls_are_tried_when_earlier_ones_lack_the_kid(settings: JwtSettings) {
        let settings = JwtSettings {
            public_key_urls: vec![format!(" {JWKS}"), String::new(), OTHER_JWKS.to_owned()],
            ..settings
        };
        let host = host(&[
            (JWKS, jwks(&[rsa_jwk("a", RSA_B_N)])),
            (OTHER_JWKS, jwks(&[rsa_jwk("b", RSA_A_N)])),
        ]);

        let verified = verify(&host, &rs256(Some("b"), json!({})), &settings, AT).await;

        assert_eq!(verified.map(|_| ()), Ok(()));
        assert_eq!(
            fetches(&host),
            [format!("fetch {JWKS}"), format!("fetch {OTHER_JWKS}")]
        );
    }

    #[rstest]
    #[tokio::test]
    async fn no_url_holding_the_kid_names_them_all(settings: JwtSettings) {
        let settings = JwtSettings {
            public_key_urls: vec![JWKS.to_owned(), OTHER_JWKS.to_owned()],
            ..settings
        };
        let host = host(&[
            (JWKS, jwks(&[rsa_jwk("a", RSA_A_N)])),
            (OTHER_JWKS, jwks(&[rsa_jwk("b", RSA_A_N)])),
        ]);

        let denial = verify(&host, &rs256(Some("c"), json!({})), &settings, AT)
            .await
            .unwrap_err();

        assert_eq!(
            denial.public(),
            Some(PublicDenial::Plain {
                message: format!(
                    "No matching public key found. keys=[\"{JWKS}\",\"{OTHER_JWKS}\"], kid=c"
                )
            })
        );
    }

    #[rstest]
    #[tokio::test]
    async fn no_public_key_url_is_refused(settings: JwtSettings) {
        let settings = JwtSettings {
            public_key_urls: Vec::new(),
            ..settings
        };
        let host = ScriptedHost::default();

        let denial = verify(&host, &rs256(Some("k"), json!({})), &settings, AT)
            .await
            .unwrap_err();

        assert_eq!(denial, VerifyDenial::Key(KeyFailure::MissingUrl).into());
        assert_eq!(
            denial.public(),
            Some(PublicDenial::Plain {
                message: "Missing JWT Public Key URL from environment.".to_owned()
            })
        );
    }

    #[rstest]
    #[tokio::test]
    async fn discovery_resolves_the_jwks_uri_and_caches_both(settings: JwtSettings) {
        let settings = JwtSettings {
            public_key_urls: vec![DISCOVERY.to_owned()],
            ..settings
        };
        let host = host(&[
            (DISCOVERY, Some(json!({"jwks_uri": JWKS}))),
            (JWKS, jwks(&[rsa_jwk("k", RSA_A_N)])),
        ]);

        let verified = verify(&host, &rs256(Some("k"), json!({})), &settings, AT).await;

        assert_eq!(verified.map(|_| ()), Ok(()));
        let discovery_key = format!("litellm_oidc_discovery_{DISCOVERY}");
        let keys_key = format!("litellm_jwt_auth_keys_{JWKS}");
        assert_eq!(
            host.calls(),
            [
                format!("get {discovery_key}"),
                format!("get litellm_jwks_unreachable_{discovery_key}"),
                format!("fetch {DISCOVERY}"),
                format!("set {discovery_key} 600"),
                format!("set litellm_stale_{discovery_key} 4200"),
                format!("set litellm_stale_written_at_{discovery_key} 4200"),
                format!("get {keys_key}"),
                format!("get litellm_jwks_unreachable_{keys_key}"),
                format!("fetch {JWKS}"),
                format!("set {keys_key} 600"),
                format!("set litellm_stale_{keys_key} 4200"),
                format!("set litellm_stale_written_at_{keys_key} 4200"),
            ]
        );
        let cache = host.json_cache.lock().unwrap();
        assert_eq!(cache.get(&discovery_key), Some(&json!(JWKS)));
        assert_eq!(
            cache.get(&format!("litellm_stale_written_at_{keys_key}")),
            Some(&json!(NOW))
        );
        assert_eq!(cache.get(&keys_key), Some(&json!([rsa_jwk("k", RSA_A_N)])));
    }

    #[rstest]
    #[tokio::test]
    async fn discovery_without_jwks_uri_is_refused(settings: JwtSettings) {
        let settings = JwtSettings {
            public_key_urls: vec![DISCOVERY.to_owned()],
            ..settings
        };
        let host = host(&[(DISCOVERY, Some(json!({"issuer": "x"})))]);

        let denial = verify(&host, &rs256(Some("k"), json!({})), &settings, AT)
            .await
            .unwrap_err();

        assert_eq!(
            denial.public(),
            Some(PublicDenial::Plain {
                message: format!(
                    "JWT Auth: OIDC discovery document at {DISCOVERY} does not contain a 'jwks_uri' field."
                )
            })
        );
    }

    #[rstest]
    #[tokio::test]
    async fn cached_keys_skip_the_fetch(settings: JwtSettings) {
        let keys_key = format!("litellm_jwt_auth_keys_{JWKS}");
        let host = with_cache(
            ScriptedHost::default(),
            &[(keys_key.clone(), json!([rsa_jwk("k", RSA_A_N)]))],
        );

        let verified = verify(&host, &rs256(Some("k"), json!({})), &settings, AT).await;

        assert_eq!(verified.map(|_| ()), Ok(()));
        assert_eq!(host.calls(), [format!("get {keys_key}")]);
    }

    #[rstest]
    #[case::inside_the_window(3600.0, Some(NOW - 4000.0), true)]
    #[case::at_the_window_edge(3600.0, Some(NOW - 4200.0), true)]
    #[case::past_the_window(3600.0, Some(NOW - 4201.0), false)]
    #[case::write_time_lost(3600.0, None, false)]
    #[case::shortened_window_applies_to_old_copies(100.0, Some(NOW - 1000.0), false)]
    #[tokio::test]
    async fn an_outage_serves_a_stale_copy_only_inside_its_window(
        settings: JwtSettings,
        #[case] stale_ttl: f64,
        #[case] written_at: Option<f64>,
        #[case] served: bool,
    ) {
        let settings = JwtSettings {
            config: JwtAuthSettings {
                public_key_stale_ttl: stale_ttl,
                ..settings.config
            },
            ..settings
        };
        let keys_key = format!("litellm_jwt_auth_keys_{JWKS}");
        let stale = (
            format!("litellm_stale_{keys_key}"),
            json!([rsa_jwk("k", RSA_A_N)]),
        );
        let written =
            written_at.map(|at| (format!("litellm_stale_written_at_{keys_key}"), json!(at)));
        let host = with_cache(
            host(&[(JWKS, None)]),
            &[Some(stale), written]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>(),
        );

        let verified = verify(&host, &rs256(Some("k"), json!({})), &settings, AT).await;

        let expected = if served {
            Ok(())
        } else {
            Err(VerifyDenial::Key(KeyFailure::Unreachable {
                error: UNREACHABLE.to_owned(),
            })
            .into())
        };
        assert_eq!(verified.map(|_| ()), expected);
        assert_eq!(
            host.calls(),
            [
                format!("get {keys_key}"),
                format!("get litellm_jwks_unreachable_{keys_key}"),
                format!("fetch {JWKS}"),
                format!("set litellm_jwks_unreachable_{keys_key} 30"),
                format!("get litellm_stale_{keys_key}"),
                format!("get litellm_stale_written_at_{keys_key}"),
            ]
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_zero_stale_window_fails_closed_without_reading_the_copy(settings: JwtSettings) {
        let settings = JwtSettings {
            config: JwtAuthSettings {
                public_key_stale_ttl: 0.0,
                ..settings.config
            },
            ..settings
        };
        let keys_key = format!("litellm_jwt_auth_keys_{JWKS}");
        let host = with_cache(
            host(&[(JWKS, None)]),
            &[
                (
                    format!("litellm_stale_{keys_key}"),
                    json!([rsa_jwk("k", RSA_A_N)]),
                ),
                (format!("litellm_stale_written_at_{keys_key}"), json!(NOW)),
            ],
        );

        let denial = verify(&host, &rs256(Some("k"), json!({})), &settings, AT)
            .await
            .unwrap_err();

        assert_eq!(
            denial,
            VerifyDenial::Key(KeyFailure::Unreachable {
                error: UNREACHABLE.to_owned()
            })
            .into()
        );
        assert_eq!(host.calls().len(), 4);
    }

    #[rstest]
    #[tokio::test]
    async fn a_recent_outage_skips_the_fetch_and_is_a_503(settings: JwtSettings) {
        let keys_key = format!("litellm_jwt_auth_keys_{JWKS}");
        let host = with_cache(
            host(&[]),
            &[(
                format!("litellm_jwks_unreachable_{keys_key}"),
                json!("idp down"),
            )],
        );

        let denial = verify(&host, &rs256(Some("k"), json!({})), &settings, AT)
            .await
            .unwrap_err();

        assert!(fetches(&host).is_empty());
        assert_eq!(
            denial.public(),
            Some(PublicDenial::Proxy {
                status: 503,
                message: "Service Unavailable, the identity provider's JWKS endpoint is temporarily unreachable, so the \
                          JWT signature could not be verified. Please retry shortly. Error: idp down"
                    .to_owned(),
                error_type: ErrorType::AuthProviderUnavailable,
                param: None,
            })
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_failed_fetch_the_host_raised_for_propagates(settings: JwtSettings) {
        let host = host(&[]);

        let verified = verify(&host, &rs256(Some("k"), json!({})), &settings, AT).await;

        assert_eq!(verified, Err(Error::HostFailed));
        assert!(!host.calls().iter().any(|call| call.starts_with("set ")));
    }

    #[rstest]
    #[tokio::test]
    async fn a_configured_issuer_maps_its_claims(settings: JwtSettings) {
        let settings = with_issuer(settings, issuer_config());
        let host = host(&[(ISSUER_JWKS, jwks(&[rsa_jwk("k", RSA_A_N)]))]);
        let token = rs256(
            Some("k"),
            json!({
                "iss": ISSUER, "aud": "proxy", "profile": {"id": "u1"}, "groups": ["t1"], "org": "",
                "_litellm_team_id": "forged",
            }),
        );

        let verified = verify(&host, &token, &settings, AT).await;

        let expected = json!({
            "iss": ISSUER, "aud": "proxy", "profile": {"id": "u1"}, "groups": ["t1"], "org": "",
            "_litellm_jwt_issuer": ISSUER, "_litellm_user_id": "u1", "_litellm_team_ids": ["t1"],
        });
        assert_eq!(verified, Ok(Claims(object(expected))));
        assert_eq!(fetches(&host), [format!("fetch {ISSUER_JWKS}")]);
    }

    #[rstest]
    #[tokio::test]
    async fn a_configured_issuer_checks_its_own_audience(settings: JwtSettings) {
        let settings = with_issuer(
            JwtSettings {
                audience: Some("global".to_owned()),
                ..settings
            },
            issuer_config(),
        );
        let host = host(&[(ISSUER_JWKS, jwks(&[rsa_jwk("k", RSA_A_N)]))]);

        let verified = verify(
            &host,
            &rs256(Some("k"), json!({"iss": ISSUER, "aud": "global"})),
            &settings,
            AT,
        )
        .await;

        assert_eq!(
            verified,
            Err(VerifyDenial::Invalid(TokenError::WrongAudience).into())
        );
    }

    #[rstest]
    #[case::audience_required(false, Err(VerifyDenial::Invalid(TokenError::AudienceRequired).into()))]
    #[case::audience_opted_out(true, Ok(()))]
    #[tokio::test]
    async fn an_issuer_without_audience_needs_the_opt_out(
        settings: JwtSettings,
        #[case] disable_audience_validation: bool,
        #[case] expected: Result<(), Error>,
    ) {
        let issuer = JwtIssuerConfig {
            audience: None,
            disable_audience_validation,
            ..issuer_config()
        };
        let settings = with_issuer(settings, issuer);
        let host = host(&[(ISSUER_JWKS, jwks(&[rsa_jwk("k", RSA_A_N)]))]);

        let verified = verify(
            &host,
            &rs256(Some("k"), json!({"iss": ISSUER, "aud": "anyone"})),
            &settings,
            AT,
        )
        .await;

        assert_eq!(verified.map(|_| ()), expected);
    }

    #[rstest]
    #[tokio::test]
    async fn an_issuer_without_jwks_url_uses_its_discovery_document(settings: JwtSettings) {
        let issuer = JwtIssuerConfig {
            jwks_url: None,
            ..issuer_config()
        };
        let settings = with_issuer(settings, issuer);
        let host = host(&[
            (ISSUER_DISCOVERY, Some(json!({"jwks_uri": ISSUER_JWKS}))),
            (ISSUER_JWKS, jwks(&[rsa_jwk("k", RSA_A_N)])),
        ]);

        let verified = verify(
            &host,
            &rs256(Some("k"), json!({"iss": ISSUER, "aud": "proxy"})),
            &settings,
            AT,
        )
        .await;

        assert_eq!(verified.map(|_| ()), Ok(()));
        assert_eq!(
            fetches(&host),
            [
                format!("fetch {ISSUER_DISCOVERY}"),
                format!("fetch {ISSUER_JWKS}")
            ]
        );
    }

    #[rstest]
    #[tokio::test]
    async fn an_unconfigured_issuer_takes_the_global_path(settings: JwtSettings) {
        let settings = with_issuer(settings, issuer_config());
        let host = host(&[(JWKS, jwks(&[rsa_jwk("k", RSA_A_N)]))]);

        let verified = verify(
            &host,
            &rs256(Some("k"), json!({"iss": "https://elsewhere.example"})),
            &settings,
            AT,
        )
        .await;

        assert_eq!(verified.map(|_| ()), Ok(()));
        assert_eq!(fetches(&host), [format!("fetch {JWKS}")]);
    }

    fn raw_token(header: &str, payload: &str) -> String {
        format!("{header}.{payload}.AAAA")
    }

    #[rstest]
    #[case::not_base64("a.b.c")]
    #[case::two_segments("e30.e30")]
    #[case::padded_header(&format!("{}=.e30.AAAA", b64(&json!({"alg": "RS256"}))))]
    #[case::header_not_object(&raw_token(&b64(&json!([1])), "e30"))]
    #[case::no_alg(&raw_token(&b64(&json!({"typ": "JWT"})), "e30"))]
    #[case::unknown_alg(&raw_token(&b64(&json!({"alg": "none"})), "e30"))]
    #[case::kid_not_string(&raw_token(&b64(&json!({"alg": "RS256", "kid": 1})), "e30"))]
    #[tokio::test]
    async fn an_unreadable_header_is_refused_before_any_lookup(
        settings: JwtSettings,
        #[case] token: &str,
    ) {
        let host = ScriptedHost::default();

        let verified = verify(&host, token, &settings, AT).await;

        assert!(
            matches!(
                verified,
                Err(Error::Jwt(JwtDenial::Verify(VerifyDenial::Malformed(
                    TokenError::Jwt(_)
                ))))
            ),
            "{verified:?}"
        );
        assert!(host.calls().is_empty());
    }

    #[rstest]
    #[case::unsupported_crit(json!({"alg": "RS256", "crit": ["exp"], "exp": 1}), TokenError::CritUnsupported("exp".to_owned()))]
    #[case::crit_without_its_header(json!({"alg": "RS256", "crit": ["b64"]}), TokenError::CritUnsupported("b64".to_owned()))]
    #[case::empty_crit(json!({"alg": "RS256", "crit": []}), TokenError::CritEmpty)]
    #[case::unencoded_payload(json!({"alg": "RS256", "b64": false, "crit": ["b64"]}), TokenError::UnencodedPayload)]
    #[case::unencoded_payload_not_critical(json!({"alg": "RS256", "b64": false}), TokenError::UnencodedPayload)]
    #[tokio::test]
    async fn an_unprocessed_extension_is_refused_before_any_lookup(
        settings: JwtSettings,
        #[case] header: Value,
        #[case] expected: TokenError,
    ) {
        let host = ScriptedHost::default();

        let verified = verify(&host, &raw_token(&b64(&header), "e30"), &settings, AT).await;

        assert_eq!(verified, Err(VerifyDenial::Malformed(expected).into()));
        assert!(host.calls().is_empty());
    }

    #[rstest]
    #[tokio::test]
    async fn an_encoded_payload_may_list_b64_as_critical(settings: JwtSettings) {
        let claims = json!({"sub": "user"});
        let header = json!({"alg": "RS256", "kid": "k", "b64": true, "crit": ["b64"]});
        let token = signed(Algorithm::RS256, Signer::RsaA, header, claims.clone());
        let host = host(&[(JWKS, jwks(&[rsa_jwk("k", RSA_A_N)]))]);

        let verified = verify(&host, &token, &settings, AT).await;

        assert_eq!(verified, Ok(Claims(object(claims))));
    }

    #[rstest]
    #[tokio::test]
    async fn an_id_jag_is_refused_before_any_lookup(settings: JwtSettings) {
        let host = ScriptedHost::default();
        let token = raw_token(
            &b64(&json!({"alg": "RS256", "typ": "oauth-id-jag+jwt"})),
            "e30",
        );

        let verified = verify(&host, &token, &settings, AT).await;

        assert_eq!(verified, Err(VerifyDenial::IdJagNotBearer.into()));
        assert!(host.calls().is_empty());
    }

    #[rstest]
    #[case::signed(&rs256(None, json!({"iss": ISSUER})), Some(json!({"iss": ISSUER})))]
    #[case::bad_signature_still_read(&raw_token(&b64(&json!({"alg": "RS256"})), &b64(&json!({"a": 1}))), Some(json!({"a": 1})))]
    #[case::two_segments("e30.e30", None)]
    #[case::payload_not_object(&raw_token(&b64(&json!({"alg": "RS256"})), &b64(&json!([1]))), None)]
    #[case::detached_payload(&raw_token(&b64(&json!({"alg": "RS256", "b64": false, "crit": ["b64"]})), ""), None)]
    fn unverified_claims_read_the_payload(#[case] token: &str, #[case] expected: Option<Value>) {
        assert_eq!(
            unverified_claims(token),
            expected.map(|claims| Claims(object(claims)))
        );
    }

    #[rstest]
    #[case::same_claims(json!({"sub": "u", "n": 1}), json!({"sub": "u", "n": 1}), true)]
    #[case::internal_claims_ignored(
        json!({"sub": "u"}),
        json!({"sub": "u", "_litellm_jwt_issuer": ISSUER, "_litellm_user_id": "u", "_litellm_end_user_id": "e"}),
        true
    )]
    #[case::extra_presented_claim(json!({"sub": "u", "extra": 1}), json!({"sub": "u"}), true)]
    #[case::different_value(json!({"sub": "other"}), json!({"sub": "u"}), false)]
    #[case::missing_claim(json!({}), json!({"sub": "u"}), false)]
    #[case::forged_internal_claim_not_enough(json!({"_litellm_user_id": "u"}), json!({"sub": "u"}), false)]
    #[case::null_matches_missing(json!({}), json!({"sub": null}), true)]
    fn presented_token_carries_the_verified_claims(
        #[case] presented: Value,
        #[case] verified: Value,
        #[case] expected: bool,
    ) {
        let token = raw_token(&b64(&json!({"alg": "RS256"})), &b64(&presented));
        assert_eq!(presents_claims(&token, &Claims(object(verified))), expected);
    }

    #[rstest]
    fn a_value_that_is_not_a_jwt_presents_no_claims() {
        assert!(!presents_claims("sk-1234", &Claims(Map::new())));
    }

    #[rstest]
    #[case::issuer_with_audience(json!(ISSUER), Some(OneOrMany::One("proxy".to_owned())), false, true)]
    #[case::issuer_with_empty_audience(json!(ISSUER), Some(OneOrMany::Many(Box::new([]))), false, false)]
    #[case::issuer_audience_opted_out(json!(ISSUER), None, true, false)]
    #[case::env_issuer_with_env_audience(json!("https://env.example"), None, true, true)]
    #[case::unknown_issuer(json!("https://evil.example"), None, true, false)]
    #[case::issuer_not_string(json!(1), None, true, false)]
    fn managed_issuers_need_an_audience(
        settings: JwtSettings,
        #[case] issuer: Value,
        #[case] audience: Option<OneOrMany>,
        #[case] disable_audience_validation: bool,
        #[case] trusted: bool,
    ) {
        let config = JwtIssuerConfig {
            audience,
            disable_audience_validation,
            ..issuer_config()
        };
        let settings = with_issuer(
            JwtSettings {
                issuer: Some("https://env.example".to_owned()),
                audience: Some("proxy".to_owned()),
                ..settings
            },
            config,
        );

        assert_eq!(managed_issuer_is_trusted(&issuer, &settings), trusted);
    }

    fn assertion_settings(settings: JwtSettings) -> JwtSettings {
        JwtSettings {
            issuer: Some(ISSUER.to_owned()),
            ..settings
        }
    }

    fn expected_assertion() -> Expected {
        Expected {
            audience: vec![
                "https://gw.example/mcp".to_owned(),
                "https://gw.example".to_owned(),
            ],
            required_claims: vec!["jti".to_owned(), "exp".to_owned()],
        }
    }

    fn assertion(claims: Value) -> String {
        let base =
            json!({"iss": ISSUER, "aud": "https://gw.example", "jti": "j1", "exp": NOW + 60.0});
        let merged: Map<String, Value> = object(base).into_iter().chain(object(claims)).collect();
        rs256(Some("k"), Value::Object(merged))
    }

    #[rstest]
    #[case::accepted(json!({"_litellm_user_id": "forged"}), Ok(()))]
    #[case::missing_required_claim(
        json!({"jti": null}),
        Err(AssertionRejection::Invalid(TokenError::MissingClaim("jti".to_owned())))
    )]
    #[case::other_audience(json!({"aud": "https://api.example"}), Err(AssertionRejection::Invalid(TokenError::WrongAudience)))]
    #[case::expired_is_a_rejection(json!({"exp": NOW - 1.0}), Err(AssertionRejection::Invalid(TokenError::Expired)))]
    #[case::untrusted_issuer(json!({"iss": "https://evil.example"}), Err(AssertionRejection::UntrustedIssuer))]
    #[tokio::test]
    async fn assertions_are_checked_against_the_callers_audience(
        settings: JwtSettings,
        #[case] claims: Value,
        #[case] expected: Result<(), AssertionRejection>,
    ) {
        let host = host(&[(JWKS, jwks(&[rsa_jwk("k", RSA_A_N)]))]);

        let verified = verify_assertion(
            &host,
            &assertion(claims),
            &expected_assertion(),
            &assertion_settings(settings),
            AT,
        )
        .await;

        assert_eq!(
            verified.map(|claims| claims.0.contains_key("_litellm_user_id")),
            expected
                .map(|()| false)
                .map_err(|rejection| VerifyDenial::Assertion(rejection).into())
        );
    }

    #[rstest]
    #[tokio::test]
    async fn an_assertion_without_a_signing_key_is_rejected(settings: JwtSettings) {
        let host = host(&[(
            JWKS,
            jwks(&[rsa_jwk("other", RSA_A_N), rsa_jwk("more", RSA_A_N)]),
        )]);

        let verified = verify_assertion(
            &host,
            &assertion(json!({})),
            &expected_assertion(),
            &assertion_settings(settings),
            AT,
        )
        .await;

        assert_eq!(
            verified,
            Err(
                VerifyDenial::Assertion(AssertionRejection::NoSigningKey(KeyFailure::NoMatch {
                    keys: Searched::Urls(vec![JWKS.to_owned()]),
                    kid: Some("k".to_owned()),
                }))
                .into()
            )
        );
    }

    #[rstest]
    #[tokio::test]
    async fn an_assertion_during_an_outage_is_the_503(settings: JwtSettings) {
        let host = host(&[(JWKS, None)]);

        let verified = verify_assertion(
            &host,
            &assertion(json!({})),
            &expected_assertion(),
            &assertion_settings(settings),
            AT,
        )
        .await;

        assert_eq!(
            verified,
            Err(VerifyDenial::Key(KeyFailure::Unreachable {
                error: UNREACHABLE.to_owned()
            })
            .into())
        );
    }

    #[rstest]
    #[case::not_configured(None, false, Ok(()), false)]
    #[case::passes(Some("hooks.validate"), true, Ok(()), true)]
    #[case::fails(
        Some("hooks.validate"),
        false,
        Err(AssertionRejection::CustomValidate),
        true
    )]
    #[tokio::test]
    async fn custom_validate_runs_on_verified_assertions_when_configured(
        settings: JwtSettings,
        #[case] custom_validate: Option<&str>,
        #[case] passes: bool,
        #[case] expected: Result<(), AssertionRejection>,
        #[case] called: bool,
    ) {
        let settings = JwtSettings {
            config: JwtAuthSettings {
                custom_validate: custom_validate.map(str::to_owned),
                ..settings.config.clone()
            },
            ..assertion_settings(settings)
        };
        let host = ScriptedHost {
            custom_validate_passes: passes,
            ..host(&[(JWKS, jwks(&[rsa_jwk("k", RSA_A_N)]))])
        };

        let verified = verify_assertion(
            &host,
            &assertion(json!({})),
            &expected_assertion(),
            &settings,
            AT,
        )
        .await;

        assert_eq!(
            verified.map(|_| ()),
            expected.map_err(|rejection| VerifyDenial::Assertion(rejection).into())
        );
        assert_eq!(host.calls().contains(&"custom_validate".to_owned()), called);
    }

    const USERINFO: &str = "https://idp.example/userinfo";
    const USERINFO_KEY: &str =
        "oidc_userinfo_84d3f23da9b5f51b3269566eff05d3fb23607eeef89567f9cd280b90ca0dbc5c";

    fn userinfo_settings(settings: JwtSettings, endpoint: Option<&str>) -> JwtSettings {
        JwtSettings {
            config: JwtAuthSettings {
                oidc_userinfo_endpoint: endpoint.map(str::to_owned),
                ..settings.config
            },
            ..settings
        }
    }

    #[rstest]
    #[case::unset(None)]
    #[case::empty(Some(""))]
    #[tokio::test]
    async fn userinfo_needs_an_endpoint(settings: JwtSettings, #[case] endpoint: Option<&str>) {
        let host = ScriptedHost::default();

        let userinfo = oidc_userinfo(
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

        let first = oidc_userinfo(&host, "opaque-token", &settings).await;
        let second = oidc_userinfo(&host, "opaque-token", &settings).await;

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
