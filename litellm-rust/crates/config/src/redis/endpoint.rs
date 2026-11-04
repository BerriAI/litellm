use litellm_redis_types::{
    ClusterEndpoint, ClusterNode, Endpoint, HostEndpoint, RedisAuth, SentinelEndpoint, TlsOptions,
};
use litellm_secret_value::SecretValue;

use super::{
    Expected, RedisError,
    layers::Entry,
    options::{Options, as_text, flag, integer, invalid, secret, text},
    tls,
};
use crate::Value;

/// The target, first match wins like `get_redis_client`: cluster, url, Sentinel, then host.
pub(super) fn endpoint(options: &Options, auth: &RedisAuth) -> Result<Endpoint, RedisError> {
    if let Some(nodes) = cluster_nodes(options.startup_nodes.as_ref())? {
        reject_cluster_db(options.db.as_ref())?;
        return Ok(Endpoint::Cluster(ClusterEndpoint {
            nodes,
            tls: tls_for(options, auth)?,
        }));
    }
    if let Some(url) = text(options.url.as_ref()) {
        return Ok(Endpoint::Url {
            tls: tls::url_tls(&url)?,
            url: SecretValue::new(match auth {
                RedisAuth::Static { .. } => url,
                _ => without_userinfo(&url),
            }),
        });
    }
    if let Some(sentinels) = options.sentinel.nodes.as_ref() {
        return sentinel(options, auth, sentinels).map(Endpoint::Sentinel);
    }
    Ok(Endpoint::Host(HostEndpoint {
        host: text(options.host.as_ref()).ok_or(RedisError::NoTarget)?,
        port: integer(options.port.as_ref())?.unwrap_or(6379),
        tls: tls_for(options, auth)?,
        db: db(options)?,
    }))
}

fn tls_for(options: &Options, auth: &RedisAuth) -> Result<Option<TlsOptions>, RedisError> {
    if !flag(options.ssl.as_ref())? {
        return Ok(None);
    }
    let gcp_ca_certs = match auth {
        RedisAuth::GcpIam { .. } => options.gcp.ssl_ca_certs.as_ref(),
        _ => None,
    };
    tls::tls_options(&options.tls, gcp_ca_certs).map(Some)
}

fn db(options: &Options) -> Result<i64, RedisError> {
    integer(options.db.as_ref()).map(|db| db.unwrap_or(0))
}

fn sentinel(
    options: &Options,
    auth: &RedisAuth,
    sentinels: &Entry,
) -> Result<SentinelEndpoint, RedisError> {
    let nodes = sentinel_nodes(&sentinels.value)
        .ok_or_else(|| invalid(&sentinels.source, Expected::SentinelNodes))?;
    let service_name =
        text(options.sentinel.service_name.as_ref()).ok_or(RedisError::NoSentinelService)?;
    // redis.asyncio.Sentinel passes the connection kwargs to each monitor's Redis(host=..., port=...), so a
    // host or port setting raises TypeError when litellm builds the client
    if let Some(entry) = options.host.as_ref().or(options.port.as_ref()) {
        return Err(RedisError::SentinelWithHost {
            setting: entry.source.clone(),
        });
    }
    Ok(SentinelEndpoint {
        sentinels: nodes,
        service_name,
        sentinel_password: secret(options.sentinel.password.as_ref()),
        tls: tls_for(options, auth)?,
        db: db(options)?,
    })
}

fn cluster_nodes(entry: Option<&Entry>) -> Result<Option<Vec<ClusterNode>>, RedisError> {
    let Some(entry) = entry else {
        return Ok(None);
    };
    let nodes = nodes(&entry.value, |node| {
        node_address(
            node.get("host").and_then(as_text),
            node.get("port").and_then(as_text),
        )
    })
    .ok_or_else(|| invalid(&entry.source, Expected::StartupNodes))?;
    Ok((!nodes.is_empty()).then_some(nodes))
}

fn sentinel_nodes(value: &Value) -> Option<Vec<ClusterNode>> {
    let nodes = nodes(value, |pair| match pair {
        Value::Sequence(pair) if pair.len() == 2 => {
            node_address(as_text(&pair[0]), as_text(&pair[1]))
        }
        _ => None,
    })?;
    (!nodes.is_empty()).then_some(nodes)
}

fn nodes(value: &Value, parse: impl Fn(&Value) -> Option<ClusterNode>) -> Option<Vec<ClusterNode>> {
    match value {
        Value::String(json) => serde_json::from_str::<Vec<Value>>(json)
            .ok()?
            .iter()
            .map(parse)
            .collect(),
        Value::Sequence(nodes) => nodes.iter().map(parse).collect(),
        _ => None,
    }
}

fn node_address(host: Option<String>, port: Option<String>) -> Option<ClusterNode> {
    Some(ClusterNode {
        host: host?,
        port: port?.parse().ok()?,
    })
}

// redis-py's RedisCluster raises unless db is 0 or None
fn reject_cluster_db(db: Option<&Entry>) -> Result<(), RedisError> {
    let Some(entry) = db else {
        return Ok(());
    };
    let is_zero = match &entry.value {
        Value::String(_) => matches!(integer::<i64>(Some(entry)), Ok(None | Some(0))),
        Value::Bool(flag) => !flag,
        Value::Number(number) => number.as_f64() == Some(0.0),
        _ => false,
    };
    match is_zero {
        true => Ok(()),
        false => Err(RedisError::DbInCluster {
            setting: entry.source.clone(),
        }),
    }
}

// _get_redis_url_kwargs keeps only plain Connection kwargs, so redis-py drops every ssl_* setting for a url
// and a credential provider replaces the url's userinfo, which _url_without_userinfo strips
fn without_userinfo(url: &str) -> String {
    let Some((scheme, rest)) = url.split_once("://") else {
        return url.to_owned();
    };
    let (authority, tail) = rest.split_at(rest.find(['/', '?', '#']).unwrap_or(rest.len()));
    let host = authority
        .rsplit_once('@')
        .map_or(authority, |(_, host)| host);
    format!("{scheme}://{host}{tail}")
}
