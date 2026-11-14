use litellm_redis_types::{
    Auth, Cluster, Node, RedisUrl, Sentinel, Standalone, StaticAuth, Target, TlsOptions, UnixSocket,
};
use litellm_secret_value::SecretValue;

use serde_json::Value;

use crate::{
    Entry, Error, Expected,
    options::{Options, as_text, flag, integer, invalid, secret, text},
    tls::{self, QuerySettings},
};

/// The target, first match wins like `get_redis_client`: cluster, url, Sentinel, then host. A
/// url also returns the settings its query carries.
pub(super) fn target(options: &Options, auth: &Auth) -> Result<(Target, QuerySettings), Error> {
    if let Some(nodes) = cluster_nodes(options.startup_nodes.as_ref())? {
        reject_cluster_db(options.db.as_ref())?;
        let cluster = Cluster {
            nodes,
            tls: tls_for(options, auth)?,
        };
        return Ok((Target::Cluster(cluster), QuerySettings::default()));
    }
    if let Some(url) = text(options.url.as_ref()) {
        let query = tls::url_query(&url)?;
        let url = RedisUrl {
            url: SecretValue::new(match auth {
                Auth::Static(_) => url,
                _ => without_userinfo(&url),
            }),
            tls: query.tls,
            db: query.db,
            username: query.username,
            password: query.password,
        };
        return Ok((Target::Url(url), query.settings));
    }
    if let Some(sentinels) = options.sentinel.nodes.as_ref() {
        let sentinel = sentinel(options, auth, sentinels)?;
        return Ok((Target::Sentinel(sentinel), QuerySettings::default()));
    }
    let host = text(options.host.as_ref()).ok_or(Error::NoTarget)?;
    let target = match text(options.unix_socket_path.as_ref()) {
        Some(path) => Target::Unix(UnixSocket {
            path: path.into(),
            db: db(options)?,
        }),
        None => Target::Standalone(Standalone {
            host,
            port: integer(options.port.as_ref())?.unwrap_or(6379),
            tls: tls_for(options, auth)?,
            db: db(options)?,
        }),
    };
    Ok((target, QuerySettings::default()))
}

fn tls_for(options: &Options, auth: &Auth) -> Result<Option<TlsOptions>, Error> {
    if !flag(options.ssl.as_ref())? {
        return Ok(None);
    }
    let gcp_ca_certs = match auth {
        Auth::GcpIam { .. } => options.gcp.ssl_ca_certs.as_ref(),
        _ => None,
    };
    tls::tls_options(&options.tls, gcp_ca_certs).map(Some)
}

fn db(options: &Options) -> Result<i64, Error> {
    integer(options.db.as_ref()).map(|db| db.unwrap_or(0))
}

fn sentinel(options: &Options, auth: &Auth, sentinels: &Entry) -> Result<Sentinel, Error> {
    let monitors = sentinel_nodes(&sentinels.value)
        .ok_or_else(|| invalid(&sentinels.source, Expected::SentinelNodes))?;
    let service_name =
        text(options.sentinel.service_name.as_ref()).ok_or(Error::NoSentinelService)?;
    // redis.asyncio.Sentinel passes the connection kwargs to each monitor's Redis(host=..., port=...), so a
    // host or port setting raises TypeError when litellm builds the client
    if let Some(entry) = options.host.as_ref().or(options.port.as_ref()) {
        return Err(Error::SentinelWithHost {
            setting: entry.source.clone(),
        });
    }
    Ok(Sentinel {
        monitors,
        service_name,
        monitor_auth: monitor_auth(options, auth),
        tls: tls_for(options, auth)?,
        db: db(options)?,
    })
}

// _sentinel_auth_kwargs gives the monitors the data node's connection kwargs with sentinel_password as the
// password, so a static username reaches them too
fn monitor_auth(options: &Options, auth: &Auth) -> Option<StaticAuth> {
    let password = secret(options.sentinel.password.as_ref())?;
    let username = match auth {
        Auth::Static(StaticAuth { username, .. }) => username.clone(),
        _ => None,
    };
    Some(StaticAuth {
        username,
        password: Some(password),
    })
}

fn cluster_nodes(entry: Option<&Entry>) -> Result<Option<Vec<Node>>, Error> {
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

fn sentinel_nodes(value: &Value) -> Option<Vec<Node>> {
    let nodes = nodes(value, |pair| match pair {
        Value::Array(pair) if pair.len() == 2 => node_address(as_text(&pair[0]), as_text(&pair[1])),
        _ => None,
    })?;
    (!nodes.is_empty()).then_some(nodes)
}

fn nodes(value: &Value, parse: impl Fn(&Value) -> Option<Node>) -> Option<Vec<Node>> {
    match value {
        Value::String(json) => serde_json::from_str::<Vec<Value>>(json)
            .ok()?
            .iter()
            .map(parse)
            .collect(),
        Value::Array(nodes) => nodes.iter().map(parse).collect(),
        _ => None,
    }
}

fn node_address(host: Option<String>, port: Option<String>) -> Option<Node> {
    Some(Node {
        host: host?,
        port: port?.parse().ok()?,
    })
}

// redis-py's RedisCluster raises unless db is 0 or None
fn reject_cluster_db(db: Option<&Entry>) -> Result<(), Error> {
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
        false => Err(Error::DbInCluster {
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
