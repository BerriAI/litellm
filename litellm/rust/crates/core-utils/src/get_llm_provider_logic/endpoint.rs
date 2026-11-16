use super::secrets::Deferred;

const USES_PARAMS: &[&str] = &[
    "", "ftp", "hdl", "prospero", "http", "imap", "https", "shttp", "rtsp", "rtsps", "rtspu",
    "sip", "sips", "mms", "sftp", "tel",
];

struct Parsed<'a> {
    host: String,
    path: &'a str,
}

fn is_scheme_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.')
}

fn parse(value: &str) -> Result<Parsed<'_>, Deferred> {
    if !value
        .chars()
        .all(|c| ('\u{21}'..='\u{7e}').contains(&c) && c != '[' && c != ']')
    {
        return Err(Deferred);
    }
    let (scheme, rest) = match value.find(':') {
        Some(index)
            if index > 0
                && value.starts_with(|c: char| c.is_ascii_alphabetic())
                && value[..index].chars().all(is_scheme_char) =>
        {
            (value[..index].to_ascii_lowercase(), &value[index + 1..])
        }
        _ => (String::new(), value),
    };
    let (netloc, rest) = match rest.strip_prefix("//") {
        Some(after) => {
            let end = after.find(['/', '?', '#']).unwrap_or(after.len());
            (&after[..end], &after[end..])
        }
        None => ("", rest),
    };
    let rest = &rest[..rest.find(['?', '#']).unwrap_or(rest.len())];
    let path = if USES_PARAMS.contains(&scheme.as_str()) && rest.contains(';') {
        let start = rest.rfind('/').unwrap_or(0);
        match rest[start..].find(';') {
            Some(offset) if rest.contains('/') => &rest[..start + offset],
            None if rest.contains('/') => rest,
            _ => &rest[..rest.find(';').unwrap_or(rest.len())],
        }
    } else {
        rest
    };
    let hostinfo = netloc.rsplit_once('@').map_or(netloc, |(_, host)| host);
    let hostname = hostinfo.split_once(':').map_or(hostinfo, |(host, _)| host);
    Ok(Parsed {
        host: hostname.to_ascii_lowercase(),
        path,
    })
}

fn normalized(value: &str) -> String {
    if value.contains("://") {
        value.to_owned()
    } else {
        format!("https://{value}")
    }
}

pub fn endpoint_matches_api_base(endpoint: &str, api_base: &str) -> Result<bool, Deferred> {
    let endpoint = normalized(endpoint);
    let api_base = normalized(api_base);
    let endpoint = parse(&endpoint)?;
    let url = parse(&api_base)?;
    if endpoint.host.is_empty() || endpoint.host != url.host {
        return Ok(false);
    }
    let endpoint_path = endpoint.path.trim_end_matches('/');
    if endpoint_path.is_empty() {
        return Ok(true);
    }
    let url_path = url.path.trim_end_matches('/');
    Ok(url_path == endpoint_path
        || url_path
            .strip_prefix(endpoint_path)
            .is_some_and(|rest| rest.starts_with('/')))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::bare_host("api.perplexity.ai", "https://api.perplexity.ai/v1", Ok(true))]
    #[case::path_prefix(
        "api.groq.com/openai/v1",
        "https://api.groq.com/openai/v1/chat",
        Ok(true)
    )]
    #[case::path_segment_boundary(
        "api.groq.com/openai/v1",
        "https://api.groq.com/openai/v10",
        Ok(false)
    )]
    #[case::attacker_host(
        "api.groq.com/openai/v1",
        "https://attacker.com/api.groq.com/openai/v1",
        Ok(false)
    )]
    #[case::case_insensitive_host(
        "api.groq.com/openai/v1",
        "HTTPS://API.GROQ.COM/openai/v1",
        Ok(true)
    )]
    #[case::userinfo_and_port(
        "api.groq.com/openai/v1",
        "https://u:p@api.groq.com:443/openai/v1",
        Ok(true)
    )]
    #[case::params_split("api.groq.com/openai/v1", "https://api.groq.com/openai/v1;x", Ok(true))]
    #[case::query("https://api.cerebras.ai/v1", "api.cerebras.ai/v1?x=/y", Ok(true))]
    #[case::bracketed_host_defers("api.groq.com", "https://[::1]/", Err(Deferred))]
    #[case::whitespace_defers("api.groq.com", " https://api.groq.com", Err(Deferred))]
    fn matches_like_urlparse(
        #[case] endpoint: &str,
        #[case] api_base: &str,
        #[case] expected: Result<bool, Deferred>,
    ) {
        assert_eq!(endpoint_matches_api_base(endpoint, api_base), expected);
    }
}
