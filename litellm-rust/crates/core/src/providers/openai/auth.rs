use crate::Error;

pub fn resolve_api_key(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
    route: &str,
) -> Result<String, Error> {
    api_key.map(str::trim).filter(|value| !value.is_empty()).map(str::to_string)
        .or_else(|| env_lookup("OPENAI_API_KEY").filter(|value| !value.trim().is_empty()))
        .ok_or_else(|| Error::Auth(format!("Missing OpenAI API Key - a {route} call is being made but no key was passed via params or the OPENAI_API_KEY environment variable")))
}
