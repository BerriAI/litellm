pub mod audio_transcription;
pub mod chat;
pub mod messages;

pub(crate) fn operation_url(
    endpoint: Option<&str>,
    region: &str,
    model_id: &str,
    operation: &str,
) -> Result<url::Url, crate::Error> {
    let base = match endpoint {
        Some(endpoint) => litellm_core_utils::url_utils::ApiUrl::parse(endpoint)?,
        None => {
            if region.is_empty()
                || !region
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
            {
                return Err(crate::Error::InvalidRequest("invalid AWS region".into()));
            }
            let mut url =
                url::Url::parse("https://bedrock-runtime.amazonaws.com").expect("static URL");
            url.set_host(Some(&format!("bedrock-runtime.{region}.amazonaws.com")))
                .map_err(|error| {
                    crate::Error::InvalidRequest(crate::ErrorDetail::invalid("AWS region", error))
                })?;
            litellm_core_utils::url_utils::ApiUrl::from_url(url)?
        }
    };
    let parsed = base.complete_path(&[])?.into_url();
    let segments: Vec<_> = parsed.path_segments().expect("validated URL").collect();
    if segments.len() >= 3
        && segments[segments.len() - 3] == "model"
        && segments.last() == Some(&operation)
    {
        return Ok(parsed);
    }
    Ok(litellm_core_utils::url_utils::ApiUrl::from_url(parsed)?
        .append_path(&["model", model_id, operation])?
        .into_url())
}
