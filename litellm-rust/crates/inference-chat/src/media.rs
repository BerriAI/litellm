//! The media a translated request needs downloaded first: an image or file URL the provider
//! cannot take by reference, which Python fetches while it builds the request.

use std::{collections::HashMap, time::Duration};

use base64::Engine;
use futures_util::future::try_join_all;
use litellm_http::media::DownloadPolicy;
use litellm_llms::base_llm::chat::media::{
    MAX_MEDIA_DOWNLOAD_BYTES, downloaded_media_type, inline_media,
};

use crate::{ChatCompletionsRoute, Error, types::ResolvedChatCompletionsRequest};

const MEDIA_DOWNLOAD: DownloadPolicy = DownloadPolicy {
    timeout: Duration::from_secs(60),
    max_bytes: MAX_MEDIA_DOWNLOAD_BYTES,
    max_redirects: 10,
};

impl ChatCompletionsRoute {
    /// Downloads every URL the config names and inlines each as a data URL. A route with no
    /// fetcher leaves such a request to the host; a URL that cannot be fetched is the
    /// caller's error, as Python's image fetch failure is.
    pub(crate) async fn inline_remote_media(
        &self,
        request: &mut ResolvedChatCompletionsRequest<'_>,
    ) -> Result<(), Error> {
        let urls = request.config.remote_media(&request.messages);
        if urls.is_empty() {
            return Ok(());
        }
        let fetcher = self
            .media
            .as_ref()
            .ok_or(Error::Unsupported("message media URL"))?;
        let fetch = |url: String| async move {
            let unable = |reason: String| {
                Error::InvalidRequest(
                    format!("Unable to fetch media from url {url}: {reason}").into(),
                )
            };
            let parsed = reqwest::Url::parse(&url).map_err(|err| unable(err.to_string()))?;
            let media = fetcher
                .fetch(parsed, MEDIA_DOWNLOAD)
                .await
                .map_err(|err| unable(err.to_string()))?;
            let data = format!(
                "data:{};base64,{}",
                downloaded_media_type(&url, &media.content_type),
                base64::engine::general_purpose::STANDARD.encode(media.bytes)
            );
            Ok::<_, Error>((url, data))
        };
        let downloaded: HashMap<String, String> = try_join_all(urls.into_iter().map(fetch))
            .await?
            .into_iter()
            .collect();
        inline_media(&mut request.messages, &downloaded);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use litellm_http::{Client, media::MediaFetcher};

    use rstest::rstest;
    use serde_json::{Value, json};
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };

    use crate::{
        ChatCompletionsRoute, Error, prepare::resolve_request, types::ChatCompletionsRequest,
    };

    fn route(media: bool) -> ChatCompletionsRoute {
        let route = crate::test_support::route();
        match media {
            true => route.with_media(MediaFetcher::for_test(Client::plain_for_test())),
            false => route,
        }
    }

    fn request(url: &str) -> ChatCompletionsRequest<'static> {
        ChatCompletionsRequest {
            model: "anthropic/claude-sonnet-4-5",
            messages: json!([{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": url}}
            ]}]),
            optional_params: serde_json::Map::new(),
            api_key: Some("sk-test"),
            api_base: None,
            custom_llm_provider: None,
            extra_headers: None,
            timeout: None,
            drop_params: Some(false),
            model_info: None,
        }
    }

    fn image(request: &crate::types::ResolvedChatCompletionsRequest<'_>) -> Value {
        serde_json::to_value(&request.messages[0].content).unwrap()[0]["image_url"]["url"].clone()
    }

    #[rstest]
    #[tokio::test]
    async fn a_remote_image_is_downloaded_and_inlined() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/cat.png"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(b"png".to_vec(), "image/png"))
            .mount(&server)
            .await;
        let url = format!("{}/cat.png", server.uri());
        let url: &'static str = Box::leak(url.into_boxed_str());
        let mut resolved = resolve_request(request(url)).unwrap();

        route(true)
            .inline_remote_media(&mut resolved)
            .await
            .unwrap();

        assert_eq!(image(&resolved), "data:image/png;base64,cG5n");
    }

    #[rstest]
    #[tokio::test]
    async fn a_media_url_that_cannot_be_fetched_is_the_callers_error() {
        let server = MockServer::start().await;
        let url = format!("{}/missing.png", server.uri());
        let url: &'static str = Box::leak(url.into_boxed_str());
        let mut resolved = resolve_request(request(url)).unwrap();

        let error = route(true)
            .inline_remote_media(&mut resolved)
            .await
            .unwrap_err();

        assert!(matches!(error, Error::InvalidRequest(_)), "{error:?}");
    }

    #[rstest]
    #[tokio::test]
    async fn without_a_fetcher_a_media_url_is_unsupported_and_a_data_url_is_left_alone() {
        let mut remote = resolve_request(request("http://example.com/cat.png")).unwrap();
        assert!(matches!(
            route(false).inline_remote_media(&mut remote).await,
            Err(Error::Unsupported(_))
        ));

        let mut inline = resolve_request(request("data:image/png;base64,cG5n")).unwrap();
        route(false).inline_remote_media(&mut inline).await.unwrap();
        assert_eq!(image(&inline), "data:image/png;base64,cG5n");
    }
}
