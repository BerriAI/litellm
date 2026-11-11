use crate::{ErrorDetail, ErrorSource};

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("Invalid Decisions request: {0}")]
    InvalidRequest(#[source] Invalid),
    #[error(
        "images belongs to the System One format; with input (OpenAI format), send images as input_image content parts"
    )]
    ImagesNextToInput,
    #[error(
        "{provider} does not support parameters: {params:?}, for model={model}. To drop these, set `litellm.drop_params=True` or for proxy:\n\n`litellm_settings:\n drop_params: true`\n"
    )]
    UnsupportedParams {
        provider: &'static str,
        params: Vec<&'static str>,
        model: String,
    },
    #[error(
        "{provider} does not support parameters: {params:?}, for model={model}, and dropping them leaves no questions to answer"
    )]
    NoQuestionsLeft {
        provider: &'static str,
        params: Vec<&'static str>,
        model: String,
    },
    #[error(
        "{}Exception - Decisions provider '{provider}' returned an unexpected response: {detail}",
        exception_label(provider)
    )]
    InvalidResponse {
        provider: &'static str,
        #[source]
        detail: ErrorDetail,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Invalid {
    #[error("{0}")]
    Body(ErrorSource),
    #[error("questions should have 1 to {max} items, got {count}", max = super::MAX_QUESTIONS)]
    QuestionCount { count: usize },
    #[error("questions[{index}].{field} should have {min} to {max} items, got {count}")]
    OptionCount {
        index: usize,
        field: &'static str,
        min: usize,
        max: usize,
        count: usize,
    },
    #[error(
        "questions[{0}]: Choice values must be unique, and a boolean cannot share its text with a string choice"
    )]
    DuplicateChoices(usize),
    #[error("safety_identifier should be a valid string")]
    SafetyIdentifier,
    #[error(
        "{0} input_image part(s) have an image_url that is neither a data:image/...;base64, URL nor an http(s) URL"
    )]
    MalformedImages(usize),
}

fn exception_label(provider: &str) -> String {
    let mut chars = provider.chars();
    chars
        .next()
        .map(|first| first.to_uppercase().chain(chars).collect())
        .unwrap_or_default()
}

impl Error {
    pub fn is_request(&self) -> bool {
        !matches!(self, Self::InvalidResponse { .. })
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    fn an_invalid_response_names_the_provider_as_python_does() {
        let error = Error::InvalidResponse {
            provider: "openai",
            detail: "a choice or score answer has no confidence".into(),
        };

        assert_eq!(
            error.to_string(),
            "OpenaiException - Decisions provider 'openai' returned an unexpected response: a choice or score answer has no confidence"
        );
        assert!(!error.is_request());
    }

    #[rstest]
    fn unsupported_params_list_every_refused_name() {
        let error = Error::UnsupportedParams {
            provider: "azure",
            params: vec!["box questions", "audio"],
            model: "azure/luna".into(),
        };

        assert!(
            error
                .to_string()
                .starts_with(r#"azure does not support parameters: ["box questions", "audio"], for model=azure/luna."#)
        );
        assert!(error.is_request());
    }
}
