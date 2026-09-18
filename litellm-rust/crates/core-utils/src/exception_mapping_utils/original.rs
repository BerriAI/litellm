/// A failure a Rust route produced, before any public class is chosen.
#[derive(Clone, Debug, PartialEq)]
pub enum OriginalException {
    Http {
        status: u16,
        body: String,
        headers: Vec<(String, String)>,
    },
    Connection {
        message: String,
    },
    Timeout {
        timeout_seconds: Option<f64>,
        elapsed_seconds: Option<f64>,
    },
    /// A failure with no HTTP response behind it, such as an unparseable body or a local
    /// file error.
    Plain {
        message: String,
    },
}

/// Which provider-specific text rules apply before the shared status table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExceptionFamily {
    OpenAiCompatible,
    VertexAi,
    Cohere,
    Other,
}

/// `openai_compatible_providers` in `litellm/constants.py`.
const OPENAI_COMPATIBLE_PROVIDERS: &[&str] = &[
    "anyscale",
    "groq",
    "nvidia_nim",
    "cerebras",
    "baseten",
    "sambanova",
    "ai21_chat",
    "ai21",
    "volcengine",
    "codestral",
    "deepseek",
    "tencent",
    "deepinfra",
    "perplexity",
    "xinference",
    "xai",
    "zai",
    "together_ai",
    "fireworks_ai",
    "empower",
    "friendliai",
    "azure_ai",
    "github",
    "litellm_proxy",
    "hosted_vllm",
    "llamafile",
    "lm_studio",
    "galadriel",
    "github_copilot",
    "chatgpt",
    "novita",
    "meta_llama",
    "publicai",
    "synthetic",
    "tensormesh",
    "apertis",
    "nano-gpt",
    "poe",
    "chutes",
    "parasail",
    "libertai",
    "featherless_ai",
    "nscale",
    "nebius",
    "dashscope",
    "qwencloud",
    "qwen_ai_platform",
    "modelscope",
    "moonshot",
    "v0",
    "helicone",
    "morph",
    "lambda_ai",
    "inception",
    "hyperbolic",
    "vercel_ai_gateway",
    "aiml",
    "wandb",
    "cometapi",
    "clarifai",
    "docker_model_runner",
    "ragflow",
    "pinstripes",
    "darkbloom",
    "meta",
    "cognition",
    "scx-ai",
];

impl ExceptionFamily {
    /// The provider dispatch at the top of Python's `exception_type`, in its order.
    pub fn for_provider(provider: &str) -> Self {
        match provider {
            "openai" | "text-completion-openai" | "custom_openai" | "mistral" | "runwayml" => {
                Self::OpenAiCompatible
            }
            provider if OPENAI_COMPATIBLE_PROVIDERS.contains(&provider) => Self::OpenAiCompatible,
            "vertex_ai" | "vertex_ai_beta" | "gemini" => Self::VertexAi,
            "cohere" | "cohere_chat" => Self::Cohere,
            _ => Self::Other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rstest::rstest]
    #[case::openai("openai", ExceptionFamily::OpenAiCompatible)]
    #[case::text_completion_openai("text-completion-openai", ExceptionFamily::OpenAiCompatible)]
    #[case::custom_openai("custom_openai", ExceptionFamily::OpenAiCompatible)]
    #[case::mistral("mistral", ExceptionFamily::OpenAiCompatible)]
    #[case::runwayml("runwayml", ExceptionFamily::OpenAiCompatible)]
    #[case::listed_compatible("azure_ai", ExceptionFamily::OpenAiCompatible)]
    #[case::compatible_list_wins_over_its_own_mapper(
        "together_ai",
        ExceptionFamily::OpenAiCompatible
    )]
    #[case::vertex_ai("vertex_ai", ExceptionFamily::VertexAi)]
    #[case::vertex_ai_beta("vertex_ai_beta", ExceptionFamily::VertexAi)]
    #[case::gemini("gemini", ExceptionFamily::VertexAi)]
    #[case::cohere("cohere", ExceptionFamily::Cohere)]
    #[case::cohere_chat("cohere_chat", ExceptionFamily::Cohere)]
    #[case::unported_mapper("anthropic", ExceptionFamily::Other)]
    #[case::unknown("reducto", ExceptionFamily::Other)]
    #[case::empty("", ExceptionFamily::Other)]
    fn provider_selects_the_family(#[case] provider: &str, #[case] family: ExceptionFamily) {
        assert_eq!(ExceptionFamily::for_provider(provider), family);
    }
}
