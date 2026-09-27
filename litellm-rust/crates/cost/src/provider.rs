//! Mirrors Python's LlmProviders (litellm/types/utils.py), generated from its variant list.

// variant names mirror Python's enum members verbatim
#![allow(non_camel_case_types)]

#[derive(
    Clone,
    Copy,
    Debug,
    Eq,
    Hash,
    PartialEq,
    strum::Display,
    strum::EnumString,
    strum::IntoStaticStr,
    strum::VariantArray,
)]
pub enum LlmProviders {
    #[strum(serialize = "openai")]
    OPENAI,
    #[strum(serialize = "chatgpt")]
    CHATGPT,
    #[strum(serialize = "openai_like")]
    OPENAI_LIKE,
    #[strum(serialize = "jina_ai")]
    JINA_AI,
    #[strum(serialize = "xai")]
    XAI,
    #[strum(serialize = "zai")]
    ZAI,
    #[strum(serialize = "custom_openai")]
    CUSTOM_OPENAI,
    #[strum(serialize = "text-completion-openai")]
    TEXT_COMPLETION_OPENAI,
    #[strum(serialize = "cohere")]
    COHERE,
    #[strum(serialize = "cohere_chat")]
    COHERE_CHAT,
    #[strum(serialize = "clarifai")]
    CLARIFAI,
    #[strum(serialize = "anthropic")]
    ANTHROPIC,
    #[strum(serialize = "anthropic_text")]
    ANTHROPIC_TEXT,
    #[strum(serialize = "bytez")]
    BYTEZ,
    #[strum(serialize = "replicate")]
    REPLICATE,
    #[strum(serialize = "reducto")]
    REDUCTO,
    #[strum(serialize = "aws_textract")]
    AWS_TEXTRACT,
    #[strum(serialize = "runwayml")]
    RUNWAYML,
    #[strum(serialize = "aws_polly")]
    AWS_POLLY,
    #[strum(serialize = "transcribe")]
    TRANSCRIBE,
    #[strum(serialize = "huggingface")]
    HUGGINGFACE,
    #[strum(serialize = "together_ai")]
    TOGETHER_AI,
    #[strum(serialize = "openrouter")]
    OPENROUTER,
    #[strum(serialize = "datarobot")]
    DATAROBOT,
    #[strum(serialize = "vertex_ai")]
    VERTEX_AI,
    #[strum(serialize = "vertex_ai_beta")]
    VERTEX_AI_BETA,
    #[strum(serialize = "gemini")]
    GEMINI,
    #[strum(serialize = "ai21")]
    AI21,
    #[strum(serialize = "baseten")]
    BASETEN,
    #[strum(serialize = "black_forest_labs")]
    BLACK_FOREST_LABS,
    #[strum(serialize = "azure")]
    AZURE,
    #[strum(serialize = "azure_text")]
    AZURE_TEXT,
    #[strum(serialize = "azure_ai")]
    AZURE_AI,
    #[strum(serialize = "sagemaker")]
    SAGEMAKER,
    #[strum(serialize = "sagemaker_chat")]
    SAGEMAKER_CHAT,
    #[strum(serialize = "sagemaker_nova")]
    SAGEMAKER_NOVA,
    #[strum(serialize = "bedrock")]
    BEDROCK,
    #[strum(serialize = "vllm")]
    VLLM,
    #[strum(serialize = "nlp_cloud")]
    NLP_CLOUD,
    #[strum(serialize = "petals")]
    PETALS,
    #[strum(serialize = "oobabooga")]
    OOBABOOGA,
    #[strum(serialize = "ollama")]
    OLLAMA,
    #[strum(serialize = "ollama_chat")]
    OLLAMA_CHAT,
    #[strum(serialize = "deepinfra")]
    DEEPINFRA,
    #[strum(serialize = "perplexity")]
    PERPLEXITY,
    #[strum(serialize = "mistral")]
    MISTRAL,
    #[strum(serialize = "milvus")]
    MILVUS,
    #[strum(serialize = "groq")]
    GROQ,
    #[strum(serialize = "a2a")]
    A2A,
    #[strum(serialize = "gigachat")]
    GIGACHAT,
    #[strum(serialize = "nvidia_nim")]
    NVIDIA_NIM,
    #[strum(serialize = "nvidia_riva")]
    NVIDIA_RIVA,
    #[strum(serialize = "soniox")]
    SONIOX,
    #[strum(serialize = "cerebras")]
    CEREBRAS,
    #[strum(serialize = "ai21_chat")]
    AI21_CHAT,
    #[strum(serialize = "volcengine")]
    VOLCENGINE,
    #[strum(serialize = "codestral")]
    CODESTRAL,
    #[strum(serialize = "text-completion-codestral")]
    TEXT_COMPLETION_CODESTRAL,
    #[strum(serialize = "dashscope")]
    DASHSCOPE,
    #[strum(serialize = "qwencloud")]
    QWENCLOUD,
    #[strum(serialize = "qwen_ai_platform")]
    QWEN_AI_PLATFORM,
    #[strum(serialize = "modelscope")]
    MODELSCOPE,
    #[strum(serialize = "moonshot")]
    MOONSHOT,
    #[strum(serialize = "publicai")]
    PUBLICAI,
    #[strum(serialize = "v0")]
    V0,
    #[strum(serialize = "morph")]
    MORPH,
    #[strum(serialize = "lambda_ai")]
    LAMBDA_AI,
    #[strum(serialize = "inception")]
    INCEPTION,
    #[strum(serialize = "text-completion-inception")]
    TEXT_COMPLETION_INCEPTION,
    #[strum(serialize = "deepseek")]
    DEEPSEEK,
    #[strum(serialize = "sambanova")]
    SAMBANOVA,
    #[strum(serialize = "maritalk")]
    MARITALK,
    #[strum(serialize = "voyage")]
    VOYAGE,
    #[strum(serialize = "cloudflare")]
    CLOUDFLARE,
    #[strum(serialize = "xinference")]
    XINFERENCE,
    #[strum(serialize = "fireworks_ai")]
    FIREWORKS_AI,
    #[strum(serialize = "friendliai")]
    FRIENDLIAI,
    #[strum(serialize = "featherless_ai")]
    FEATHERLESS_AI,
    #[strum(serialize = "watsonx")]
    WATSONX,
    #[strum(serialize = "watsonx_text")]
    WATSONX_TEXT,
    #[strum(serialize = "triton")]
    TRITON,
    #[strum(serialize = "predibase")]
    PREDIBASE,
    #[strum(serialize = "databricks")]
    DATABRICKS,
    #[strum(serialize = "empower")]
    EMPOWER,
    #[strum(serialize = "github")]
    GITHUB,
    #[strum(serialize = "ragflow")]
    RAGFLOW,
    #[strum(serialize = "compactifai")]
    COMPACTIFAI,
    #[strum(serialize = "docker_model_runner")]
    DOCKER_MODEL_RUNNER,
    #[strum(serialize = "custom")]
    CUSTOM,
    #[strum(serialize = "litellm_proxy")]
    LITELLM_PROXY,
    #[strum(serialize = "hosted_vllm")]
    HOSTED_VLLM,
    #[strum(serialize = "tencent")]
    TENCENT,
    #[strum(serialize = "llamafile")]
    LLAMAFILE,
    #[strum(serialize = "lm_studio")]
    LM_STUDIO,
    #[strum(serialize = "galadriel")]
    GALADRIEL,
    #[strum(serialize = "nebius")]
    NEBIUS,
    #[strum(serialize = "infinity")]
    INFINITY,
    #[strum(serialize = "deepgram")]
    DEEPGRAM,
    #[strum(serialize = "elevenlabs")]
    ELEVENLABS,
    #[strum(serialize = "novita")]
    NOVITA,
    #[strum(serialize = "aiohttp_openai")]
    AIOHTTP_OPENAI,
    #[strum(serialize = "langfuse")]
    LANGFUSE,
    #[strum(serialize = "humanloop")]
    HUMANLOOP,
    #[strum(serialize = "topaz")]
    TOPAZ,
    #[strum(serialize = "sap")]
    SAP_GENERATIVE_AI_HUB,
    #[strum(serialize = "assemblyai")]
    ASSEMBLYAI,
    #[strum(serialize = "azure_speech")]
    AZURE_SPEECH,
    #[strum(serialize = "charity_engine")]
    CHARITY_ENGINE,
    #[strum(serialize = "github_copilot")]
    GITHUB_COPILOT,
    #[strum(serialize = "snowflake")]
    SNOWFLAKE,
    #[strum(serialize = "gradient_ai")]
    GRADIENT_AI,
    #[strum(serialize = "meta_llama")]
    LLAMA,
    #[strum(serialize = "nscale")]
    NSCALE,
    #[strum(serialize = "pg_vector")]
    PG_VECTOR,
    #[strum(serialize = "s3_vectors")]
    S3_VECTORS,
    #[strum(serialize = "valkey")]
    VALKEY,
    #[strum(serialize = "mongodb")]
    MONGODB,
    #[strum(serialize = "helicone")]
    HELICONE,
    #[strum(serialize = "hyperbolic")]
    HYPERBOLIC,
    #[strum(serialize = "recraft")]
    RECRAFT,
    #[strum(serialize = "fal_ai")]
    FAL_AI,
    #[strum(serialize = "stability")]
    STABILITY,
    #[strum(serialize = "heroku")]
    HEROKU,
    #[strum(serialize = "aiml")]
    AIML,
    #[strum(serialize = "cometapi")]
    COMETAPI,
    #[strum(serialize = "oci")]
    OCI,
    #[strum(serialize = "auto_router")]
    AUTO_ROUTER,
    #[strum(serialize = "vercel_ai_gateway")]
    VERCEL_AI_GATEWAY,
    #[strum(serialize = "edenai")]
    EDENAI,
    #[strum(serialize = "dotprompt")]
    DOTPROMPT,
    #[strum(serialize = "manus")]
    MANUS,
    #[strum(serialize = "wandb")]
    WANDB,
    #[strum(serialize = "ovhcloud")]
    OVHCLOUD,
    #[strum(serialize = "scaleway")]
    SCALEWAY,
    #[strum(serialize = "lemonade")]
    LEMONADE,
    #[strum(serialize = "amazon_nova")]
    AMAZON_NOVA,
    #[strum(serialize = "a2a_agent")]
    A2A_AGENT,
    #[strum(serialize = "langgraph")]
    LANGGRAPH,
    #[strum(serialize = "langflow")]
    LANGFLOW,
    #[strum(serialize = "minimax")]
    MINIMAX,
    #[strum(serialize = "synthetic")]
    SYNTHETIC,
    #[strum(serialize = "apertis")]
    APERTIS,
    #[strum(serialize = "nano-gpt")]
    NANOGPT,
    #[strum(serialize = "poe")]
    POE,
    #[strum(serialize = "chutes")]
    CHUTES,
    #[strum(serialize = "neosantara")]
    NEOSANTARA,
    #[strum(serialize = "parasail")]
    PARASAIL,
    #[strum(serialize = "xiaomi_mimo")]
    XIAOMI_MIMO,
    #[strum(serialize = "tensormesh")]
    TENSORMESH,
    #[strum(serialize = "libertai")]
    LIBERTAI,
    #[strum(serialize = "pinstripes")]
    PINSTRIPES,
    #[strum(serialize = "cognition")]
    COGNITION,
    #[strum(serialize = "scx-ai")]
    SCX_AI,
    #[strum(serialize = "darkbloom")]
    DARKBLOOM,
    #[strum(serialize = "meta")]
    META,
    #[strum(serialize = "litellm_agent")]
    LITELLM_AGENT,
    #[strum(serialize = "cursor")]
    CURSOR,
    #[strum(serialize = "bedrock_mantle")]
    BEDROCK_MANTLE,
    #[strum(serialize = "gdc")]
    GDC,
}

impl LlmProviders {
    pub fn as_str(self) -> &'static str {
        self.into()
    }

    pub fn matches(self, name: Option<&str>) -> bool {
        name == Some(self.as_str())
    }
}
