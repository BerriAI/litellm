export enum Providers {
  A2A_Agent = "A2A Agent",
  AI21 = "Ai21",
  AI21_CHAT = "Ai21 Chat",
  AIML = "AI/ML API",
  AIOHTTP_OPENAI = "Aiohttp Openai",
  Anthropic = "Anthropic",
  ANTHROPIC_TEXT = "Anthropic Text",
  AssemblyAI = "AssemblyAI",
  AUTO_ROUTER = "Auto Router",
  Bedrock = "Amazon Bedrock",
  SageMaker = "AWS SageMaker",
  Azure = "Azure",
  Azure_AI_Studio = "Azure AI Foundry (Studio)",
  AZURE_TEXT = "Azure Text",
  BASETEN = "Baseten",
  BYTEZ = "Bytez",
  Cerebras = "Cerebras",
  CLARIFAI = "Clarifai",
  CLOUDFLARE = "Cloudflare",
  CODESTRAL = "Codestral",
  Cohere = "Cohere",
  COHERE_CHAT = "Cohere Chat",
  COMETAPI = "Cometapi",
  COMPACTIFAI = "Compactifai",
  CUSTOM = "Custom",
  CUSTOM_OPENAI = "Custom Openai",
  Cursor = "Cursor",
  Dashscope = "Dashscope",
  Databricks = "Databricks (Qwen API)",
  DATAROBOT = "Datarobot",
  DeepInfra = "DeepInfra",
  Deepgram = "Deepgram",
  Deepseek = "Deepseek",
  DOCKER_MODEL_RUNNER = "Docker Model Runner",
  DOTPROMPT = "Dotprompt",
  ElevenLabs = "ElevenLabs",
  EMPOWER = "Empower",
  FalAI = "Fal AI",
  FEATHERLESS_AI = "Featherless Ai",
  FireworksAI = "Fireworks AI",
  FRIENDLIAI = "Friendliai",
  GALADRIEL = "Galadriel",
  GITHUB = "Github",
  GITHUB_COPILOT = "Github Copilot",
  Google_AI_Studio = "Google AI Studio",
  GradientAI = "GradientAI",
  Groq = "Groq",
  HEROKU = "Heroku",
  Hosted_Vllm = "vllm",
  HUGGINGFACE = "Huggingface",
  HUMANLOOP = "Humanloop",
  HYPERBOLIC = "Hyperbolic",
  Infinity = "Infinity",
  JinaAI = "Jina AI",
  LAMBDA_AI = "Lambda Ai",
  LANGFUSE = "Langfuse",
  LEMONADE = "Lemonade",
  LITELLM_PROXY = "Litellm Proxy",
  LLAMAFILE = "Llamafile",
  LM_STUDIO = "Lm Studio",
  LLAMA = "Meta Llama",
  MARITALK = "Maritalk",
  MILVUS = "Milvus",
  MiniMax = "MiniMax",
  MistralAI = "Mistral AI",
  MOONSHOT = "Moonshot",
  MORPH = "Morph",
  NEBIUS = "Nebius",
  NLP_CLOUD = "Nlp Cloud",
  NOVITA = "Novita",
  NSCALE = "Nscale",
  NVIDIA_NIM = "Nvidia Nim",
  Ollama = "Ollama",
  OLLAMA_CHAT = "Ollama Chat",
  OOBABOOGA = "Oobabooga",
  OpenAI = "OpenAI",
  OPENAI_LIKE = "Openai Like",
  OpenAI_Compatible = "OpenAI-Compatible Endpoints (Together AI, etc.)",
  OpenAI_Text = "OpenAI Text Completion",
  OpenAI_Text_Compatible = "OpenAI-Compatible Text Completion Models (Together AI, etc.)",
  Openrouter = "Openrouter",
  Oracle = "Oracle Cloud Infrastructure (OCI)",
  OVHCLOUD = "Ovhcloud",
  Perplexity = "Perplexity",
  PETALS = "Petals",
  PG_VECTOR = "Pg Vector",
  PREDIBASE = "Predibase",
  RECRAFT = "Recraft",
  REPLICATE = "Replicate",
  RunwayML = "RunwayML",
  SAGEMAKER_LEGACY = "Sagemaker",
  Sambanova = "Sambanova",
  SAP = "SAP Generative AI Hub",
  Snowflake = "Snowflake",
  TEXT_COMPLETION_CODESTRAL = "Text-Completion-Codestral",
  TogetherAI = "TogetherAI",
  TOPAZ = "Topaz",
  Triton = "Triton",
  V0 = "V0",
  VERCEL_AI_GATEWAY = "Vercel Ai Gateway",
  Vertex_AI = "Vertex AI (Anthropic, Gemini, etc.)",
  VERTEX_AI_BETA = "Vertex Ai Beta",
  VLLM = "Vllm",
  VolcEngine = "VolcEngine",
  Voyage = "Voyage AI",
  WANDB = "Wandb",
  WATSONX = "Watsonx",
  WATSONX_TEXT = "Watsonx Text",
  xAI = "xAI",
  XINFERENCE = "Xinference",
}

export const provider_map: Record<string, string> = {
  A2A_Agent: "a2a_agent",
  AI21: "ai21",
  AI21_CHAT: "ai21_chat",
  AIML: "aiml",
  AIOHTTP_OPENAI: "aiohttp_openai",
  Anthropic: "anthropic",
  ANTHROPIC_TEXT: "anthropic_text",
  AssemblyAI: "assemblyai",
  AUTO_ROUTER: "auto_router",
  Azure: "azure",
  Azure_AI_Studio: "azure_ai",
  AZURE_TEXT: "azure_text",
  BASETEN: "baseten",
  Bedrock: "bedrock",
  BYTEZ: "bytez",
  Cerebras: "cerebras",
  CLARIFAI: "clarifai",
  CLOUDFLARE: "cloudflare",
  CODESTRAL: "codestral",
  Cohere: "cohere",
  COHERE_CHAT: "cohere_chat",
  COMETAPI: "cometapi",
  COMPACTIFAI: "compactifai",
  CUSTOM: "custom",
  CUSTOM_OPENAI: "custom_openai",
  Cursor: "cursor",
  Dashscope: "dashscope",
  Databricks: "databricks",
  DATAROBOT: "datarobot",
  DeepInfra: "deepinfra",
  Deepgram: "deepgram",
  Deepseek: "deepseek",
  DOCKER_MODEL_RUNNER: "docker_model_runner",
  DOTPROMPT: "dotprompt",
  ElevenLabs: "elevenlabs",
  EMPOWER: "empower",
  FalAI: "fal_ai",
  FEATHERLESS_AI: "featherless_ai",
  FireworksAI: "fireworks_ai",
  FRIENDLIAI: "friendliai",
  GALADRIEL: "galadriel",
  GITHUB: "github",
  GITHUB_COPILOT: "github_copilot",
  Google_AI_Studio: "gemini",
  GradientAI: "gradient_ai",
  Groq: "groq",
  HEROKU: "heroku",
  Hosted_Vllm: "hosted_vllm",
  HUGGINGFACE: "huggingface",
  HUMANLOOP: "humanloop",
  HYPERBOLIC: "hyperbolic",
  Infinity: "infinity",
  JinaAI: "jina_ai",
  LAMBDA_AI: "lambda_ai",
  LANGFUSE: "langfuse",
  LEMONADE: "lemonade",
  LITELLM_PROXY: "litellm_proxy",
  LLAMAFILE: "llamafile",
  LLAMA: "meta_llama",
  LM_STUDIO: "lm_studio",
  MARITALK: "maritalk",
  MILVUS: "milvus",
  MiniMax: "minimax",
  MistralAI: "mistral",
  MOONSHOT: "moonshot",
  MORPH: "morph",
  NEBIUS: "nebius",
  NLP_CLOUD: "nlp_cloud",
  NOVITA: "novita",
  NSCALE: "nscale",
  NVIDIA_NIM: "nvidia_nim",
  Ollama: "ollama",
  OLLAMA_CHAT: "ollama_chat",
  OOBABOOGA: "oobabooga",
  OpenAI: "openai",
  OPENAI_LIKE: "openai_like",
  OpenAI_Compatible: "openai",
  OpenAI_Text: "text-completion-openai",
  OpenAI_Text_Compatible: "text-completion-openai",
  Openrouter: "openrouter",
  Oracle: "oci",
  OVHCLOUD: "ovhcloud",
  Perplexity: "perplexity",
  PETALS: "petals",
  PG_VECTOR: "pg_vector",
  PREDIBASE: "predibase",
  RECRAFT: "recraft",
  REPLICATE: "replicate",
  RunwayML: "runwayml",
  SAGEMAKER_LEGACY: "sagemaker",
  SageMaker: "sagemaker_chat",
  Sambanova: "sambanova",
  SAP: "sap",
  Snowflake: "snowflake",
  TEXT_COMPLETION_CODESTRAL: "text-completion-codestral",
  TogetherAI: "together_ai",
  TOPAZ: "topaz",
  Triton: "triton",
  V0: "v0",
  VERCEL_AI_GATEWAY: "vercel_ai_gateway",
  Vertex_AI: "vertex_ai",
  VERTEX_AI_BETA: "vertex_ai_beta",
  VLLM: "vllm",
  VolcEngine: "volcengine",
  Voyage: "voyage",
  WANDB: "wandb",
  WATSONX: "watsonx",
  WATSONX_TEXT: "watsonx_text",
  xAI: "xai",
  XINFERENCE: "xinference",
};

const asset_logos_folder = "../ui/assets/logos/";

export const providerLogoMap: Record<string, string> = {
  [Providers.A2A_Agent]: `${asset_logos_folder}a2a_agent.png`,
  [Providers.AI21]: `${asset_logos_folder}ai21.svg`,
  [Providers.AI21_CHAT]: `${asset_logos_folder}ai21.svg`,
  [Providers.AIML]: `${asset_logos_folder}aiml_api.svg`,
  [Providers.AIOHTTP_OPENAI]: `${asset_logos_folder}openai_small.svg`,
  [Providers.Anthropic]: `${asset_logos_folder}anthropic.svg`,
  [Providers.ANTHROPIC_TEXT]: `${asset_logos_folder}anthropic.svg`,
  [Providers.AssemblyAI]: `${asset_logos_folder}assemblyai_small.png`,
  [Providers.Azure]: `${asset_logos_folder}microsoft_azure.svg`,
  [Providers.Azure_AI_Studio]: `${asset_logos_folder}microsoft_azure.svg`,
  [Providers.AZURE_TEXT]: `${asset_logos_folder}microsoft_azure.svg`,
  [Providers.BASETEN]: `${asset_logos_folder}baseten.svg`,
  [Providers.Bedrock]: `${asset_logos_folder}bedrock.svg`,
  [Providers.SageMaker]: `${asset_logos_folder}bedrock.svg`,
  [Providers.Cerebras]: `${asset_logos_folder}cerebras.svg`,
  [Providers.CLOUDFLARE]: `${asset_logos_folder}cloudflare.svg`,
  [Providers.CODESTRAL]: `${asset_logos_folder}mistral.svg`,
  [Providers.Cohere]: `${asset_logos_folder}cohere.svg`,
  [Providers.COHERE_CHAT]: `${asset_logos_folder}cohere.svg`,
  [Providers.COMETAPI]: `${asset_logos_folder}cometapi.svg`,
  [Providers.Cursor]: `${asset_logos_folder}cursor.svg`,
  [Providers.CUSTOM_OPENAI]: `${asset_logos_folder}openai_small.svg`,
  [Providers.Databricks]: `${asset_logos_folder}databricks.svg`,
  [Providers.Dashscope]: `${asset_logos_folder}dashscope.svg`,
  [Providers.Deepseek]: `${asset_logos_folder}deepseek.svg`,
  [Providers.Deepgram]: `${asset_logos_folder}deepgram.png`,
  [Providers.DeepInfra]: `${asset_logos_folder}deepinfra.png`,
  [Providers.ElevenLabs]: `${asset_logos_folder}elevenlabs.png`,
  [Providers.FalAI]: `${asset_logos_folder}fal_ai.jpg`,
  [Providers.FEATHERLESS_AI]: `${asset_logos_folder}featherless.svg`,
  [Providers.FireworksAI]: `${asset_logos_folder}fireworks.svg`,
  [Providers.FRIENDLIAI]: `${asset_logos_folder}friendli.svg`,
  [Providers.GITHUB]: `${asset_logos_folder}github.svg`,
  [Providers.GITHUB_COPILOT]: `${asset_logos_folder}github_copilot.svg`,
  [Providers.Google_AI_Studio]: `${asset_logos_folder}google.svg`,
  [Providers.GradientAI]: `${asset_logos_folder}gradientai.svg`,
  [Providers.Groq]: `${asset_logos_folder}groq.svg`,
  [Providers.Hosted_Vllm]: `${asset_logos_folder}vllm.png`,
  [Providers.HUGGINGFACE]: `${asset_logos_folder}huggingface.svg`,
  [Providers.HYPERBOLIC]: `${asset_logos_folder}hyperbolic.svg`,
  [Providers.Infinity]: `${asset_logos_folder}infinity.png`,
  [Providers.JinaAI]: `${asset_logos_folder}jina.png`,
  [Providers.LAMBDA_AI]: `${asset_logos_folder}lambda.svg`,
  [Providers.LANGFUSE]: `${asset_logos_folder}langfuse.svg`,
  [Providers.LITELLM_PROXY]: `${asset_logos_folder}litellm.jpg`,
  [Providers.LM_STUDIO]: `${asset_logos_folder}lmstudio.svg`,
  [Providers.LLAMA]: `${asset_logos_folder}meta_llama.svg`,
  [Providers.MILVUS]: `${asset_logos_folder}milvus.svg`,
  [Providers.MiniMax]: `${asset_logos_folder}minimax.svg`,
  [Providers.MistralAI]: `${asset_logos_folder}mistral.svg`,
  [Providers.MOONSHOT]: `${asset_logos_folder}moonshot.svg`,
  [Providers.MORPH]: `${asset_logos_folder}morph.svg`,
  [Providers.NEBIUS]: `${asset_logos_folder}nebius.svg`,
  [Providers.NOVITA]: `${asset_logos_folder}novita.svg`,
  [Providers.NVIDIA_NIM]: `${asset_logos_folder}nvidia_nim.svg`,
  [Providers.Ollama]: `${asset_logos_folder}ollama.svg`,
  [Providers.OLLAMA_CHAT]: `${asset_logos_folder}ollama.svg`,
  [Providers.OOBABOOGA]: `${asset_logos_folder}openai_small.svg`,
  [Providers.OpenAI]: `${asset_logos_folder}openai_small.svg`,
  [Providers.OPENAI_LIKE]: `${asset_logos_folder}openai_small.svg`,
  [Providers.OpenAI_Text]: `${asset_logos_folder}openai_small.svg`,
  [Providers.OpenAI_Text_Compatible]: `${asset_logos_folder}openai_small.svg`,
  [Providers.OpenAI_Compatible]: `${asset_logos_folder}openai_small.svg`,
  [Providers.Openrouter]: `${asset_logos_folder}openrouter.svg`,
  [Providers.Oracle]: `${asset_logos_folder}oracle.svg`,
  [Providers.Perplexity]: `${asset_logos_folder}perplexity-ai.svg`,
  [Providers.RECRAFT]: `${asset_logos_folder}recraft.svg`,
  [Providers.REPLICATE]: `${asset_logos_folder}replicate.svg`,
  [Providers.RunwayML]: `${asset_logos_folder}runwayml.png`,
  [Providers.SAGEMAKER_LEGACY]: `${asset_logos_folder}bedrock.svg`,
  [Providers.Sambanova]: `${asset_logos_folder}sambanova.svg`,
  [Providers.SAP]: `${asset_logos_folder}sap.png`,
  [Providers.Snowflake]: `${asset_logos_folder}snowflake.svg`,
  [Providers.TEXT_COMPLETION_CODESTRAL]: `${asset_logos_folder}mistral.svg`,
  [Providers.TogetherAI]: `${asset_logos_folder}togetherai.svg`,
  [Providers.TOPAZ]: `${asset_logos_folder}topaz.svg`,
  [Providers.Triton]: `${asset_logos_folder}nvidia_triton.png`,
  [Providers.V0]: `${asset_logos_folder}v0.svg`,
  [Providers.VERCEL_AI_GATEWAY]: `${asset_logos_folder}vercel.svg`,
  [Providers.Vertex_AI]: `${asset_logos_folder}google.svg`,
  [Providers.VERTEX_AI_BETA]: `${asset_logos_folder}google.svg`,
  [Providers.VLLM]: `${asset_logos_folder}vllm.png`,
  [Providers.VolcEngine]: `${asset_logos_folder}volcengine.png`,
  [Providers.Voyage]: `${asset_logos_folder}voyage.webp`,
  [Providers.WATSONX]: `${asset_logos_folder}watsonx.svg`,
  [Providers.WATSONX_TEXT]: `${asset_logos_folder}watsonx.svg`,
  [Providers.xAI]: `${asset_logos_folder}xai.svg`,
  [Providers.XINFERENCE]: `${asset_logos_folder}xinference.svg`,
};

export const getProviderLogoAndName = (providerValue: string): { logo: string; displayName: string } => {
  if (!providerValue) {
    return { logo: "", displayName: "-" };
  }

  // Handle special case for "gemini" provider value
  if (providerValue.toLowerCase() === "gemini") {
    const displayName = Providers.Google_AI_Studio;
    const logo = providerLogoMap[displayName];
    return { logo, displayName };
  }

  // Find the enum key by matching provider_map values
  const enumKey = Object.keys(provider_map).find(
    (key) => provider_map[key].toLowerCase() === providerValue.toLowerCase(),
  );

  if (!enumKey) {
    return { logo: "", displayName: providerValue };
  }

  // Get the display name from Providers enum and logo from map
  const displayName = Providers[enumKey as keyof typeof Providers];
  const logo = providerLogoMap[displayName as keyof typeof providerLogoMap];

  return { logo, displayName };
};

export const getPlaceholder = (selectedProvider: string): string => {
  if (selectedProvider === Providers.AIML) {
    return "aiml/flux-pro/v1.1";
  } else if (selectedProvider === Providers.Vertex_AI) {
    return "gemini-pro";
  } else if (selectedProvider == Providers.Anthropic) {
    return "claude-3-opus";
  } else if (selectedProvider == Providers.Bedrock) {
    return "claude-3-opus";
  } else if (selectedProvider == Providers.SageMaker) {
    return "sagemaker/jumpstart-dft-meta-textgeneration-llama-2-7b";
  } else if (selectedProvider == Providers.Google_AI_Studio) {
    return "gemini-pro";
  } else if (selectedProvider == Providers.Azure_AI_Studio) {
    return "azure_ai/command-r-plus";
  } else if (selectedProvider == Providers.Azure) {
    return "my-deployment";
  } else if (selectedProvider == Providers.Oracle) {
    return "oci/xai.grok-4";
  } else if (selectedProvider == Providers.Snowflake) {
    return "snowflake/mistral-7b";
  } else if (selectedProvider == Providers.Voyage) {
    return "voyage/";
  } else if (selectedProvider == Providers.JinaAI) {
    return "jina_ai/";
  } else if (selectedProvider == Providers.VolcEngine) {
    return "volcengine/<any-model-on-volcengine>";
  } else if (selectedProvider == Providers.DeepInfra) {
    return "deepinfra/<any-model-on-deepinfra>";
  } else if (selectedProvider == Providers.FalAI) {
    return "fal_ai/fal-ai/flux-pro/v1.1-ultra";
  } else if (selectedProvider == Providers.RunwayML) {
    return "runwayml/gen4_turbo";
  } else if (selectedProvider === Providers.WATSONX) {
    return "watsonx/ibm/granite-3-3-8b-instruct";
  } else if (selectedProvider === Providers.Cursor) {
    return "cursor/claude-4-sonnet";
  } else {
    return "gpt-3.5-turbo";
  }
};

export const getProviderModels = (provider: Providers, modelMap: any): Array<string> => {
  let providerKey = provider;
  console.log(`Provider key: ${providerKey}`);
  let custom_llm_provider = provider_map[providerKey];
  console.log(`Provider mapped to: ${custom_llm_provider}`);

  let providerModels: Array<string> = [];

  if (providerKey && typeof modelMap === "object") {
    Object.entries(modelMap).forEach(([key, value]) => {
      if (value !== null && typeof value === "object" && "litellm_provider" in (value as object)) {
        const litellmProvider = (value as any)["litellm_provider"];
        if (
          litellmProvider === custom_llm_provider ||
          (typeof litellmProvider === "string" && litellmProvider.includes(custom_llm_provider))
        ) {
          providerModels.push(key);
        }
      }
    });
    // Special case for cohere
    // we need both cohere_chat and cohere models to show on dropdown
    if (providerKey == Providers.Cohere) {
      console.log("Adding cohere chat models");
      Object.entries(modelMap).forEach(([key, value]) => {
        if (
          value !== null &&
          typeof value === "object" &&
          "litellm_provider" in (value as object) &&
          (value as any)["litellm_provider"] === "cohere_chat"
        ) {
          providerModels.push(key);
        }
      });
    }

    // Special case for sagemaker
    // we need both sagemaker and sagemaker_chat models to show on dropdown
    if (providerKey == Providers.SageMaker) {
      console.log("Adding sagemaker chat models");
      Object.entries(modelMap).forEach(([key, value]) => {
        if (
          value !== null &&
          typeof value === "object" &&
          "litellm_provider" in (value as object) &&
          (value as any)["litellm_provider"] === "sagemaker_chat"
        ) {
          providerModels.push(key);
        }
      });
    }
  }

  return providerModels;
};
