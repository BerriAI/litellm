
export enum Providers {
  A2A_Agent = "A2A Agent",
  AIML = "AI/ML API",
  Bedrock = "Amazon Bedrock",
  Anthropic = "Anthropic",
  AssemblyAI = "AssemblyAI",
  SageMaker = "AWS SageMaker",
  Azure = "Azure",
  Azure_AI_Studio = "Azure AI Foundry (Studio)",
  Cerebras = "Cerebras",
  Cohere = "Cohere",
  Dashscope = "Dashscope",
  Databricks = "Databricks (Qwen API)",
  DeepInfra = "DeepInfra",
  Deepgram = "Deepgram",
  Deepseek = "Deepseek",
  ElevenLabs = "ElevenLabs",
  FalAI = "Fal AI",
  FireworksAI = "Fireworks AI",
  Google_AI_Studio = "Google AI Studio",
  GradientAI = "GradientAI",
  Groq = "Groq",
  Hosted_Vllm = "vllm",
  Infinity = "Infinity",
  JinaAI = "Jina AI",
  MistralAI = "Mistral AI",
  Ollama = "Ollama",
  OpenAI = "OpenAI",
  OpenAI_Compatible = "OpenAI-Compatible Endpoints (Together AI, etc.)",
  OpenAI_Text = "OpenAI Text Completion",
  OpenAI_Text_Compatible = "OpenAI-Compatible Text Completion Models (Together AI, etc.)",
  Openrouter = "Openrouter",
  Oracle = "Oracle Cloud Infrastructure (OCI)",
  Perplexity = "Perplexity",
  RunwayML = "RunwayML",
  Sambanova = "Sambanova",
  Snowflake = "Snowflake",
  TogetherAI = "TogetherAI",
  Triton = "Triton",
  Vertex_AI = "Vertex AI (Anthropic, Gemini, etc.)",
  VolcEngine = "VolcEngine",
  Voyage = "Voyage AI",
  xAI = "xAI",
}

export const provider_map: Record<string, string> = {
  A2A_Agent: "a2a_agent",
  AIML: "aiml",
  OpenAI: "openai",
  OpenAI_Text: "text-completion-openai",
  Azure: "azure",
  Azure_AI_Studio: "azure_ai",
  Anthropic: "anthropic",
  Google_AI_Studio: "gemini",
  Bedrock: "bedrock",
  Groq: "groq",
  MistralAI: "mistral",
  Cohere: "cohere",
  OpenAI_Compatible: "openai",
  OpenAI_Text_Compatible: "text-completion-openai",
  Vertex_AI: "vertex_ai",
  Databricks: "databricks",
  Dashscope: "dashscope",
  xAI: "xai",
  Deepseek: "deepseek",
  Ollama: "ollama",
  AssemblyAI: "assemblyai",
  Cerebras: "cerebras",
  Sambanova: "sambanova",
  Perplexity: "perplexity",
  RunwayML: "runwayml",
  TogetherAI: "together_ai",
  Openrouter: "openrouter",
  Oracle: "oci",
  Snowflake: "snowflake",
  FireworksAI: "fireworks_ai",
  GradientAI: "gradient_ai",
  Triton: "triton",
  Deepgram: "deepgram",
  ElevenLabs: "elevenlabs",
  FalAI: "fal_ai",
  SageMaker: "sagemaker_chat",
  Voyage: "voyage",
  JinaAI: "jina_ai",
  VolcEngine: "volcengine",
  DeepInfra: "deepinfra",
  Hosted_Vllm: "hosted_vllm",
  Infinity: "infinity",
};

const asset_logos_folder = "../ui/assets/logos/";

export const providerLogoMap: Record<string, string> = {
  [Providers.A2A_Agent]: `${asset_logos_folder}a2a_agent.png`,
  [Providers.AIML]: `${asset_logos_folder}aiml_api.svg`,
  [Providers.Anthropic]: `${asset_logos_folder}anthropic.svg`,
  [Providers.AssemblyAI]: `${asset_logos_folder}assemblyai_small.png`,
  [Providers.Azure]: `${asset_logos_folder}microsoft_azure.svg`,
  [Providers.Azure_AI_Studio]: `${asset_logos_folder}microsoft_azure.svg`,
  [Providers.Bedrock]: `${asset_logos_folder}bedrock.svg`,
  [Providers.SageMaker]: `${asset_logos_folder}bedrock.svg`,
  [Providers.Cerebras]: `${asset_logos_folder}cerebras.svg`,
  [Providers.Cohere]: `${asset_logos_folder}cohere.svg`,
  [Providers.Databricks]: `${asset_logos_folder}databricks.svg`,
  [Providers.Dashscope]: `${asset_logos_folder}dashscope.svg`,
  [Providers.Deepseek]: `${asset_logos_folder}deepseek.svg`,
  [Providers.FireworksAI]: `${asset_logos_folder}fireworks.svg`,
  [Providers.Groq]: `${asset_logos_folder}groq.svg`,
  [Providers.Google_AI_Studio]: `${asset_logos_folder}google.svg`,
  [Providers.Hosted_Vllm]: `${asset_logos_folder}vllm.png`,
  [Providers.Infinity]: `${asset_logos_folder}infinity.png`,
  [Providers.MistralAI]: `${asset_logos_folder}mistral.svg`,
  [Providers.Ollama]: `${asset_logos_folder}ollama.svg`,
  [Providers.OpenAI]: `${asset_logos_folder}openai_small.svg`,
  [Providers.OpenAI_Text]: `${asset_logos_folder}openai_small.svg`,
  [Providers.OpenAI_Text_Compatible]: `${asset_logos_folder}openai_small.svg`,
  [Providers.OpenAI_Compatible]: `${asset_logos_folder}openai_small.svg`,
  [Providers.Openrouter]: `${asset_logos_folder}openrouter.svg`,
  [Providers.Oracle]: `${asset_logos_folder}oracle.svg`,
  [Providers.Perplexity]: `${asset_logos_folder}perplexity-ai.svg`,
  [Providers.RunwayML]: `${asset_logos_folder}runwayml.png`,
  [Providers.Sambanova]: `${asset_logos_folder}sambanova.svg`,
  [Providers.Snowflake]: `${asset_logos_folder}snowflake.svg`,
  [Providers.TogetherAI]: `${asset_logos_folder}togetherai.svg`,
  [Providers.Vertex_AI]: `${asset_logos_folder}google.svg`,
  [Providers.xAI]: `${asset_logos_folder}xai.svg`,
  [Providers.GradientAI]: `${asset_logos_folder}gradientai.svg`,
  [Providers.Triton]: `${asset_logos_folder}nvidia_triton.png`,
  [Providers.Deepgram]: `${asset_logos_folder}deepgram.png`,
  [Providers.ElevenLabs]: `${asset_logos_folder}elevenlabs.png`,
  [Providers.FalAI]: `${asset_logos_folder}fal_ai.jpg`,
  [Providers.Voyage]: `${asset_logos_folder}voyage.webp`,
  [Providers.JinaAI]: `${asset_logos_folder}jina.png`,
  [Providers.VolcEngine]: `${asset_logos_folder}volcengine.png`,
  [Providers.DeepInfra]: `${asset_logos_folder}deepinfra.png`,
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
      if (
        value !== null &&
        typeof value === "object" &&
        "litellm_provider" in (value as object) &&
        ((value as any)["litellm_provider"] === custom_llm_provider ||
          (value as any)["litellm_provider"].includes(custom_llm_provider))
      ) {
        providerModels.push(key);
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
