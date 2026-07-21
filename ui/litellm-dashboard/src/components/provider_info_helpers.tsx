import { resolveLogoSrc } from "@/lib/assetPaths";
import a2aAgentLogo from "../../public/assets/logos/a2a_agent.png";
import ai21Logo from "../../public/assets/logos/ai21.svg";
import aimlApiLogo from "../../public/assets/logos/aiml_api.svg";
import anthropicLogo from "../../public/assets/logos/anthropic.svg";
import assemblyaiSmallLogo from "../../public/assets/logos/assemblyai_small.png";
import basetenLogo from "../../public/assets/logos/baseten.svg";
import bedrockLogo from "../../public/assets/logos/bedrock.svg";
import cerebrasLogo from "../../public/assets/logos/cerebras.svg";
import cloudflareLogo from "../../public/assets/logos/cloudflare.svg";
import cohereLogo from "../../public/assets/logos/cohere.svg";
import cometapiLogo from "../../public/assets/logos/cometapi.svg";
import cursorLogo from "../../public/assets/logos/cursor.svg";
import databricksLogo from "../../public/assets/logos/databricks.svg";
import deepgramLogo from "../../public/assets/logos/deepgram.png";
import deepinfraLogo from "../../public/assets/logos/deepinfra.png";
import deepseekLogo from "../../public/assets/logos/deepseek.svg";
import elevenlabsLogo from "../../public/assets/logos/elevenlabs.png";
import falAiLogo from "../../public/assets/logos/fal_ai.jpg";
import featherlessLogo from "../../public/assets/logos/featherless.svg";
import fireworksLogo from "../../public/assets/logos/fireworks.svg";
import friendliLogo from "../../public/assets/logos/friendli.svg";
import githubCopilotLogo from "../../public/assets/logos/github_copilot.svg";
import googleLogo from "../../public/assets/logos/google.svg";
import groqLogo from "../../public/assets/logos/groq.svg";
import huggingfaceLogo from "../../public/assets/logos/huggingface.svg";
import hyperbolicLogo from "../../public/assets/logos/hyperbolic.svg";
import infinityLogo from "../../public/assets/logos/infinity.png";
import jinaLogo from "../../public/assets/logos/jina.png";
import lambdaLogo from "../../public/assets/logos/lambda.svg";
import lmstudioLogo from "../../public/assets/logos/lmstudio.svg";
import metaLlamaLogo from "../../public/assets/logos/meta_llama.svg";
import microsoftAzureLogo from "../../public/assets/logos/microsoft_azure.svg";
import minimaxLogo from "../../public/assets/logos/minimax.svg";
import mistralLogo from "../../public/assets/logos/mistral.svg";
import moonshotLogo from "../../public/assets/logos/moonshot.svg";
import morphLogo from "../../public/assets/logos/morph.svg";
import nebiusLogo from "../../public/assets/logos/nebius.svg";
import novitaLogo from "../../public/assets/logos/novita.svg";
import nvidiaNimLogo from "../../public/assets/logos/nvidia_nim.svg";
import nvidiaTritonLogo from "../../public/assets/logos/nvidia_triton.png";
import ollamaLogo from "../../public/assets/logos/ollama.svg";
import openaiSmallLogo from "../../public/assets/logos/openai_small.svg";
import openrouterLogo from "../../public/assets/logos/openrouter.svg";
import oracleLogo from "../../public/assets/logos/oracle.svg";
import perplexityAiLogo from "../../public/assets/logos/perplexity-ai.svg";
import qwenLogo from "../../public/assets/logos/qwen.png";
import recraftLogo from "../../public/assets/logos/recraft.svg";
import replicateLogo from "../../public/assets/logos/replicate.svg";
import runwayLogo from "../../public/assets/logos/runway.png";
import sambanovaLogo from "../../public/assets/logos/sambanova.svg";
import sapLogo from "../../public/assets/logos/sap.png";
import snowflakeLogo from "../../public/assets/logos/snowflake.svg";
import sonioxLogo from "../../public/assets/logos/soniox.svg";
import togetheraiLogo from "../../public/assets/logos/togetherai.svg";
import topazLogo from "../../public/assets/logos/topaz.svg";
import v0Logo from "../../public/assets/logos/v0.svg";
import vercelLogo from "../../public/assets/logos/vercel.svg";
import vllmLogo from "../../public/assets/logos/vllm.png";
import volcengineLogo from "../../public/assets/logos/volcengine.png";
import voyageLogo from "../../public/assets/logos/voyage.webp";
import watsonxLogo from "../../public/assets/logos/watsonx.svg";
import xaiLogo from "../../public/assets/logos/xai.svg";
import xinferenceLogo from "../../public/assets/logos/xinference.svg";

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
  BedrockMantle = "Amazon Bedrock Mantle",
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
  GITHUB_COPILOT = "Github Copilot",
  Google_AI_Studio = "Google AI Studio",
  GradientAI = "GradientAI",
  Groq = "Groq",
  HEROKU = "Heroku",
  Hosted_Vllm = "vllm",
  HUGGINGFACE = "Huggingface",
  HYPERBOLIC = "Hyperbolic",
  Infinity = "Infinity",
  JinaAI = "Jina AI",
  LAMBDA_AI = "Lambda Ai",
  LEMONADE = "Lemonade",
  LLAMAFILE = "Llamafile",
  LM_STUDIO = "Lm Studio",
  LLAMA = "Meta Llama",
  MARITALK = "Maritalk",
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
  OpenAI_Compatible = "OpenAI-Compatible Chat Completions (Together AI, vLLM, etc.)",
  OpenAI_Text = "OpenAI Text Completion",
  OpenAI_Text_Compatible = "OpenAI-Compatible Completions (legacy /v1/completions)",
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
  Soniox = "Soniox",
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
  ZAI = "Z.AI (Zhipu AI)",
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
  BedrockMantle: "bedrock_mantle",
  BYTEZ: "bytez",
  Cerebras: "cerebras",
  CLARIFAI: "clarifai",
  CLOUDFLARE: "cloudflare",
  CODESTRAL: "codestral",
  Cohere: "cohere",
  COHERE_CHAT: "cohere_chat",
  COMETAPI: "cometapi",
  COMPACTIFAI: "compactifai",
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
  GITHUB_COPILOT: "github_copilot",
  Google_AI_Studio: "gemini",
  GradientAI: "gradient_ai",
  Groq: "groq",
  HEROKU: "heroku",
  Hosted_Vllm: "hosted_vllm",
  HUGGINGFACE: "huggingface",
  HYPERBOLIC: "hyperbolic",
  Infinity: "infinity",
  JinaAI: "jina_ai",
  LAMBDA_AI: "lambda_ai",
  LEMONADE: "lemonade",
  LLAMAFILE: "llamafile",
  LLAMA: "meta_llama",
  LM_STUDIO: "lm_studio",
  MARITALK: "maritalk",
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
  Soniox: "soniox",
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
  ZAI: "zai",
};

const standaloneSubproviderSlugs = new Set<string>(["bedrock_mantle"]);

export const providerLogoMap: Record<string, string> = {
  [Providers.A2A_Agent]: a2aAgentLogo.src,
  [Providers.AI21]: ai21Logo.src,
  [Providers.AI21_CHAT]: ai21Logo.src,
  [Providers.AIML]: aimlApiLogo.src,
  [Providers.AIOHTTP_OPENAI]: openaiSmallLogo.src,
  [Providers.Anthropic]: anthropicLogo.src,
  [Providers.ANTHROPIC_TEXT]: anthropicLogo.src,
  [Providers.AssemblyAI]: assemblyaiSmallLogo.src,
  [Providers.Azure]: microsoftAzureLogo.src,
  [Providers.Azure_AI_Studio]: microsoftAzureLogo.src,
  [Providers.AZURE_TEXT]: microsoftAzureLogo.src,
  [Providers.BASETEN]: basetenLogo.src,
  [Providers.Bedrock]: bedrockLogo.src,
  [Providers.BedrockMantle]: bedrockLogo.src,
  [Providers.SageMaker]: bedrockLogo.src,
  [Providers.Cerebras]: cerebrasLogo.src,
  [Providers.CLOUDFLARE]: cloudflareLogo.src,
  [Providers.CODESTRAL]: mistralLogo.src,
  [Providers.Cohere]: cohereLogo.src,
  [Providers.COHERE_CHAT]: cohereLogo.src,
  [Providers.COMETAPI]: cometapiLogo.src,
  [Providers.Cursor]: cursorLogo.src,
  [Providers.Databricks]: databricksLogo.src,
  [Providers.Dashscope]: qwenLogo.src,
  [Providers.Deepseek]: deepseekLogo.src,
  [Providers.Deepgram]: deepgramLogo.src,
  [Providers.DeepInfra]: deepinfraLogo.src,
  [Providers.ElevenLabs]: elevenlabsLogo.src,
  [Providers.FalAI]: falAiLogo.src,
  [Providers.FEATHERLESS_AI]: featherlessLogo.src,
  [Providers.FireworksAI]: fireworksLogo.src,
  [Providers.FRIENDLIAI]: friendliLogo.src,
  [Providers.GITHUB_COPILOT]: githubCopilotLogo.src,
  [Providers.Google_AI_Studio]: googleLogo.src,
  [Providers.Groq]: groqLogo.src,
  [Providers.Hosted_Vllm]: vllmLogo.src,
  [Providers.HUGGINGFACE]: huggingfaceLogo.src,
  [Providers.HYPERBOLIC]: hyperbolicLogo.src,
  [Providers.Infinity]: infinityLogo.src,
  [Providers.JinaAI]: jinaLogo.src,
  [Providers.LAMBDA_AI]: lambdaLogo.src,
  [Providers.LM_STUDIO]: lmstudioLogo.src,
  [Providers.LLAMA]: metaLlamaLogo.src,
  [Providers.MiniMax]: minimaxLogo.src,
  [Providers.MistralAI]: mistralLogo.src,
  [Providers.MOONSHOT]: moonshotLogo.src,
  [Providers.MORPH]: morphLogo.src,
  [Providers.NEBIUS]: nebiusLogo.src,
  [Providers.NOVITA]: novitaLogo.src,
  [Providers.NVIDIA_NIM]: nvidiaNimLogo.src,
  [Providers.Ollama]: ollamaLogo.src,
  [Providers.OLLAMA_CHAT]: ollamaLogo.src,
  [Providers.OOBABOOGA]: openaiSmallLogo.src,
  [Providers.OpenAI]: openaiSmallLogo.src,
  [Providers.OPENAI_LIKE]: openaiSmallLogo.src,
  [Providers.OpenAI_Text]: openaiSmallLogo.src,
  [Providers.OpenAI_Text_Compatible]: openaiSmallLogo.src,
  [Providers.OpenAI_Compatible]: openaiSmallLogo.src,
  [Providers.Openrouter]: openrouterLogo.src,
  [Providers.Oracle]: oracleLogo.src,
  [Providers.Perplexity]: perplexityAiLogo.src,
  [Providers.RECRAFT]: recraftLogo.src,
  [Providers.REPLICATE]: replicateLogo.src,
  [Providers.RunwayML]: runwayLogo.src,
  [Providers.SAGEMAKER_LEGACY]: bedrockLogo.src,
  [Providers.Sambanova]: sambanovaLogo.src,
  [Providers.SAP]: sapLogo.src,
  [Providers.Snowflake]: snowflakeLogo.src,
  [Providers.Soniox]: sonioxLogo.src,
  [Providers.TEXT_COMPLETION_CODESTRAL]: mistralLogo.src,
  [Providers.TogetherAI]: togetheraiLogo.src,
  [Providers.TOPAZ]: topazLogo.src,
  [Providers.Triton]: nvidiaTritonLogo.src,
  [Providers.V0]: v0Logo.src,
  [Providers.VERCEL_AI_GATEWAY]: vercelLogo.src,
  [Providers.Vertex_AI]: googleLogo.src,
  [Providers.VERTEX_AI_BETA]: googleLogo.src,
  [Providers.VLLM]: vllmLogo.src,
  [Providers.VolcEngine]: volcengineLogo.src,
  [Providers.Voyage]: voyageLogo.src,
  [Providers.WATSONX]: watsonxLogo.src,
  [Providers.WATSONX_TEXT]: watsonxLogo.src,
  [Providers.xAI]: xaiLogo.src,
  [Providers.XINFERENCE]: xinferenceLogo.src,
};

export const getProviderLogoAndName = (providerValue: string): { logo: string; displayName: string } => {
  if (!providerValue) {
    return { logo: "", displayName: "-" };
  }

  // Handle special case for "gemini" provider value
  if (providerValue.toLowerCase() === "gemini") {
    const displayName = Providers.Google_AI_Studio;
    const logo = resolveLogoSrc(providerLogoMap[displayName]) ?? "";
    return { logo, displayName };
  }

  // Resolve by the litellm provider slug (e.g. "bedrock_mantle"); fall back to
  // the enum key (e.g. "BedrockMantle") for callers like the Add Model dropdown
  // that pass the key instead of the slug.
  const enumKey =
    Object.keys(provider_map).find((key) => provider_map[key].toLowerCase() === providerValue.toLowerCase()) ??
    Object.keys(provider_map).find((key) => key.toLowerCase() === providerValue.toLowerCase());

  if (!enumKey) {
    return { logo: "", displayName: providerValue };
  }

  // Get the display name from Providers enum and logo from map
  const displayName = Providers[enumKey as keyof typeof Providers];
  const logo = resolveLogoSrc(providerLogoMap[displayName as keyof typeof providerLogoMap]) ?? "";

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
  } else if (selectedProvider === Providers.ZAI) {
    return "zai/glm-4.5";
  } else {
    return "gpt-3.5-turbo";
  }
};

export const getProviderModels = (provider: Providers, modelMap: any): Array<string> => {
  let providerKey = provider;
  let custom_llm_provider = provider_map[providerKey];

  let providerModels: Array<string> = [];

  if (providerKey && typeof modelMap === "object") {
    Object.entries(modelMap).forEach(([key, value]) => {
      if (value !== null && typeof value === "object" && "litellm_provider" in (value as object)) {
        const litellmProvider = (value as any)["litellm_provider"];
        const isPrefixVariant =
          typeof litellmProvider === "string" &&
          (litellmProvider.startsWith(`${custom_llm_provider}_`) ||
            litellmProvider.startsWith(`${custom_llm_provider}-`));
        if (
          litellmProvider === custom_llm_provider ||
          (isPrefixVariant && !standaloneSubproviderSlugs.has(litellmProvider))
        ) {
          providerModels.push(key);
        }
      }
    });
    // Special case for cohere
    // we need both cohere_chat and cohere models to show on dropdown
    if (providerKey == Providers.Cohere) {
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
