import React from "react";

export enum Providers {
    OpenAI = "OpenAI",
    Azure = "Azure",
    Azure_AI_Studio = "Azure AI Studio",
    Anthropic = "Anthropic",
    Vertex_AI = "Vertex AI (Anthropic, Gemini, etc.)",
    Google_AI_Studio = "Google AI Studio",
    Bedrock = "Amazon Bedrock",
    Groq = "Groq",
    MistralAI = "Mistral AI",
    Deepseek = "Deepseek",
    OpenAI_Compatible = "OpenAI-Compatible Endpoints (Together AI, etc.)",
    Cohere = "Cohere",
    Databricks = "Databricks",
    Ollama = "Ollama",
    xAI = "xAI",
  }
  
export const provider_map: Record<string, string> = {
    OpenAI: "openai",
    Azure: "azure",
    Azure_AI_Studio: "azure_ai",
    Anthropic: "anthropic",
    Google_AI_Studio: "gemini",
    Bedrock: "bedrock",
    Groq: "groq",
    MistralAI: "mistral",
    Cohere: "cohere_chat",
    OpenAI_Compatible: "openai",
    Vertex_AI: "vertex_ai",
    Databricks: "databricks",
    xAI: "xai",
    Deepseek: "deepseek",
    Ollama: "ollama",
};

export const providerLogoMap: Record<string, string> = {
    [Providers.OpenAI]: "https://artificialanalysis.ai/img/logos/openai_small.svg",
    [Providers.Azure]: "https://upload.wikimedia.org/wikipedia/commons/a/a8/Microsoft_Azure_Logo.svg",
    [Providers.Azure_AI_Studio]: "https://upload.wikimedia.org/wikipedia/commons/a/a8/Microsoft_Azure_Logo.svg",
    [Providers.Anthropic]: "https://artificialanalysis.ai/img/logos/anthropic_small.svg",
    [Providers.Google_AI_Studio]: "https://artificialanalysis.ai/img/logos/google_small.svg",
    [Providers.Bedrock]: "https://artificialanalysis.ai/img/logos/aws_small.png",
    [Providers.Groq]: "https://artificialanalysis.ai/img/logos/groq_small.png",
    [Providers.MistralAI]: "https://artificialanalysis.ai/img/logos/mistral_small.png",
    [Providers.Cohere]: "https://artificialanalysis.ai/img/logos/cohere_small.png",
    [Providers.OpenAI_Compatible]: "https://upload.wikimedia.org/wikipedia/commons/4/4e/OpenAI_Logo.svg",
    [Providers.Vertex_AI]: "https://artificialanalysis.ai/img/logos/google_small.svg",
    [Providers.Databricks]: "https://artificialanalysis.ai/img/logos/databricks_small.png",
    [Providers.Ollama]: "https://artificialanalysis.ai/img/logos/ollama_small.svg",
    [Providers.xAI]: "https://artificialanalysis.ai/img/logos/xai_small.svg",
    [Providers.Deepseek]: "https://artificialanalysis.ai/img/logos/deepseek_small.jpg",
};

export const getProviderLogoAndName = (providerValue: string): { logo: string, displayName: string } => {
    if (!providerValue) {
        return { logo: "", displayName: "-" };
    }

    // Find the enum key by matching provider_map values
    const enumKey = Object.keys(provider_map).find(
        key => provider_map[key].toLowerCase() === providerValue.toLowerCase()
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
    if (selectedProvider === Providers.Vertex_AI) {
      return "gemini-pro";
    } else if (selectedProvider == Providers.Anthropic) {
      return "claude-3-opus";
    } else if (selectedProvider == Providers.Bedrock) {
      return "claude-3-opus";
    } else if (selectedProvider == Providers.Google_AI_Studio) {
      return "gemini-pro";
    } else if (selectedProvider == Providers.Azure_AI_Studio) {
      return "azure_ai/command-r-plus";
    } else if (selectedProvider == Providers.Azure) {
      return "azure/my-deployment";
    } else {
      return "gpt-3.5-turbo";
    }
  };
