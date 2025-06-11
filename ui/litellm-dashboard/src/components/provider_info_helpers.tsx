import OpenAI from "openai";
import React from "react";

export enum Providers {
    OpenAI = "OpenAI",
    OpenAI_Compatible = "OpenAI-Compatible Endpoints (Together AI, etc.)",
    OpenAI_Text = "OpenAI Text Completion",
    OpenAI_Text_Compatible = "OpenAI-Compatible Text Completion Models (Together AI, etc.)",
    Azure = "Azure",
    Azure_AI_Studio = "Azure AI Foundry (Studio)",
    Anthropic = "Anthropic",
    Vertex_AI = "Vertex AI (Anthropic, Gemini, etc.)",
    Google_AI_Studio = "Google AI Studio",
    Bedrock = "Amazon Bedrock",
    Groq = "Groq",
    MistralAI = "Mistral AI",
    Deepseek = "Deepseek",
    Cohere = "Cohere",
    Databricks = "Databricks",
    Ollama = "Ollama",
    xAI = "xAI",
    AssemblyAI = "AssemblyAI",
    Cerebras = "Cerebras",
    Sambanova = "Sambanova",
    Perplexity = "Perplexity",
    TogetherAI = "TogetherAI",
    Openrouter = "Openrouter",
    FireworksAI = "Fireworks AI",
    Triton = "Triton"

  }
  
export const provider_map: Record<string, string> = {
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
    xAI: "xai",
    Deepseek: "deepseek",
    Ollama: "ollama",
    AssemblyAI: "assemblyai",
    Cerebras: "cerebras",
    Sambanova: "sambanova",
    Perplexity: "perplexity",
    TogetherAI: "together_ai",
    Openrouter: "openrouter",
    FireworksAI: "fireworks_ai",
    Triton: "triton"
};

const asset_logos_folder = '/ui/assets/logos/';

export const providerLogoMap: Record<string, string> = {
    [Providers.Anthropic]: `${asset_logos_folder}anthropic.svg`,
    [Providers.AssemblyAI]: `${asset_logos_folder}assemblyai_small.png`,
    [Providers.Azure]: `${asset_logos_folder}microsoft_azure.svg`,
    [Providers.Azure_AI_Studio]: `${asset_logos_folder}microsoft_azure.svg`,
    [Providers.Bedrock]: `${asset_logos_folder}bedrock.svg`,
    [Providers.Cerebras]: `${asset_logos_folder}cerebras.svg`,
    [Providers.Cohere]: `${asset_logos_folder}cohere.svg`,
    [Providers.Databricks]: `${asset_logos_folder}databricks.svg`,
    [Providers.Deepseek]: `${asset_logos_folder}deepseek.svg`,
    [Providers.FireworksAI]: `${asset_logos_folder}fireworks.svg`,
    [Providers.Groq]: `${asset_logos_folder}groq.svg`,
    [Providers.Google_AI_Studio]: `${asset_logos_folder}google.svg`,
    [Providers.MistralAI]: `${asset_logos_folder}mistral.svg`,
    [Providers.Ollama]: `${asset_logos_folder}ollama.svg`,
    [Providers.OpenAI]: `${asset_logos_folder}openai_small.svg`,
    [Providers.OpenAI_Text]: `${asset_logos_folder}openai_small.svg`,
    [Providers.OpenAI_Text_Compatible]: `${asset_logos_folder}openai_small.svg`,
    [Providers.OpenAI_Compatible]: `${asset_logos_folder}openai_small.svg`,
    [Providers.Openrouter]: `${asset_logos_folder}openrouter.svg`,
    [Providers.Perplexity]: `${asset_logos_folder}perplexity-ai.svg`,
    [Providers.Sambanova]: `${asset_logos_folder}sambanova.svg`,
    [Providers.TogetherAI]: `${asset_logos_folder}togetherai.svg`,
    [Providers.Vertex_AI]: `${asset_logos_folder}google.svg`,
    [Providers.xAI]: `${asset_logos_folder}xai.svg`,
    [Providers.Triton]: `${asset_logos_folder}nvidia_triton.png`
};

export const getProviderLogoAndName = (providerValue: string): { logo: string, displayName: string } => {
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
            ((value as any)["litellm_provider"] === "cohere_chat")
          ) {
            providerModels.push(key);
          }
        });
      }
    }
  
    return providerModels;
  };
