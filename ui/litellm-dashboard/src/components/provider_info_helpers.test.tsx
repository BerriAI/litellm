import { afterEach, describe, expect, it, vi } from "vitest";
import {
  Providers,
  getPlaceholder,
  getProviderLogoAndName,
  getProviderModels,
  providerLogoMap,
  providerMatchesModelGroup,
  provider_map,
} from "./provider_info_helpers";

describe("provider_info_helpers", () => {
  describe("getProviderLogoAndName", () => {
    it("should return empty logo and dash display name when providerValue is empty", () => {
      const result = getProviderLogoAndName("");
      expect(result).toEqual({ logo: "", displayName: "-" });
    });

    it("should return empty logo and dash display name when providerValue is undefined", () => {
      const result = getProviderLogoAndName(undefined as any);
      expect(result).toEqual({ logo: "", displayName: "-" });
    });

    it("should handle gemini provider value case-insensitively", () => {
      const result = getProviderLogoAndName("gemini");
      expect(result.displayName).toBe(Providers.Google_AI_Studio);
      expect(result.logo).toBe(providerLogoMap[Providers.Google_AI_Studio]);
    });

    it("should handle GEMINI provider value in uppercase", () => {
      const result = getProviderLogoAndName("GEMINI");
      expect(result.displayName).toBe(Providers.Google_AI_Studio);
      expect(result.logo).toBe(providerLogoMap[Providers.Google_AI_Studio]);
    });

    it("should map openai provider value to OpenAI display name and logo", () => {
      const result = getProviderLogoAndName("openai");
      expect(result.displayName).toBe(Providers.OpenAI);
      expect(result.logo).toBe(providerLogoMap[Providers.OpenAI]);
    });

    it("should map anthropic provider value to Anthropic display name and logo", () => {
      const result = getProviderLogoAndName("anthropic");
      expect(result.displayName).toBe(Providers.Anthropic);
      expect(result.logo).toBe(providerLogoMap[Providers.Anthropic]);
    });

    it("should map azure provider value to Azure display name and logo", () => {
      const result = getProviderLogoAndName("azure");
      expect(result.displayName).toBe(Providers.Azure);
      expect(result.logo).toBe(providerLogoMap[Providers.Azure]);
    });

    it("should map bedrock provider value to Bedrock display name and logo", () => {
      const result = getProviderLogoAndName("bedrock");
      expect(result.displayName).toBe(Providers.Bedrock);
      expect(result.logo).toBe(providerLogoMap[Providers.Bedrock]);
    });

    it("should map groq provider value to Groq display name and logo", () => {
      const result = getProviderLogoAndName("groq");
      expect(result.displayName).toBe(Providers.Groq);
      expect(result.logo).toBe(providerLogoMap[Providers.Groq]);
    });

    it("should handle provider values case-insensitively", () => {
      const result = getProviderLogoAndName("OPENAI");
      expect(result.displayName).toBe(Providers.OpenAI);
      expect(result.logo).toBe(providerLogoMap[Providers.OpenAI]);
    });

    it("should resolve the zai (Z.AI) provider value to the Z.AI display name", () => {
      // Regression test for https://github.com/BerriAI/litellm/issues/25482 —
      // the backend already returns `zai` from /public/providers and the docs
      // have a dedicated page, but the UI dropdown was missing an entry, so
      // `getProviderLogoAndName("zai")` previously returned the raw value as
      // the display name (no mapping).
      const result = getProviderLogoAndName("zai");
      expect(result.displayName).toBe(Providers.ZAI);
    });

    it("should return provider value as display name when no mapping exists", () => {
      const unknownProvider = "unknown_provider";
      const result = getProviderLogoAndName(unknownProvider);
      expect(result.displayName).toBe(unknownProvider);
      expect(result.logo).toBe("");
    });

    it("should map provider_map values to valid display names", () => {
      const uniqueProviderValues = new Set(Object.values(provider_map));
      uniqueProviderValues.forEach((providerValue) => {
        if (providerValue.toLowerCase() !== "gemini") {
          const result = getProviderLogoAndName(providerValue);
          expect(result.displayName).not.toBe("-");
          expect(result.displayName).toBeTruthy();
        }
      });
    });
  });

  describe("getPlaceholder", () => {
    it("should return aiml placeholder for AIML provider", () => {
      expect(getPlaceholder(Providers.AIML)).toBe("aiml/flux-pro/v1.1");
    });

    it("should return gemini-pro placeholder for Vertex_AI provider", () => {
      expect(getPlaceholder(Providers.Vertex_AI)).toBe("gemini-pro");
    });

    it("should return claude-3-opus placeholder for Anthropic provider", () => {
      expect(getPlaceholder(Providers.Anthropic)).toBe("claude-3-opus");
    });

    it("should return claude-3-opus placeholder for Bedrock provider", () => {
      expect(getPlaceholder(Providers.Bedrock)).toBe("claude-3-opus");
    });

    it("should return sagemaker placeholder for SageMaker provider", () => {
      expect(getPlaceholder(Providers.SageMaker)).toBe("sagemaker/jumpstart-dft-meta-textgeneration-llama-2-7b");
    });

    it("should return gemini-pro placeholder for Google_AI_Studio provider", () => {
      expect(getPlaceholder(Providers.Google_AI_Studio)).toBe("gemini-pro");
    });

    it("should return azure_ai placeholder for Azure_AI_Studio provider", () => {
      expect(getPlaceholder(Providers.Azure_AI_Studio)).toBe("azure_ai/command-r-plus");
    });

    it("should return my-deployment placeholder for Azure provider", () => {
      expect(getPlaceholder(Providers.Azure)).toBe("my-deployment");
    });

    it("should return oci placeholder for Oracle provider", () => {
      expect(getPlaceholder(Providers.Oracle)).toBe("oci/xai.grok-4");
    });

    it("should return snowflake placeholder for Snowflake provider", () => {
      expect(getPlaceholder(Providers.Snowflake)).toBe("snowflake/mistral-7b");
    });

    it("should return voyage placeholder for Voyage provider", () => {
      expect(getPlaceholder(Providers.Voyage)).toBe("voyage/");
    });

    it("should return jina_ai placeholder for JinaAI provider", () => {
      expect(getPlaceholder(Providers.JinaAI)).toBe("jina_ai/");
    });

    it("should return volcengine placeholder for VolcEngine provider", () => {
      expect(getPlaceholder(Providers.VolcEngine)).toBe("volcengine/<any-model-on-volcengine>");
    });

    it("should return deepinfra placeholder for DeepInfra provider", () => {
      expect(getPlaceholder(Providers.DeepInfra)).toBe("deepinfra/<any-model-on-deepinfra>");
    });

    it("should return fal_ai placeholder for FalAI provider", () => {
      expect(getPlaceholder(Providers.FalAI)).toBe("fal_ai/fal-ai/flux-pro/v1.1-ultra");
    });

    it("should return runwayml placeholder for RunwayML provider", () => {
      expect(getPlaceholder(Providers.RunwayML)).toBe("runwayml/gen4_turbo");
    });

    it("should return watsonx placeholder for Watsonx provider", () => {
      expect(getPlaceholder(Providers.WATSONX)).toBe("watsonx/ibm/granite-3-3-8b-instruct");
    });

    it("should return zai/glm-4.5 placeholder for Z.AI provider", () => {
      expect(getPlaceholder(Providers.ZAI)).toBe("zai/glm-4.5");
    });

    it("should return default gpt-3.5-turbo placeholder for unknown provider", () => {
      expect(getPlaceholder("UnknownProvider" as any)).toBe("gpt-3.5-turbo");
    });

    it("should return default gpt-3.5-turbo placeholder for OpenAI provider", () => {
      expect(getPlaceholder(Providers.OpenAI)).toBe("gpt-3.5-turbo");
    });
  });

  describe("getProviderModels", () => {
    const consoleSpy = vi.spyOn(console, "log").mockImplementation(() => {});

    afterEach(() => {
      consoleSpy.mockClear();
    });

    it("should return empty array when provider is not provided", () => {
      const modelMap = {};
      const result = getProviderModels(undefined as any, modelMap);
      expect(result).toEqual([]);
    });

    it("should return empty array when modelMap is not an object type", () => {
      const result = getProviderModels(Providers.OpenAI, "not-an-object" as any);
      expect(result).toEqual([]);
    });

    it("should return empty array when modelMap is undefined", () => {
      const result = getProviderModels(Providers.OpenAI, undefined as any);
      expect(result).toEqual([]);
    });

    it("should return empty array when modelMap is empty", () => {
      const modelMap = {};
      const result = getProviderModels(Providers.OpenAI, modelMap);
      expect(result).toEqual([]);
    });

    it("should return models matching the provider from modelMap", () => {
      const modelMap = {
        "gpt-3.5-turbo": { litellm_provider: "openai" },
        "gpt-4": { litellm_provider: "openai" },
        "claude-3-opus": { litellm_provider: "anthropic" },
      };
      const result = getProviderModels(Providers.OpenAI, modelMap);
      expect(result).toEqual(["gpt-3.5-turbo", "gpt-4"]);
    });

    it("should return models when litellm_provider includes the provider string", () => {
      const modelMap = {
        "custom-openai-model": { litellm_provider: "custom_openai_endpoint" },
        "another-model": { litellm_provider: "openai" },
      };
      const result = getProviderModels(Providers.OpenAI, modelMap);
      expect(result).toContain("custom-openai-model");
      expect(result).toContain("another-model");
    });

    it("should filter out models with null values", () => {
      const modelMap = {
        "gpt-3.5-turbo": { litellm_provider: "openai" },
        "null-model": null,
        "gpt-4": { litellm_provider: "openai" },
      };
      const result = getProviderModels(Providers.OpenAI, modelMap);
      expect(result).toEqual(["gpt-3.5-turbo", "gpt-4"]);
    });

    it("should filter out models without litellm_provider property", () => {
      const modelMap = {
        "gpt-3.5-turbo": { litellm_provider: "openai" },
        "no-provider-model": { someOtherProperty: "value" },
      };
      const result = getProviderModels(Providers.OpenAI, modelMap);
      expect(result).toEqual(["gpt-3.5-turbo"]);
    });

    it("should include both cohere and cohere_chat models for Cohere provider", () => {
      const modelMap = {
        "cohere-model-1": { litellm_provider: "cohere" },
        "cohere-model-2": { litellm_provider: "cohere_chat" },
        "other-model": { litellm_provider: "openai" },
      };
      const result = getProviderModels(Providers.Cohere, modelMap);
      expect(result).toContain("cohere-model-1");
      expect(result).toContain("cohere-model-2");
      expect(result).not.toContain("other-model");
      expect(result.length).toBeGreaterThanOrEqual(2);
    });

    it("should include sagemaker_chat models for SageMaker provider", () => {
      const modelMap = {
        "sagemaker-model-1": { litellm_provider: "sagemaker_chat" },
        "sagemaker-model-2": { litellm_provider: "sagemaker" },
        "other-model": { litellm_provider: "openai" },
      };
      const result = getProviderModels(Providers.SageMaker, modelMap);
      expect(result).toContain("sagemaker-model-1");
      expect(result).not.toContain("other-model");
    });

    it("should handle modelMap with non-object values", () => {
      const modelMap = {
        "string-model": "not-an-object",
        "number-model": 123,
        "valid-model": { litellm_provider: "openai" },
      };
      const result = getProviderModels(Providers.OpenAI, modelMap);
      expect(result).toEqual(["valid-model"]);
    });

    it("should log provider key and mapped provider when called", () => {
      const modelMap = { "gpt-3.5-turbo": { litellm_provider: "openai" } };
      getProviderModels(Providers.OpenAI, modelMap);
      expect(consoleSpy).toHaveBeenCalledWith(`Provider key: ${Providers.OpenAI}`);
      expect(consoleSpy).toHaveBeenCalledWith(`Provider mapped to: ${provider_map[Providers.OpenAI]}`);
    });

    it("should return empty array for provider with no matching models", () => {
      const modelMap = {
        "gpt-3.5-turbo": { litellm_provider: "openai" },
        "claude-3-opus": { litellm_provider: "anthropic" },
      };
      const result = getProviderModels(Providers.Bedrock, modelMap);
      expect(result).toEqual([]);
    });

    it("should handle multiple providers correctly", () => {
      const modelMap = {
        "gpt-3.5-turbo": { litellm_provider: "openai" },
        "claude-3-opus": { litellm_provider: "anthropic" },
        "groq-model": { litellm_provider: "groq" },
      };
      const openaiResult = getProviderModels(Providers.OpenAI, modelMap);
      const anthropicResult = getProviderModels(Providers.Anthropic, modelMap);
      const groqResult = getProviderModels(Providers.Groq, modelMap);

      expect(openaiResult).toEqual(["gpt-3.5-turbo"]);
      expect(anthropicResult).toEqual(["claude-3-opus"]);
      expect(groqResult).toContain("groq-model");
    });
  });

  describe("providerMatchesModelGroup (LIT-3311)", () => {
    it("matches an exact provider key", () => {
      expect(providerMatchesModelGroup("anthropic", "anthropic")).toBe(true);
      expect(providerMatchesModelGroup("openai", "openai")).toBe(true);
    });

    it("matches a hyphen-separated sub-category", () => {
      expect(providerMatchesModelGroup("vertex_ai-anthropic_models", "vertex_ai")).toBe(true);
      expect(providerMatchesModelGroup("vertex_ai-llama_models", "vertex_ai")).toBe(true);
      expect(providerMatchesModelGroup("fireworks_ai-embedding-models", "fireworks_ai")).toBe(true);
    });

    it("matches an underscore-separated sub-category", () => {
      expect(providerMatchesModelGroup("cohere_chat", "cohere")).toBe(true);
      expect(providerMatchesModelGroup("bedrock_converse", "bedrock")).toBe(true);
    });

    it("does NOT match an unanchored substring (the LIT-3311 bug)", () => {
      expect(providerMatchesModelGroup("vertex_ai-anthropic_models", "anthropic")).toBe(false);
      expect(providerMatchesModelGroup("vertex_ai-llama_models", "meta_llama")).toBe(false);
      expect(providerMatchesModelGroup("vertex_ai-mistral_models", "mistral")).toBe(false);
      expect(providerMatchesModelGroup("vertex_ai-deepseek_models", "deepseek")).toBe(false);
      expect(providerMatchesModelGroup("vertex_ai-ai21_models", "ai21")).toBe(false);
      expect(providerMatchesModelGroup("text-completion-openai", "openai")).toBe(false);
    });

    it("does NOT match when the candidate is shorter than the key", () => {
      expect(providerMatchesModelGroup("ai", "ai21")).toBe(false);
    });

    it("does NOT match when the separator is not '-' or '_'", () => {
      expect(providerMatchesModelGroup("anthropic2", "anthropic")).toBe(false);
      expect(providerMatchesModelGroup("anthropic.next", "anthropic")).toBe(false);
    });

    it("returns false on an empty key (defensive)", () => {
      expect(providerMatchesModelGroup("anything", "")).toBe(false);
    });
  });

  describe("getProviderModels - LIT-3311 regressions", () => {
    const modelMap = {
      "claude-3-opus-20240229": { litellm_provider: "anthropic" },
      "claude-3-5-sonnet-20241022": { litellm_provider: "anthropic" },
      "vertex_ai/claude-3-opus@20240229": {
        litellm_provider: "vertex_ai-anthropic_models",
      },
      "vertex_ai/claude-3-5-sonnet@20240620": {
        litellm_provider: "vertex_ai-anthropic_models",
      },
      "vertex_ai/gemini-1.5-pro": {
        litellm_provider: "vertex_ai-language-models",
      },
      "vertex_ai/llama-3.3-70b": {
        litellm_provider: "vertex_ai-llama_models",
      },
      "vertex_ai/mistral-large": {
        litellm_provider: "vertex_ai-mistral_models",
      },
      "vertex_ai/deepseek-v3": {
        litellm_provider: "vertex_ai-deepseek_models",
      },
      "gpt-4o": { litellm_provider: "openai" },
      "fireworks_ai/llama-v3p1-405b": { litellm_provider: "fireworks_ai" },
      "fireworks_ai/nomic-embed-text-v1.5": {
        litellm_provider: "fireworks_ai-embedding-models",
      },
      "cohere/command-r": { litellm_provider: "cohere_chat" },
      "cohere/embed-english-v3.0": { litellm_provider: "cohere" },
    };

    it("does not leak vertex_ai-anthropic_models into the Anthropic dropdown", () => {
      const result = getProviderModels(Providers.Anthropic, modelMap);
      expect(result).toContain("claude-3-opus-20240229");
      expect(result).toContain("claude-3-5-sonnet-20241022");
      const leaks = result.filter((m) => m.startsWith("vertex_ai/"));
      expect(leaks).toEqual([]);
    });

    it("does not leak vertex_ai-llama_models into the Meta Llama dropdown", () => {
      const result = getProviderModels(Providers.LLAMA, modelMap);
      const leaks = result.filter((m) => m.startsWith("vertex_ai/"));
      expect(leaks).toEqual([]);
    });

    it("does not leak vertex_ai-mistral_models into the MistralAI dropdown", () => {
      const result = getProviderModels(Providers.MistralAI, modelMap);
      const leaks = result.filter((m) => m.startsWith("vertex_ai/"));
      expect(leaks).toEqual([]);
    });

    it("does not leak vertex_ai-deepseek_models into the Deepseek dropdown", () => {
      const result = getProviderModels(Providers.Deepseek, modelMap);
      const leaks = result.filter((m) => m.startsWith("vertex_ai/"));
      expect(leaks).toEqual([]);
    });

    it("keeps vertex_ai-anthropic_models inside the Vertex_AI dropdown", () => {
      const result = getProviderModels(Providers.Vertex_AI, modelMap);
      expect(result).toContain("vertex_ai/claude-3-opus@20240229");
      expect(result).toContain("vertex_ai/claude-3-5-sonnet@20240620");
      expect(result).toContain("vertex_ai/gemini-1.5-pro");
      expect(result).toContain("vertex_ai/llama-3.3-70b");
    });

    it("keeps fireworks_ai-embedding-models inside the Fireworks dropdown", () => {
      const result = getProviderModels(Providers.FireworksAI, modelMap);
      expect(result).toContain("fireworks_ai/llama-v3p1-405b");
      expect(result).toContain("fireworks_ai/nomic-embed-text-v1.5");
    });
  });
});
