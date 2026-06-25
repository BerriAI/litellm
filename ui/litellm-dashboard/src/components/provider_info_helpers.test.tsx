import { afterEach, describe, expect, it, vi } from "vitest";
import {
  Providers,
  getPlaceholder,
  getProviderLogoAndName,
  getProviderModels,
  providerLogoMap,
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

    it("should map bedrock_mantle slug to Bedrock Mantle display name and logo", () => {
      const result = getProviderLogoAndName("bedrock_mantle");
      expect(result.displayName).toBe(Providers.BedrockMantle);
      expect(result.logo).toBe(providerLogoMap[Providers.BedrockMantle]);
    });

    it("should resolve the BedrockMantle enum key to the Bedrock Mantle logo", () => {
      // The Add Model dropdown passes the provider_map key ("BedrockMantle"),
      // not the slug ("bedrock_mantle"). Unlike "Bedrock", the key does not
      // lowercase-match its slug, so without the enum-key fallback this would
      // render a blank fallback logo for a Bedrock variant (LIT-3885).
      const result = getProviderLogoAndName("BedrockMantle");
      expect(result.displayName).toBe(Providers.BedrockMantle);
      expect(result.logo).toBe(providerLogoMap[Providers.BedrockMantle]);
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

  describe("provider logo asset paths", () => {
    // Regression: a relative "../ui/assets/logos/" base resolved to
    // "/ui/ui/assets/logos/..." (404) on the public model hub at
    // /ui/model_hub_table/, which sits a level below the /ui/ SPA. Root-absolute
    // paths resolve correctly at any route depth.
    it("should expose every provider logo as a root-absolute /ui path", () => {
      const logos = Object.values(providerLogoMap);
      expect(logos.length).toBeGreaterThan(0);
      logos.forEach((logo) => {
        expect(logo.startsWith("/ui/assets/logos/")).toBe(true);
        expect(logo).not.toContain("../");
      });
    });

    it("should resolve a provider logo to a root-absolute path via getProviderLogoAndName", () => {
      const { logo } = getProviderLogoAndName("openai");
      expect(logo.startsWith("/ui/assets/logos/")).toBe(true);
      expect(logo).not.toContain("../");
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

    it("should return models whose litellm_provider is a prefix-anchored variant of the provider", () => {
      const modelMap = {
        "anthropic-text-model": { litellm_provider: "anthropic_text" },
        "claude-3-opus": { litellm_provider: "anthropic" },
      };
      const result = getProviderModels(Providers.Anthropic, modelMap);
      expect(result).toContain("anthropic-text-model");
      expect(result).toContain("claude-3-opus");
    });

    it("should not leak vertex_ai-anthropic_models into the Anthropic provider", () => {
      const modelMap = {
        "claude-3-opus": { litellm_provider: "anthropic" },
        "vertex_ai/claude-3-5-sonnet": { litellm_provider: "vertex_ai-anthropic_models" },
        "vertex_ai/claude-haiku-4-5": { litellm_provider: "vertex_ai-anthropic_models" },
      };
      const result = getProviderModels(Providers.Anthropic, modelMap);
      expect(result).toEqual(["claude-3-opus"]);
      expect(result).not.toContain("vertex_ai/claude-3-5-sonnet");
      expect(result).not.toContain("vertex_ai/claude-haiku-4-5");
    });

    it("should not leak vertex_ai-openai_models into the OpenAI provider", () => {
      const modelMap = {
        "gpt-4": { litellm_provider: "openai" },
        "vertex_ai/openai-something": { litellm_provider: "vertex_ai-openai_models" },
      };
      const result = getProviderModels(Providers.OpenAI, modelMap);
      expect(result).toEqual(["gpt-4"]);
      expect(result).not.toContain("vertex_ai/openai-something");
    });

    // Note on the next three tests: in production, AddModelForm passes the
    // backend `provider` field (the provider_map *key*, e.g. "Vertex_AI",
    // "Bedrock", "FireworksAI") into getProviderModels, not the Providers
    // enum value. The `as Providers` cast in callers is misleading. We mirror
    // the production shape here by passing the key directly.
    it("should include all vertex_ai variants when called with 'Vertex_AI' provider key", () => {
      const modelMap = {
        "vertex_ai/gemini-pro": { litellm_provider: "vertex_ai" },
        "vertex_ai/claude-3-5-sonnet": { litellm_provider: "vertex_ai-anthropic_models" },
        "vertex_ai/text-bison": { litellm_provider: "vertex_ai-text-models" },
        "vertex_ai_beta/something": { litellm_provider: "vertex_ai_beta" },
        "anthropic-native": { litellm_provider: "anthropic" },
      };
      const result = getProviderModels("Vertex_AI" as Providers, modelMap);
      expect(result).toContain("vertex_ai/gemini-pro");
      expect(result).toContain("vertex_ai/claude-3-5-sonnet");
      expect(result).toContain("vertex_ai/text-bison");
      expect(result).toContain("vertex_ai_beta/something");
      expect(result).not.toContain("anthropic-native");
    });

    it("should include bedrock variants (converse, mantle) when called with 'Bedrock' provider key", () => {
      const modelMap = {
        "bedrock-base": { litellm_provider: "bedrock" },
        "bedrock-converse-model": { litellm_provider: "bedrock_converse" },
        "bedrock-mantle-model": { litellm_provider: "bedrock_mantle" },
        "openai-model": { litellm_provider: "openai" },
      };
      const result = getProviderModels("Bedrock" as Providers, modelMap);
      expect(result).toContain("bedrock-base");
      expect(result).toContain("bedrock-converse-model");
      expect(result).toContain("bedrock-mantle-model");
      expect(result).not.toContain("openai-model");
    });

    it("should return only bedrock_mantle models when called with 'BedrockMantle' provider key", () => {
      // Selecting "Amazon Bedrock Mantle" in the dropdown must populate the
      // model field with the Mantle models and exclude the regular Bedrock
      // ones, so onboarding a gpt-oss model is a one-click flow (LIT-3885).
      const modelMap = {
        "bedrock_mantle/openai.gpt-oss-120b": { litellm_provider: "bedrock_mantle" },
        "bedrock_mantle/openai.gpt-5.5": { litellm_provider: "bedrock_mantle" },
        "bedrock-base": { litellm_provider: "bedrock" },
        "bedrock-converse-model": { litellm_provider: "bedrock_converse" },
      };
      const result = getProviderModels("BedrockMantle" as Providers, modelMap);
      expect(result).toContain("bedrock_mantle/openai.gpt-oss-120b");
      expect(result).toContain("bedrock_mantle/openai.gpt-5.5");
      expect(result).not.toContain("bedrock-base");
      expect(result).not.toContain("bedrock-converse-model");
    });

    it("should include fireworks_ai-embedding-models when called with 'FireworksAI' provider key", () => {
      const modelMap = {
        "fireworks-base": { litellm_provider: "fireworks_ai" },
        "fireworks-embed": { litellm_provider: "fireworks_ai-embedding-models" },
        "openai-model": { litellm_provider: "openai" },
      };
      const result = getProviderModels("FireworksAI" as Providers, modelMap);
      expect(result).toContain("fireworks-base");
      expect(result).toContain("fireworks-embed");
      expect(result).not.toContain("openai-model");
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
});

describe("getProviderLogoAndName under a custom server_root_path", () => {
  afterEach(() => {
    vi.resetModules();
    vi.doUnmock("@/lib/serverRootPath");
  });

  // Regression: under SERVER_ROOT_PATH=/litellm the logo must be requested at
  // /litellm/ui/assets/logos/... A bare /ui/... path is served off the root and
  // 404s behind the reverse proxy.
  it("prefixes the server root path onto the resolved logo", async () => {
    vi.resetModules();
    vi.doMock("@/lib/serverRootPath", () => ({ serverRootPath: "/litellm" }));
    const { getProviderLogoAndName } = await import("./provider_info_helpers");
    expect(getProviderLogoAndName("openai").logo).toBe("/litellm/ui/assets/logos/openai_small.svg");
  });

  it("leaves the logo at /ui/... when mounted at the root", async () => {
    vi.resetModules();
    vi.doMock("@/lib/serverRootPath", () => ({ serverRootPath: "/" }));
    const { getProviderLogoAndName } = await import("./provider_info_helpers");
    expect(getProviderLogoAndName("openai").logo).toBe("/ui/assets/logos/openai_small.svg");
  });
});
