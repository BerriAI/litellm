import { describe, expect, it, beforeEach } from "vitest";
import {
  populateGuardrailProviders,
  populateGuardrailProviderMap,
  getGuardrailProviders,
  shouldRenderPIIConfigSettings,
  shouldRenderContentFilterConfigSettings,
  shouldRenderAzureTextModerationConfigSettings,
  getGuardrailLogoAndName,
  DynamicGuardrailProviders,
  guardrail_provider_map,
  GuardrailProviders,
  skipSystemMessageToChoice,
  choiceToSkipSystemForCreate,
} from "./guardrail_info_helpers";

describe("guardrail_info_helpers", () => {
  // Reset mutable module state between tests
  beforeEach(() => {
    // Clear DynamicGuardrailProviders by repopulating with empty
    Object.keys(DynamicGuardrailProviders).forEach(
      (key) => delete DynamicGuardrailProviders[key]
    );
    // Remove any dynamically added keys from guardrail_provider_map
    const staticKeys = new Set([
      "PresidioPII",
      "Bedrock",
      "Lakera",
      "LitellmContentFilter",
      "ToolPermission",
      "BlockCodeExecution",
    ]);
    Object.keys(guardrail_provider_map).forEach((key) => {
      if (!staticKeys.has(key)) delete guardrail_provider_map[key];
    });
  });

  describe("populateGuardrailProviders", () => {
    it("should populate dynamic providers from API response while preserving legacy providers", () => {
      const apiResponse = {
        zscaler_ai_guard: {
          ui_friendly_name: "Zscaler AI Guard",
          some_param: { required: true },
        },
        aporia_ai: {
          ui_friendly_name: "Aporia AI",
        },
      };

      const result = populateGuardrailProviders(apiResponse);

      // Legacy providers preserved
      expect(result.PresidioPII).toBe("Presidio PII");
      expect(result.Bedrock).toBe("Bedrock Guardrail");
      expect(result.Lakera).toBe("Lakera");

      // Dynamic providers added with PascalCase keys
      expect(result.ZscalerAiGuard).toBe("Zscaler AI Guard");
      expect(result.AporiaAi).toBe("Aporia AI");

      // Should also update the module-level DynamicGuardrailProviders
      expect(DynamicGuardrailProviders).toEqual(result);
    });

    it("should skip entries without ui_friendly_name", () => {
      const apiResponse = {
        valid_provider: { ui_friendly_name: "Valid Provider" },
        invalid_provider: { some_field: "no ui_friendly_name" },
        string_value: "not an object",
      };

      const result = populateGuardrailProviders(apiResponse);

      expect(result.ValidProvider).toBe("Valid Provider");
      expect(result.InvalidProvider).toBeUndefined();
      expect(result.StringValue).toBeUndefined();
    });
  });

  describe("getGuardrailProviders", () => {
    it("should return legacy GuardrailProviders enum when no dynamic providers are populated", () => {
      const result = getGuardrailProviders();

      expect(result).toEqual(GuardrailProviders);
      expect(result).toHaveProperty("PresidioPII", "Presidio PII");
    });

    it("should return dynamic providers when populated", () => {
      populateGuardrailProviders({
        custom_guardrail: { ui_friendly_name: "Custom Guardrail" },
      });

      const result = getGuardrailProviders();

      // Returns dynamic (which includes legacy + custom)
      expect(result.CustomGuardrail).toBe("Custom Guardrail");
      expect(result.PresidioPII).toBe("Presidio PII");
    });
  });

  describe("shouldRenderPIIConfigSettings", () => {
    it("should return true for PresidioPII provider key", () => {
      expect(shouldRenderPIIConfigSettings("PresidioPII")).toBe(true);
    });

    it("should return false for non-Presidio providers", () => {
      expect(shouldRenderPIIConfigSettings("Bedrock")).toBe(false);
      expect(shouldRenderPIIConfigSettings("Lakera")).toBe(false);
    });

    it("should return false for null provider", () => {
      expect(shouldRenderPIIConfigSettings(null)).toBe(false);
    });
  });

  describe("shouldRenderContentFilterConfigSettings", () => {
    it("should return true when dynamic providers include LiteLLM Content Filter", () => {
      populateGuardrailProviders({
        litellm_content_filter: {
          ui_friendly_name: "LiteLLM Content Filter",
        },
      });

      expect(
        shouldRenderContentFilterConfigSettings("LitellmContentFilter")
      ).toBe(true);
    });

    it("should return false for unrelated providers", () => {
      expect(shouldRenderContentFilterConfigSettings("PresidioPII")).toBe(
        false
      );
    });

    it("should return false for null", () => {
      expect(shouldRenderContentFilterConfigSettings(null)).toBe(false);
    });
  });

  describe("shouldRenderAzureTextModerationConfigSettings", () => {
    it("should return true when dynamic providers include Azure Content Safety Text Moderation", () => {
      populateGuardrailProviders({
        azure_content_safety: {
          ui_friendly_name: "Azure Content Safety Text Moderation",
        },
      });

      expect(
        shouldRenderAzureTextModerationConfigSettings("AzureContentSafety")
      ).toBe(true);
    });

    it("should return false for null", () => {
      expect(shouldRenderAzureTextModerationConfigSettings(null)).toBe(false);
    });
  });

  describe("getGuardrailLogoAndName", () => {
    it("should return correct logo and display name for a known provider value", () => {
      const result = getGuardrailLogoAndName("presidio");

      expect(result.displayName).toBe("Presidio PII");
      expect(result.logo).toContain("microsoft_azure.svg");
    });

    it("should return the raw value as displayName when provider is unknown", () => {
      const result = getGuardrailLogoAndName("unknown_provider");

      expect(result.displayName).toBe("unknown_provider");
      expect(result.logo).toBe("");
    });

    it("should return fallback for empty string", () => {
      const result = getGuardrailLogoAndName("");

      expect(result.displayName).toBe("-");
      expect(result.logo).toBe("");
    });

    it("should handle case-insensitive matching of provider values", () => {
      const lower = getGuardrailLogoAndName("presidio");
      const upper = getGuardrailLogoAndName("PRESIDIO");
      const mixed = getGuardrailLogoAndName("Presidio");

      expect(lower.displayName).toBe("Presidio PII");
      expect(upper.displayName).toBe("Presidio PII");
      expect(mixed.displayName).toBe("Presidio PII");
    });

    it("should work with dynamically populated providers", () => {
      populateGuardrailProviders({
        noma: { ui_friendly_name: "Noma Security" },
      });
      populateGuardrailProviderMap({
        noma: { ui_friendly_name: "Noma Security" },
      });

      const result = getGuardrailLogoAndName("noma");

      expect(result.displayName).toBe("Noma Security");
      expect(result.logo).toContain("noma_security.png");
    });
  });

  describe("skipSystemMessageToChoice / choiceToSkipSystemForCreate", () => {
    it("maps API values to form choices and back for create", () => {
      expect(skipSystemMessageToChoice(undefined)).toBe("inherit");
      expect(skipSystemMessageToChoice(null)).toBe("inherit");
      expect(skipSystemMessageToChoice(true)).toBe("yes");
      expect(skipSystemMessageToChoice(false)).toBe("no");

      expect(choiceToSkipSystemForCreate("inherit")).toBeUndefined();
      expect(choiceToSkipSystemForCreate(undefined)).toBeUndefined();
      expect(choiceToSkipSystemForCreate("yes")).toBe(true);
      expect(choiceToSkipSystemForCreate("no")).toBe(false);
    });
  });
});
