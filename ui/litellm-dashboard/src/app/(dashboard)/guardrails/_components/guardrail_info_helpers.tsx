import aimSecurityLogo from "../../../../../public/assets/logos/aim_security.jpeg";
import aktoLogo from "../../../../../public/assets/logos/akto.svg";
import aporiaLogo from "../../../../../public/assets/logos/aporia.png";
import bedrockLogo from "../../../../../public/assets/logos/bedrock.svg";
import catoNetworksLogo from "../../../../../public/assets/logos/cato_networks.svg";
import ciscoLogo from "../../../../../public/assets/logos/cisco.png";
import deepkeepLogo from "../../../../../public/assets/logos/deepkeep.svg";
import enkryptAiLogo from "../../../../../public/assets/logos/enkrypt_ai.avif";
import googleLogo from "../../../../../public/assets/logos/google.svg";
import guardrailsAiLogo from "../../../../../public/assets/logos/guardrails_ai.jpeg";
import javelinLogo from "../../../../../public/assets/logos/javelin.png";
import lakeraAiLogo from "../../../../../public/assets/logos/lakeraai.jpeg";
import lassoLogo from "../../../../../public/assets/logos/lasso.png";
import litellmLogo from "../../../../../public/assets/logos/litellm_logo.jpg";
import microsoftAzureLogo from "../../../../../public/assets/logos/microsoft_azure.svg";
import nomaSecurityLogo from "../../../../../public/assets/logos/noma_security.png";
import openaiSmallLogo from "../../../../../public/assets/logos/openai_small.svg";
import paloAltoNetworksLogo from "../../../../../public/assets/logos/palo_alto_networks.jpeg";
import pangeaLogo from "../../../../../public/assets/logos/pangea.png";
import pillarLogo from "../../../../../public/assets/logos/pillar.jpeg";
import promptSecurityLogo from "../../../../../public/assets/logos/prompt_security.png";
import promptguardLogo from "../../../../../public/assets/logos/promptguard.svg";
import qohashLogo from "../../../../../public/assets/logos/qohash.jpg";
import repelloAiLogo from "../../../../../public/assets/logos/repelloai.png";
import straikerLogo from "../../../../../public/assets/logos/straiker.svg";
import xecguardLogo from "../../../../../public/assets/logos/xecguard.svg";
import zscalerLogo from "../../../../../public/assets/logos/zscaler.svg";

// Legacy enum - keeping for backward compatibility
export enum GuardrailProviders {
  PresidioPII = "Presidio PII",
  Bedrock = "Bedrock Guardrail",
  Lakera = "Lakera",
}

// Dynamic guardrail providers object - populated from API response
export let DynamicGuardrailProviders: Record<string, string> = {};

// Function to populate dynamic providers from API response
export const populateGuardrailProviders = (providerParamsResponse: Record<string, any>) => {
  const providers: Record<string, string> = {};

  // Legacy hardcoded providers for backward compatibility
  providers.PresidioPII = "Presidio PII";
  providers.Bedrock = "Bedrock Guardrail";
  providers.Lakera = "Lakera";
  providers.LlmAsAJudge = "LiteLLM LLM as a Judge";

  // Add dynamic providers from API response
  Object.entries(providerParamsResponse).forEach(([key, value]) => {
    if (value && typeof value === "object" && "ui_friendly_name" in value) {
      // Create a key from the provider name (camelCase)
      const providerKey = key
        .split("_")
        .map((word, index) =>
          index === 0 ? word.charAt(0).toUpperCase() + word.slice(1) : word.charAt(0).toUpperCase() + word.slice(1),
        )
        .join("");

      providers[providerKey] = value.ui_friendly_name;
    }
  });

  DynamicGuardrailProviders = providers;
  return providers;
};

// Function to get current guardrail providers (dynamic or fallback to legacy)
export const getGuardrailProviders = () => {
  return Object.keys(DynamicGuardrailProviders).length > 0 ? DynamicGuardrailProviders : GuardrailProviders;
};

export const guardrail_provider_map: Record<string, string> = {
  PresidioPII: "presidio",
  Bedrock: "bedrock",
  Lakera: "lakera_v2",
  LitellmContentFilter: "litellm_content_filter",
  ToolPermission: "tool_permission",
  BlockCodeExecution: "block_code_execution",
  Promptguard: "promptguard",
  LlmAsAJudge: "llm_as_a_judge",
  Xecguard: "xecguard",
  Deepkeep: "deepkeep",
  QostodianNexus: "qostodian_nexus",
  Repelloai: "repelloai",
};

// Function to populate provider map from API response - updates the original map
export const populateGuardrailProviderMap = (providerParamsResponse: Record<string, any>) => {
  // Add dynamic providers from API response directly to the main map
  Object.entries(providerParamsResponse).forEach(([key, value]) => {
    if (value && typeof value === "object" && "ui_friendly_name" in value) {
      // Create a key from the provider name (camelCase)
      const providerKey = key
        .split("_")
        .map((word, index) =>
          index === 0 ? word.charAt(0).toUpperCase() + word.slice(1) : word.charAt(0).toUpperCase() + word.slice(1),
        )
        .join("");

      guardrail_provider_map[providerKey] = key; // Add directly to the main map
    }
  });
};

// Normalizes a form "mode" value (string, string[], or empty) into a string array
export const toModeArray = (raw: unknown): string[] => {
  if (Array.isArray(raw)) return raw.filter((m): m is string => typeof m === "string");
  if (typeof raw === "string") return [raw];
  return [];
};

// Resolves the supported modes for the selected provider, falling back to the global list
export const getSupportedModesForProvider = (
  settings: { supported_modes?: string[]; supported_modes_by_provider?: Record<string, string[]> } | null,
  selectedProvider: string | null,
): string[] | undefined => {
  const providerKey = selectedProvider ? guardrail_provider_map[selectedProvider]?.toLowerCase() : null;
  const perProvider =
    providerKey && settings?.supported_modes_by_provider
      ? settings.supported_modes_by_provider[providerKey]
      : undefined;
  return perProvider ?? settings?.supported_modes;
};

// Decides if we should render the PII config settings for a given provider
// For now we only support PII config settings for Presidio PII
export const shouldRenderPIIConfigSettings = (provider: string | null) => {
  if (!provider) {
    return false;
  }
  // Check both dynamic and legacy providers
  const currentProviders = getGuardrailProviders();
  const providerEnum = currentProviders[provider as keyof typeof currentProviders];
  return providerEnum === "Presidio PII";
};

// Decides if we should render the Azure Text Moderation config settings for a given provider
export const shouldRenderAzureTextModerationConfigSettings = (provider: string | null) => {
  if (!provider) {
    return false;
  }
  // Check both dynamic and legacy providers
  const currentProviders = getGuardrailProviders();
  const providerEnum = currentProviders[provider as keyof typeof currentProviders];
  return providerEnum === "Azure Content Safety Text Moderation";
};

// Decides if we should render the Content Filter config settings for a given provider
export const shouldRenderContentFilterConfigSettings = (provider: string | null) => {
  if (!provider) {
    return false;
  }
  // Check both dynamic and legacy providers
  const currentProviders = getGuardrailProviders();
  const providerEnum = currentProviders[provider as keyof typeof currentProviders];
  return providerEnum === "LiteLLM Content Filter";
};

export const shouldRenderLLMJudgeFields = (provider: string | null) => {
  if (!provider) return false;
  return guardrail_provider_map[provider] === "llm_as_a_judge";
};

export const guardrailLogoMap = {
  "Zscaler AI Guard": zscalerLogo.src,
  "Presidio PII": microsoftAzureLogo.src,
  "Bedrock Guardrail": bedrockLogo.src,
  Lakera: lakeraAiLogo.src,
  "Azure Content Safety Prompt Shield": microsoftAzureLogo.src,
  "Azure Content Safety Text Moderation": microsoftAzureLogo.src,
  "Aporia AI": aporiaLogo.src,
  "PANW Prisma AIRS": paloAltoNetworksLogo.src,
  "Cisco AI Defense": ciscoLogo.src,
  "Noma Security": nomaSecurityLogo.src,
  "Javelin Guardrails": javelinLogo.src,
  "Pillar Guardrail": pillarLogo.src,
  "Google Cloud Model Armor": googleLogo.src,
  "Guardrails AI": guardrailsAiLogo.src,
  "Lasso Guardrail": lassoLogo.src,
  "Pangea Guardrail": pangeaLogo.src,
  "AIM Guardrail": aimSecurityLogo.src,
  "Cato Networks Guardrail": catoNetworksLogo.src,
  "OpenAI Moderation": openaiSmallLogo.src,
  EnkryptAI: enkryptAiLogo.src,
  "Prompt Security": promptSecurityLogo.src,
  PromptGuard: promptguardLogo.src,
  XecGuard: xecguardLogo.src,
  "LiteLLM Content Filter": litellmLogo.src,
  "LiteLLM LLM as a Judge": litellmLogo.src,
  Akto: aktoLogo.src,
  "DeepKeep AI Firewall": deepkeepLogo.src,
  "Qostodian Nexus": qohashLogo.src,
  "RepelloAI Argus": repelloAiLogo.src,
  Straiker: straikerLogo.src,
} satisfies Record<string, string>;

export const getGuardrailLogo = (displayName: string): string | undefined =>
  Object.prototype.hasOwnProperty.call(guardrailLogoMap, displayName)
    ? guardrailLogoMap[displayName as keyof typeof guardrailLogoMap]
    : undefined;

export const getGuardrailLogoAndName = (guardrailValue: string): { logo: string; displayName: string } => {
  if (!guardrailValue) {
    return { logo: "", displayName: "-" };
  }

  // Find the enum key by matching guardrail_provider_map values
  const enumKey = Object.keys(guardrail_provider_map).find(
    (key) => guardrail_provider_map[key].toLowerCase() === guardrailValue.toLowerCase(),
  );

  if (!enumKey) {
    return { logo: "", displayName: guardrailValue };
  }

  // Get the display name from current GuardrailProviders and logo from map
  const currentProviders = getGuardrailProviders();
  const displayName = currentProviders[enumKey as keyof typeof currentProviders];
  const logo = getGuardrailLogo(displayName ?? "") ?? "";

  return { logo, displayName: displayName || guardrailValue };
};

/** Tri-state UI value for `litellm_params.skip_system_message_in_guardrail` (inherit = use global). */
export type SkipSystemMessageChoice = "inherit" | "yes" | "no";

export function skipSystemMessageToChoice(v: boolean | null | undefined): SkipSystemMessageChoice {
  if (v === true) return "yes";
  if (v === false) return "no";
  return "inherit";
}

/** Create flow: omit key when inheriting global default. */
export function choiceToSkipSystemForCreate(choice: SkipSystemMessageChoice | undefined): boolean | undefined {
  if (choice === "yes") return true;
  if (choice === "no") return false;
  return undefined;
}

/** Tri-state UI value for `litellm_params.skip_tool_message_in_guardrail` (inherit = use global). */
export type SkipToolMessageChoice = "inherit" | "yes" | "no";

export function skipToolMessageToChoice(v: boolean | null | undefined): SkipToolMessageChoice {
  if (v === true) return "yes";
  if (v === false) return "no";
  return "inherit";
}

/** Create flow: omit key when inheriting global default. */
export function choiceToSkipToolForCreate(choice: SkipToolMessageChoice | undefined): boolean | undefined {
  if (choice === "yes") return true;
  if (choice === "no") return false;
  return undefined;
}
