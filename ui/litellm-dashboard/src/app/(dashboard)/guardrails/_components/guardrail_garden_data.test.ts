import { describe, expect, it } from "vitest";
import { GUARDRAIL_PRESETS } from "./guardrail_garden_configs";
import {
  ALL_CARDS,
  DECISION_MODEL_CARDS,
  LITELLM_CONTENT_FILTER_CARDS,
  PARTNER_GUARDRAIL_CARDS,
} from "./guardrail_garden_data";

const EXPECTED_PARTNER_LOGO_FILES: Record<string, string> = {
  presidio: "microsoft_azure.svg",
  bedrock: "bedrock.svg",
  lakera: "lakeraai.jpeg",
  openai_moderation: "openai_small.svg",
  google_model_armor: "google.svg",
  guardrails_ai: "guardrails_ai.jpeg",
  zscaler: "zscaler.svg",
  panw: "palo_alto_networks.jpeg",
  cisco_ai_defense: "cisco.png",
  noma: "noma_security.png",
  aporia: "aporia.png",
  aim: "aim_security.jpeg",
  cato_networks: "cato_networks.svg",
  prompt_security: "prompt_security.png",
  lasso: "lasso.png",
  pangea: "pangea.png",
  enkryptai: "enkrypt_ai.avif",
  javelin: "javelin.png",
  pillar: "pillar.jpeg",
  akto: "akto.svg",
  promptguard: "promptguard.svg",
  xecguard: "xecguard.svg",
  deepkeep: "deepkeep.svg",
  repelloai: "repelloai.png",
  straiker: "straiker.svg",
  alice: "alice.svg",
  agent_365: "microsoft_azure.svg",
  llm_shield_proxy: "llm_shield_proxy.svg",
  conduct: "conduct.png",
};

describe("guardrail_garden_data logos", () => {
  it("points every partner card at its own provider's bundled logo file", () => {
    expect(new Set(PARTNER_GUARDRAIL_CARDS.map((card) => card.id))).toEqual(
      new Set(Object.keys(EXPECTED_PARTNER_LOGO_FILES)),
    );
    for (const card of PARTNER_GUARDRAIL_CARDS) {
      expect(card.logo, `card ${card.id}`).toContain(EXPECTED_PARTNER_LOGO_FILES[card.id]);
    }
  });

  it("uses the LiteLLM logo for every content filter card", () => {
    for (const card of LITELLM_CONTENT_FILTER_CARDS) {
      expect(card.logo, `card ${card.id}`).toContain("litellm_monogram.svg");
    }
  });

  it("uses each decision model card's provider logo", () => {
    const expected: Record<string, string> = {
      dm_typesafe_jev: "typesafe.png",
      dm_perplexity: "perplexity-ai.svg",
      dm_microsoft: "microsoft_azure.svg",
      dm_openai: "openai_small.svg",
      dm_databricks: "databricks.svg",
    };
    expect(new Set(DECISION_MODEL_CARDS.map((card) => card.id))).toEqual(new Set(Object.keys(expected)));
    for (const card of DECISION_MODEL_CARDS) {
      expect(card.logo, `card ${card.id}`).toContain(expected[card.id]);
    }
  });

  it("bundles every card logo instead of referencing runtime /ui asset paths", () => {
    for (const card of ALL_CARDS) {
      expect(card.logo, `card ${card.id}`).not.toBe("");
      expect(card.logo, `card ${card.id}`).not.toContain("/ui/assets/logos/");
    }
  });
});

describe("decision model presets", () => {
  it("opens each decision model card as a pre_call Decision Model guardrail scoped to its provider", () => {
    const expectedProviders: Record<string, string> = {
      dm_typesafe_jev: "typesafe",
      dm_perplexity: "perplexity",
      dm_microsoft: "azure_ai",
      dm_openai: "openai",
      dm_databricks: "databricks",
    };
    for (const card of DECISION_MODEL_CARDS) {
      const expectedPreset = {
        provider: "DecisionModel",
        guardrailNameSuggestion: card.name,
        mode: "pre_call",
        defaultOn: false,
        decisionProvider: expectedProviders[card.id],
      };
      expect(GUARDRAIL_PRESETS[card.id], `card ${card.id}`).toEqual(expectedPreset);
    }
  });
});
