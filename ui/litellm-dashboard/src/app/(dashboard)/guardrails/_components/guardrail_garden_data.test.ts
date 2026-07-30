import { describe, expect, it } from "vitest";
import { ALL_CARDS, LITELLM_CONTENT_FILTER_CARDS, PARTNER_GUARDRAIL_CARDS } from "./guardrail_garden_data";

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
      expect(card.logo, `card ${card.id}`).toContain("litellm_logo.jpg");
    }
  });

  it("bundles every card logo instead of referencing runtime /ui asset paths", () => {
    for (const card of ALL_CARDS) {
      expect(card.logo, `card ${card.id}`).not.toBe("");
      expect(card.logo, `card ${card.id}`).not.toContain("/ui/assets/logos/");
    }
  });
});
