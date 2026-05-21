import { describe, expect, it } from "vitest";
import { ALL_CARDS } from "./guardrail_garden_data";
import { GUARDRAIL_PRESETS } from "./guardrail_garden_configs";

describe("Silmaril Guardrail Garden metadata", () => {
  it("should include the Silmaril partner card and preset", () => {
    const card = ALL_CARDS.find((card) => card.id === "silmaril");

    expect(card).toBeDefined();
    expect(card?.name).toBe("Silmaril Firewall");
    expect(card?.logo).toContain("silmaril.png");
    expect(card?.providerKey).toBe("Silmaril");
    expect(GUARDRAIL_PRESETS.silmaril).toMatchObject({
      provider: "Silmaril",
      guardrailNameSuggestion: "Silmaril Firewall",
      mode: "pre_call",
      defaultOn: false,
    });
  });
});
