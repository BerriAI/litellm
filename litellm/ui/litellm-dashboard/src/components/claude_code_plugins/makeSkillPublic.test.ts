import { describe, expect, it } from "vitest";
import { toSkillPublicationChanges } from "./makeSkillPublicMapper";
import { makeSkillPublicSchema } from "./makeSkillPublicSchema";

const skills = [
  { name: "turn-off", enabled: true },
  { name: "keep-off", enabled: false },
  { name: "turn-on", enabled: false },
  { name: "keep-on", enabled: true },
];

describe("Skill publication", () => {
  it("plans only the changed skills in catalog order", () => {
    const selection = makeSkillPublicSchema.parse({ skillNames: ["turn-on", "keep-on"] });
    expect(toSkillPublicationChanges(selection, skills)).toEqual([
      { name: "turn-off", enable: false },
      { name: "turn-on", enable: true },
    ]);
  });

  it("plans no writes for an unchanged selection", () => {
    const selection = makeSkillPublicSchema.parse({ skillNames: ["keep-on", "turn-off"] });
    expect(toSkillPublicationChanges(selection, skills)).toEqual([]);
  });

  it("rejects an empty selection", () => {
    expect(makeSkillPublicSchema.safeParse({ skillNames: [] }).success).toBe(false);
  });
});
