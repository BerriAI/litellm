import { describe, expect, it } from "vitest";
import { createAccessGroupSchema } from "./createAccessGroupSchema";

const schema = createAccessGroupSchema(new Set(["premium"]));

describe("createAccessGroupSchema", () => {
  it("accepts a fresh name with at least one model and trims the name", () => {
    const result = schema.safeParse({ access_group: "  gold ", model_names: ["gold-nano"] });

    expect(result.success).toBe(true);
    expect(result.data).toEqual({ access_group: "gold", model_names: ["gold-nano"] });
  });

  it.each([
    ["", "Enter a name for the access group"],
    ["   ", "Enter a name for the access group"],
    ["openai/prod", 'A group name cannot contain "/"'],
    ["premium", "An access group with this name already exists"],
    [" premium ", "An access group with this name already exists"],
  ])("rejects the name %j", (name, message) => {
    const result = schema.safeParse({ access_group: name, model_names: ["gold-nano"] });

    expect(result.success).toBe(false);
    expect(result.error?.issues.map((issue) => issue.message)).toContain(message);
  });

  it("rejects an empty model list", () => {
    const result = schema.safeParse({ access_group: "gold", model_names: [] });

    expect(result.success).toBe(false);
    expect(result.error?.issues.map((issue) => issue.message)).toEqual(["Pick at least one model"]);
  });
});
