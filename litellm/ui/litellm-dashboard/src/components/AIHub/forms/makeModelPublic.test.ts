import { describe, expect, it } from "vitest";
import { toModelPublicationRequest } from "./makeModelPublicMapper";
import { makeModelPublicSchema } from "./makeModelPublicSchema";

describe("Model publication", () => {
  it("keeps the selected identifiers in selection order in the request", () => {
    const modelGroups = ["model-z", "model-a"];
    expect(toModelPublicationRequest(makeModelPublicSchema.parse({ modelGroups }))).toEqual({
      model_groups: modelGroups,
    });
  });

  it("rejects an empty selection", () => {
    expect(makeModelPublicSchema.safeParse({ modelGroups: [] }).success).toBe(false);
  });
});
