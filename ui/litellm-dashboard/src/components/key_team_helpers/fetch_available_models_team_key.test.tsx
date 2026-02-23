import { describe, expect, it } from "vitest";

import { getModelDisplayName } from "./fetch_available_models_team_key";

describe("getModelDisplayName", () => {
  it("should return display label for all proxy models", () => {
    expect(getModelDisplayName("all-proxy-models")).toBe("All Proxy Models");
  });

  it("should return provider-wide label for wildcard models", () => {
    expect(getModelDisplayName("openai/*")).toBe("All openai models");
  });
});
