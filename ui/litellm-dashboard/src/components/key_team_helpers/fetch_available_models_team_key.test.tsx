import { describe, expect, it } from "vitest";

import {
  collapseModelSentinelSelection,
  getModelDisplayName,
  hasModelSentinel,
} from "./fetch_available_models_team_key";

describe("getModelDisplayName", () => {
  it("should return display label for all proxy models", () => {
    expect(getModelDisplayName("all-proxy-models")).toBe("All Proxy Models");
  });

  it("should return provider-wide label for wildcard models", () => {
    expect(getModelDisplayName("openai/*")).toBe("All openai models");
  });
});

describe("hasModelSentinel", () => {
  it.each(["all-proxy-models", "all-team-models", "no-default-models"])(
    "treats %s as exclusive, so the concrete models alongside it are unpickable",
    (sentinel) => {
      expect(hasModelSentinel([sentinel])).toBe(true);
    },
  );

  it("leaves a plain selection pickable", () => {
    expect(hasModelSentinel(["gpt-4o", "claude-sonnet-4-5"])).toBe(false);
    expect(hasModelSentinel([])).toBe(false);
  });
});

describe("collapseModelSentinelSelection", () => {
  it.each(["all-team-models", "all-proxy-models", "no-default-models"])(
    "drops the concrete models already picked when %s is chosen",
    (sentinel) => {
      expect(collapseModelSentinelSelection(["gpt-4o", sentinel])).toEqual([sentinel]);
    },
  );

  it("leaves a selection of real models alone", () => {
    expect(collapseModelSentinelSelection(["gpt-4o", "claude-sonnet-4-5"])).toEqual(["gpt-4o", "claude-sonnet-4-5"]);
    expect(collapseModelSentinelSelection([])).toEqual([]);
  });
});
