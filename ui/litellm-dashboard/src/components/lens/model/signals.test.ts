import { describe, expect, it } from "vitest";

import { systemOneModels } from "./signals";
import type { AnalysisModelInfo } from "./types";

describe("systemOneModels", () => {
  it("returns only decisions models", () => {
    const models: readonly AnalysisModelInfo[] = [
      { model_group: "decisions", providers: ["typesafe"], mode: "decisions" },
      { model_group: "chat", providers: ["openai"], mode: "chat" },
    ];

    expect(systemOneModels(models).map(({ model_group }) => model_group)).toEqual(["decisions"]);
  });
});
