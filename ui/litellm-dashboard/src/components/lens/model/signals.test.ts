import { describe, expect, it } from "vitest";

import { systemOneModels } from "./signals";
import type { AnalysisModelInfo } from "./types";

describe("systemOneModels", () => {
  it("keeps decisions and legacy evaluation models", () => {
    const models: readonly AnalysisModelInfo[] = [
      { model_group: "decisions", providers: ["typesafe"], mode: "decisions" },
      { model_group: "evaluation", providers: ["typesafe"], mode: "evaluation" },
      { model_group: "chat", providers: ["openai"], mode: "chat" },
    ];

    expect(systemOneModels(models).map(({ model_group }) => model_group)).toEqual(["decisions", "evaluation"]);
  });
});
