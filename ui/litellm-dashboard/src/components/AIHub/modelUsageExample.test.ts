import { describe, expect, it } from "vitest";
import { modelUsageExample } from "./modelUsageExample";

const BASE_URL = "https://proxy.example.com";

describe("modelUsageExample", () => {
  it("posts a decision model to /v1/systemone on the proxy", () => {
    const example = modelUsageExample("evaluation", BASE_URL, "jev-latest");
    expect(example).toContain('"https://proxy.example.com/v1/systemone"');
    expect(example).toContain('"model": "jev-latest"');
    expect(example).toContain('"type": "noul"');
    expect(example).not.toContain("chat.completions");
  });

  it("keeps the chat completions example for every other mode", () => {
    for (const mode of ["chat", null, undefined]) {
      const example = modelUsageExample(mode, BASE_URL, "gpt-5.5");
      expect(example).toContain("client.chat.completions.create(");
      expect(example).toContain('base_url="https://proxy.example.com"');
      expect(example).toContain('model="gpt-5.5"');
      expect(example).not.toContain("/v1/systemone");
    }
  });

  it("escapes quotes in a model name so the snippet stays valid Python", () => {
    expect(modelUsageExample("evaluation", BASE_URL, 'team"model')).toContain('"model": "team\\"model"');
    expect(modelUsageExample("chat", BASE_URL, 'team"model')).toContain('model="team\\"model"');
  });
});
