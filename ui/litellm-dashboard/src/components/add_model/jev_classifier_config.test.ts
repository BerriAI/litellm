import { describe, expect, it } from "vitest";
import {
  defaultLayaClassifierConfig,
  jevClassifierConfigSchema,
  normalizeJevClassifierConfig,
} from "./jev_classifier_config";

describe("Laya classifier settings", () => {
  it("preserves the Laya server when reopening, while dropping masked credentials", () => {
    const saved = { ...defaultLayaClassifierConfig(), api_base: "https://laya.example.com", api_key: "masked****" };
    const restored = jevClassifierConfigSchema.parse(saved);
    const expected = {
      provider: "laya",
      model: "english",
      timeout_ms: 3000,
      api_base: saved.api_base,
    };
    expect(normalizeJevClassifierConfig(restored)).toEqual(expected);
  });

  it("sends an explicitly entered Laya key and trimmed server URL", () => {
    const supplied = {
      ...defaultLayaClassifierConfig(),
      api_base: " https://laya.example.com/ ",
      api_key: " laya-key ",
    };
    const expected = {
      provider: "laya",
      model: "english",
      timeout_ms: 3000,
      api_base: "https://laya.example.com/",
      api_key: "laya-key",
    };
    expect(normalizeJevClassifierConfig(supplied)).toEqual(expected);
  });
});
