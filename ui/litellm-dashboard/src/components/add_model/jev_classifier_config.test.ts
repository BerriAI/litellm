import { describe, expect, it } from "vitest";
import {
  defaultJevClassifierConfig,
  fixedClassifierModels,
  hydrateOssClassifier,
  isOssClassifierProvider,
  jevClassifierConfigSchema,
} from "./jev_classifier_config";

describe("OSS classifier provider defaults", () => {
  it.each([
    ["jev", "jev-latest", undefined],
    ["laya", "english", ["english", "multilingual", "typed-decisions"]],
    ["bespoke", "nimble-latest", ["nimble-latest", "nimble", "bespokelabs/Bespoke-Nimble-9B"]],
    ["strands_decider", "strands-decider-2B-hobson-v19", undefined],
    ["cloudflare", "clef", ["clef", "clef-flash"]],
  ] as const)("defaults %s to %s", (provider, model, fixed) => {
    expect(defaultJevClassifierConfig(provider)).toEqual({ provider, model, timeout_ms: 3000 });
    expect(fixedClassifierModels(provider)).toEqual(fixed);
    expect(isOssClassifierProvider(provider)).toBe(true);
  });

  it("keeps a provider-specific model that was stored", () => {
    expect(jevClassifierConfigSchema.parse({ provider: "cloudflare", model: "clef-flash" }).model).toBe("clef-flash");
    expect(jevClassifierConfigSchema.parse({ provider: "strands_decider", model: "custom-checkpoint" }).model).toBe(
      "custom-checkpoint",
    );
    expect(jevClassifierConfigSchema.safeParse({ provider: "cloudflare", model: "nimble" }).success).toBe(false);
  });

  it("hydrates a saved Decisions provider config into the editor", () => {
    expect(
      hydrateOssClassifier({
        classifier_type: "oss_classifier",
        opensource_classifier_config: { provider: "cloudflare", model: "clef-flash", timeout_ms: 900 },
      }),
    ).toEqual({
      classifier_type: "jev",
      jev_classifier_config: { provider: "cloudflare", model: "clef-flash", timeout_ms: 900 },
    });
  });

  it.each(["typesafe", "", undefined, 3])("rejects %s as a provider radio value", (value) => {
    expect(isOssClassifierProvider(value)).toBe(false);
  });
});
