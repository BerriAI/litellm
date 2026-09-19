import type { ModelDeploymentSummary } from "@/app/(dashboard)/hooks/models/useModels";
import { describe, expect, it } from "vitest";
import { buildModelOptions, resolveSelectedModelOption, splitWildcardModels } from "./modelUtils";

describe("splitWildcardModels", () => {
  it("should return empty arrays when given empty array", () => {
    const result = splitWildcardModels([]);
    expect(result).toEqual({ wildcard: [], regular: [] });
  });

  it("should split models into wildcard and regular groups", () => {
    const models: string[] = ["gpt-4", "openai/*", "claude-3", "anthropic/*"];

    const result = splitWildcardModels(models);

    expect(result.wildcard).toHaveLength(2);
    expect(result.wildcard[0]).toBe("openai/*");
    expect(result.wildcard[1]).toBe("anthropic/*");

    expect(result.regular).toHaveLength(2);
    expect(result.regular[0]).toBe("gpt-4");
    expect(result.regular[1]).toBe("claude-3");
  });

  it("should return only wildcard models when all models are wildcard", () => {
    const models: string[] = ["openai/*", "anthropic/*"];

    const result = splitWildcardModels(models);

    expect(result.wildcard).toHaveLength(2);
    expect(result.regular).toHaveLength(0);
  });

  it("should return only regular models when no models are wildcard", () => {
    const models: string[] = ["gpt-4", "claude-3"];

    const result = splitWildcardModels(models);

    expect(result.wildcard).toHaveLength(0);
    expect(result.regular).toHaveLength(2);
  });

  it("should correctly identify wildcard models ending with /*", () => {
    const models: string[] = ["provider/*", "not-wildcard", "also-not/*/wildcard"];

    const result = splitWildcardModels(models);

    expect(result.wildcard).toHaveLength(1);
    expect(result.wildcard[0]).toBe("provider/*");
    expect(result.regular).toHaveLength(2);
  });
});

describe("buildModelOptions", () => {
  const deployments: ModelDeploymentSummary[] = [
    { id: "openai-gpt-4-id", modelName: "gpt-4", litellmModel: "openai/gpt-4" },
    { id: "azure-gpt-4-id", modelName: "gpt-4", litellmModel: "azure/gpt-4-eu" },
    { id: "claude-id", modelName: "claude-3", litellmModel: "anthropic/claude-3" },
  ];

  it("should keep a single-deployment model as one plain option", () => {
    expect(buildModelOptions(["claude-3"], deployments, false)).toEqual([
      { label: "claude-3", value: "claude-3", disabled: false },
    ]);
  });

  it("should follow a duplicated model name with one option per deployment whose value is the deployment id", () => {
    const options = buildModelOptions(["gpt-4"], deployments, true);

    expect(options.map((option) => option.value)).toEqual(["gpt-4", "openai-gpt-4-id", "azure-gpt-4-id"]);
    expect(options[0]).toMatchObject({ label: "gpt-4", description: "All 2 deployments", disabled: true });
    expect(options[2]).toMatchObject({
      label: "gpt-4 · azure/gpt-4-eu · azure-gp",
      description: "Deployment ID azure-gpt-4-id",
      disabled: true,
    });
  });

  it("should not expand deployments whose model name is not in the offered list", () => {
    expect(buildModelOptions(["claude-3"], deployments, false).map((option) => option.value)).toEqual(["claude-3"]);
  });

  it("should render an offered deployment id as that deployment", () => {
    expect(buildModelOptions(["azure-gpt-4-id"], deployments, false)).toEqual([
      {
        label: "gpt-4 · azure/gpt-4-eu · azure-gp",
        value: "azure-gpt-4-id",
        description: "Deployment ID azure-gpt-4-id",
        disabled: false,
      },
    ]);
  });

  it("should not expand a model name into deployments the caller was only granted one of by id", () => {
    const options = buildModelOptions(["azure-gpt-4-id", "gpt-4"], deployments, false);

    expect(options.map((option) => option.value)).toEqual(["azure-gpt-4-id", "gpt-4"]);
    expect(options[1]).toEqual({ label: "gpt-4", value: "gpt-4", disabled: false });
  });

  it("should apply the plain label only to model names, not to deployment rows", () => {
    const options = buildModelOptions(["gpt-4", "openai/*"], deployments, false, (name) => `label:${name}`);

    expect(options.map((option) => option.label)).toEqual([
      "label:gpt-4",
      "gpt-4 · openai/gpt-4 · openai-g",
      "gpt-4 · azure/gpt-4-eu · azure-gp",
      "label:openai/*",
    ]);
  });
});

describe("resolveSelectedModelOption", () => {
  const deployments: ModelDeploymentSummary[] = [
    { id: "azure-gpt-4-id", modelName: "gpt-4", litellmModel: "azure/gpt-4-eu" },
  ];

  it("should prefer the offered option when the value is offered", () => {
    const offered = new Map([["gpt-4", { label: "gpt-4", value: "gpt-4" }]]);
    expect(resolveSelectedModelOption("gpt-4", offered, deployments)).toBe(offered.get("gpt-4"));
  });

  it("should label a saved deployment id that is no longer offered from the deployment list", () => {
    expect(resolveSelectedModelOption("azure-gpt-4-id", new Map(), deployments)).toMatchObject({
      label: "gpt-4 · azure/gpt-4-eu · azure-gp",
      value: "azure-gpt-4-id",
    });
  });

  it("should fall back to the raw value when nothing knows it", () => {
    expect(resolveSelectedModelOption("gone-id", new Map(), deployments)).toEqual({
      label: "gone-id",
      value: "gone-id",
    });
  });
});
