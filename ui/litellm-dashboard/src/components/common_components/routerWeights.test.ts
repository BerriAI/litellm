import { describe, expect, it } from "vitest";
import {
  hasNoPositiveWeight,
  parseWeightInput,
  routerWeightsFrom,
  savedRouterWeights,
  weightRows,
  withDeploymentWeight,
  withGroup,
  withoutGroup,
} from "./routerWeights";

const deployment = (id: string, modelName: string, model: string, team?: { id: string; publicName: string }) => ({
  model_name: modelName,
  litellm_params: { model },
  model_info: { id, ...(team ? { team_id: team.id, team_public_model_name: team.publicName } : {}) },
});

describe("routerWeightsFrom", () => {
  it("should keep well-formed groups and drop malformed weights from the stored blob", () => {
    expect(
      routerWeightsFrom({
        "gpt-5": { a: 9, b: 1, bad: "1", negative: -1, inf: Infinity },
        broken: [1, 2],
      }),
    ).toStrictEqual({ "gpt-5": { a: 9, b: 1 } });
  });

  it("should read missing or non-object weights as none", () => {
    expect(routerWeightsFrom(undefined)).toStrictEqual({});
    expect(routerWeightsFrom(null)).toStrictEqual({});
    expect(routerWeightsFrom("x")).toStrictEqual({});
  });
});

describe("weight edits", () => {
  it("should set, replace and clear one deployment without touching others", () => {
    const start = { "gpt-5": { a: 9 }, other: { c: 2 } };
    const added = withDeploymentWeight(start, "gpt-5", "b", 1);
    expect(added).toStrictEqual({ "gpt-5": { a: 9, b: 1 }, other: { c: 2 } });
    expect(withDeploymentWeight(added, "gpt-5", "a", 3)).toStrictEqual({ "gpt-5": { a: 3, b: 1 }, other: { c: 2 } });
    expect(withDeploymentWeight(added, "gpt-5", "a", null)).toStrictEqual({ "gpt-5": { b: 1 }, other: { c: 2 } });
    expect(start).toStrictEqual({ "gpt-5": { a: 9 }, other: { c: 2 } });
  });

  it("should add an empty group once and remove a group entirely", () => {
    const withEmpty = withGroup({ "gpt-5": { a: 9 } }, "claude");
    expect(withEmpty).toStrictEqual({ "gpt-5": { a: 9 }, claude: {} });
    expect(withGroup(withEmpty, "gpt-5")).toBe(withEmpty);
    expect(withoutGroup(withEmpty, "gpt-5")).toStrictEqual({ claude: {} });
  });
});

describe("savedRouterWeights", () => {
  it("should drop groups with no weights so they are not saved", () => {
    expect(savedRouterWeights({ "gpt-5": { a: 0, b: 2 }, claude: {} })).toStrictEqual({ "gpt-5": { a: 0, b: 2 } });
  });

  it("should save no weights as null so the field is cleared", () => {
    expect(savedRouterWeights({})).toBeNull();
    expect(savedRouterWeights({ claude: {} })).toBeNull();
  });
});

describe("hasNoPositiveWeight", () => {
  it("should flag a group whose weights are all zero, as the proxy rejects it", () => {
    expect(hasNoPositiveWeight({ a: 0, b: 0 })).toBe(true);
    expect(hasNoPositiveWeight({ a: 0, b: 0.5 })).toBe(false);
    expect(hasNoPositiveWeight({})).toBe(false);
  });
});

describe("weightRows", () => {
  it("should list the group's deployments with each one's share of traffic", () => {
    const rows = weightRows(
      "gpt-5",
      { a: 9, b: 1 },
      [
        deployment("a", "gpt-5", "openai/gpt-5"),
        deployment("b", "gpt-5", "azure/gpt-5"),
        deployment("c", "gpt-5", "bedrock/gpt-5"),
      ],
      null,
    );
    expect(rows).toStrictEqual([
      { deploymentId: "a", label: "openai/gpt-5", weight: 9, sharePercent: 90 },
      { deploymentId: "b", label: "azure/gpt-5", weight: 1, sharePercent: 10 },
      { deploymentId: "c", label: "bedrock/gpt-5", weight: null, sharePercent: 0 },
    ]);
  });

  it("should match team deployments by their public model name, not their internal routing name", () => {
    const rows = weightRows(
      "gpt-5",
      {},
      [
        deployment("team-dep", "model_name_team-1_abc", "openai/gpt-5", { id: "team-1", publicName: "gpt-5" }),
        deployment("other-group", "gpt-5-mini", "openai/gpt-5-mini"),
      ],
      "team-1",
    );
    expect(rows.map((row) => row.deploymentId)).toStrictEqual(["team-dep"]);
  });

  it("should group a deployment with no team by its model name even if it carries a public name", () => {
    const orphanedPublicName = {
      model_name: "gpt-5-internal",
      litellm_params: { model: "openai/gpt-5" },
      model_info: { id: "x", team_public_model_name: "gpt-5" },
    };
    expect(weightRows("gpt-5", {}, [orphanedPublicName], null)).toStrictEqual([]);
    expect(weightRows("gpt-5-internal", {}, [orphanedPublicName], null)).toHaveLength(1);
  });

  it("should leave out another team's deployments, which the proxy rejects", () => {
    const deployments = [
      deployment("global", "gpt-5", "openai/gpt-5"),
      deployment("team-1-dep", "model_name_team-1_abc", "openai/gpt-5", { id: "team-1", publicName: "gpt-5" }),
      deployment("team-2-dep", "model_name_team-2_def", "openai/gpt-5", { id: "team-2", publicName: "gpt-5" }),
    ];
    expect(weightRows("gpt-5", {}, deployments, "team-1").map((row) => row.deploymentId)).toStrictEqual([
      "global",
      "team-1-dep",
    ]);
    expect(weightRows("gpt-5", {}, deployments, null).map((row) => row.deploymentId)).toStrictEqual(["global"]);
  });

  it("should show no share until some weight is set", () => {
    const rows = weightRows("gpt-5", {}, [deployment("a", "gpt-5", "openai/gpt-5")], null);
    expect(rows[0].sharePercent).toBeNull();
  });

  it("should keep a weighted id whose deployment is gone so it can still be cleared", () => {
    const rows = weightRows("gpt-5", { a: 1, deleted: 1 }, [deployment("a", "gpt-5", "openai/gpt-5")], null);
    const staleRow = { deploymentId: "deleted", label: null, weight: 1, sharePercent: 50 };
    expect(rows[1]).toStrictEqual(staleRow);
  });
});

describe("parseWeightInput", () => {
  it("should accept non-negative numbers, clear on empty and reject the rest", () => {
    expect(parseWeightInput("2.5")).toBe(2.5);
    expect(parseWeightInput("0")).toBe(0);
    expect(parseWeightInput("  ")).toBeNull();
    expect(parseWeightInput("-1")).toBeUndefined();
    expect(parseWeightInput("abc")).toBeUndefined();
  });
});
