import { describe, expect, it } from "vitest";
import {
  computeTeamModelBadges,
  normalizeTeamModelSelection,
  TeamAccessGroupModelGrant,
  teamPerModelLimitOptions,
} from "./teamModelAccess";

const GRANTS: TeamAccessGroupModelGrant[] = [
  { access_group_id: "ag-1", access_group_name: "shared", models: ["haiku", "gpt-4o-mini"] },
  { access_group_id: "ag-2", access_group_name: "extra", models: ["haiku", "sonnet"] },
];

describe("normalizeTeamModelSelection", () => {
  it("substitutes the no-default-models sentinel for an empty selection", () => {
    expect(normalizeTeamModelSelection([])).toEqual(["no-default-models"]);
    expect(normalizeTeamModelSelection(undefined)).toEqual(["no-default-models"]);
  });

  it("passes a non-empty selection through untouched", () => {
    expect(normalizeTeamModelSelection(["gpt-4o-mini"])).toEqual(["gpt-4o-mini"]);
    expect(normalizeTeamModelSelection(["all-proxy-models"])).toEqual(["all-proxy-models"]);
  });
});

describe("computeTeamModelBadges", () => {
  it("attributes group-only models to the groups granting them", () => {
    const badges = computeTeamModelBadges(["sonnet-direct"], [], GRANTS);
    expect(badges).toEqual([
      {
        label: "sonnet-direct",
        kind: "direct",
        tooltip: "Granted directly in the team's model list",
      },
      { label: "haiku", kind: "access-group", tooltip: "Granted via access groups shared, extra" },
      { label: "gpt-4o-mini", kind: "access-group", tooltip: "Granted via access group shared" },
      { label: "sonnet", kind: "access-group", tooltip: "Granted via access group extra" },
    ]);
  });

  it("marks a model both direct and group-granted on the direct badge, without a duplicate badge", () => {
    const badges = computeTeamModelBadges(["haiku"], [], GRANTS);
    expect(badges).toEqual([
      {
        label: "haiku",
        kind: "direct",
        tooltip: "Granted directly in the team's model list, and also via access groups shared, extra",
      },
      { label: "gpt-4o-mini", kind: "access-group", tooltip: "Granted via access group shared" },
      { label: "sonnet", kind: "access-group", tooltip: "Granted via access group extra" },
    ]);
  });

  it("shows the no-default-models sentinel as its own badge and keeps group badges visible", () => {
    const badges = computeTeamModelBadges(["no-default-models"], [], [GRANTS[0]]);
    expect(badges.map((b) => [b.label, b.kind])).toEqual([
      ["No default models", "no-default"],
      ["haiku", "access-group"],
      ["gpt-4o-mini", "access-group"],
    ]);
  });

  it("still shows group badges when the empty model list grants everything", () => {
    const badges = computeTeamModelBadges([], [], [GRANTS[0]]);
    expect(badges[0]).toEqual({
      label: "All proxy models",
      kind: "all-proxy",
      tooltip: "The team's model list is empty, so it can access every model on the proxy",
    });
    expect(badges.slice(1).map((b) => b.label)).toEqual(["haiku", "gpt-4o-mini"]);
  });

  it("distinguishes the all-proxy-models sentinel from an empty list in the tooltip", () => {
    const badges = computeTeamModelBadges(["all-proxy-models"], [], []);
    expect(badges).toEqual([
      {
        label: "All proxy models",
        kind: "all-proxy",
        tooltip: "Granted by the All Proxy Models entry in the team's model list",
      },
    ]);
  });

  it("falls back to the flat access_group_models list when per-group details are absent", () => {
    const badges = computeTeamModelBadges(["direct-model"], ["haiku"], undefined);
    expect(badges).toEqual([
      { label: "direct-model", kind: "direct", tooltip: "Granted directly in the team's model list" },
      { label: "haiku", kind: "access-group", tooltip: "Granted via an access group" },
    ]);
  });
});

describe("teamPerModelLimitOptions", () => {
  const USER_MODELS = ["gpt-4o", "haiku", "openai/gpt-4o-mini", "openai/gpt-4.1"];

  it("offers access-group models, not the sentinel, for a no-default-models team", () => {
    expect(teamPerModelLimitOptions(["no-default-models"], ["gpt-4o", "haiku"], USER_MODELS)).toEqual([
      "gpt-4o",
      "haiku",
    ]);
  });

  it("merges direct and access-group models without duplicates", () => {
    expect(teamPerModelLimitOptions(["gpt-4o"], ["gpt-4o", "haiku"], USER_MODELS)).toEqual(["gpt-4o", "haiku"]);
  });

  it("offers every user model when the team covers all models", () => {
    expect(teamPerModelLimitOptions([], [], USER_MODELS)).toEqual(USER_MODELS);
    expect(teamPerModelLimitOptions(["all-proxy-models"], [], USER_MODELS)).toEqual(USER_MODELS);
    expect(teamPerModelLimitOptions(["all-team-models"], [], USER_MODELS)).toEqual(USER_MODELS);
  });

  it("expands wildcards to concrete names and drops the pattern itself", () => {
    expect(teamPerModelLimitOptions(["no-default-models"], ["openai/*"], USER_MODELS)).toEqual([
      "openai/gpt-4o-mini",
      "openai/gpt-4.1",
    ]);
  });

  it("offers nothing when the team has no models it can call", () => {
    expect(teamPerModelLimitOptions(["no-default-models"], [], USER_MODELS)).toEqual([]);
  });
});
