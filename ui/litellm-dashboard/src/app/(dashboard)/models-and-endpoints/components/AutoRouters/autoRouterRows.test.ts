import { describe, expect, it } from "vitest";

import { autoRouterStrategy, isComplexityRouter } from "@/components/add_model/auto_router_strategies";
import { toAutoRouterRow, toAutoRouterRows } from "./autoRouterRows";

// Existing cases assert resource classification, so they run as a proxy admin: the actor
// gate is then a pass-through and canEdit/canDelete still reflect the row itself.
const ADMIN = { userRole: "Admin", userID: "u-admin" };
const TEAM_ADMIN = { userRole: "Internal User", userID: "u-team-admin" };

const complexityDeployment = {
  model_name: "tri-tier-router",
  litellm_params: {
    model: "auto_router/complexity_router",
    complexity_router_config: {
      tiers: {
        SIMPLE: ["gpt-4o-mini"],
        MEDIUM: ["anthropic-sonnet-4-6"],
        COMPLEX: ["anthropic-opus-4-6", "gpt-4o-mini"],
        REASONING: [],
      },
      classifier_type: "heuristic",
    },
    complexity_router_default_model: "gpt-4o-mini",
  },
  model_info: { id: "cid-1", db_model: true, created_at: "2026-07-28T21:40:09.900000+00:00" },
};

const semanticDeployment = {
  model_name: "support-router",
  litellm_params: {
    model: "auto_router/support-router",
    auto_router_config: JSON.stringify({
      routes: [
        { name: "gpt-4o-mini", utterances: ["reset my password"] },
        { name: "anthropic-opus-4-6", utterances: ["design a distributed system"] },
      ],
    }),
    auto_router_default_model: "gpt-4o-mini",
  },
  model_info: { id: "sid-1", db_model: true, created_at: "2026-07-27T10:00:00.000000+00:00" },
};

describe("autoRouterRows", () => {
  it("classifies a complexity router and unions its tier models as targets", () => {
    const row = toAutoRouterRow(complexityDeployment, 0, ADMIN, null);

    expect(row.kind).toBe("complexity");
    expect(row.typeLabel).toBe("Heuristic");
    // Union across tiers, de-duplicated: gpt-4o-mini appears in both SIMPLE and COMPLEX.
    expect(row.targets).toEqual(["gpt-4o-mini", "anthropic-sonnet-4-6", "anthropic-opus-4-6"]);
    expect(row.defaultModel).toBe("gpt-4o-mini");
    expect(row.id).toBe("cid-1");
  });

  it("parses a semantic router whose config arrives as a JSON string", () => {
    const row = toAutoRouterRow(semanticDeployment, 0, ADMIN, null);

    expect(row.kind).toBe("semantic");
    expect(row.typeLabel).toBe("Semantic");
    expect(row.targets).toEqual(["gpt-4o-mini", "anthropic-opus-4-6"]);
    expect(row.defaultModel).toBe("gpt-4o-mini");
  });

  it("shows a tier pinned as a bare string, which the backend accepts as `str | list[str]`", () => {
    const row = toAutoRouterRow(
      {
        ...complexityDeployment,
        litellm_params: {
          ...complexityDeployment.litellm_params,
          complexity_router_config: {
            tiers: { SIMPLE: "gpt-4o-mini", MEDIUM: ["anthropic-sonnet-4-6"], COMPLEX: "", REASONING: [] },
            classifier_type: "heuristic",
          },
        },
      },
      0,
      ADMIN,
      null,
    );

    expect(row.targets).toEqual(["gpt-4o-mini", "anthropic-sonnet-4-6"]);
  });

  it("labels a router using the LLM classifier", () => {
    const row = toAutoRouterRow(
      {
        ...complexityDeployment,
        litellm_params: {
          ...complexityDeployment.litellm_params,
          complexity_router_config: { tiers: {}, classifier_type: "llm", adaptive: true },
        },
      },
      0,
      ADMIN,
      null,
    );

    expect(row.typeLabel).toBe("LLM Classifier");
  });

  it("treats a deployment carrying complexity_router_config as complexity even off the canonical model string", () => {
    expect(isComplexityRouter({ model: "auto_router/legacy", complexity_router_config: { tiers: {} } })).toBe(true);
  });

  it("survives an unparseable config instead of throwing", () => {
    const row = toAutoRouterRow(
      {
        model_name: "broken",
        litellm_params: { model: "auto_router/broken", auto_router_config: "{not json" },
        model_info: { id: "bid-1" },
      },
      0,
      ADMIN,
      null,
    );

    expect(row.kind).toBe("semantic");
    expect(row.targets).toEqual([]);
  });

  it("falls back to a stable synthetic id when the deployment has no model_info id", () => {
    const rows = toAutoRouterRows(
      [
        { model_name: "a", litellm_params: { model: "auto_router/a" } },
        { model_name: "b", litellm_params: { model: "auto_router/b" } },
      ],
      ADMIN,
      null,
    );

    expect(rows.map((row) => row.id)).toEqual(["a-0", "b-1"]);
  });
  // Regression: adaptive and quality routers used to fall through to the semantic branch,
  // which read the wrong config key and reported an empty route list and a null default.
  it("classifies an adaptive router as adaptive, not semantic", () => {
    const row = toAutoRouterRow(
      {
        model_name: "smart-router",
        litellm_params: {
          model: "auto_router/adaptive_router",
          adaptive_router_default_model: "gpt-4o-mini",
          adaptive_router_config: { available_models: ["gpt-4o", "gpt-4o-mini"] },
        },
        model_info: { id: "ad-1" },
      },
      0,
      ADMIN,
      null,
    );

    expect(row.kind).toBe("adaptive");
    expect(row.typeLabel).toBe("Adaptive");
    expect(row.targets).toEqual(["gpt-4o", "gpt-4o-mini"]);
    expect(row.defaultModel).toBe("gpt-4o-mini");
  });

  it("classifies a quality router as quality, not semantic", () => {
    const row = toAutoRouterRow(
      {
        model_name: "quality-router",
        litellm_params: {
          model: "auto_router/quality_router",
          quality_router_default_model: "gpt-4o",
          quality_router_config: { available_models: ["gpt-4o"] },
        },
        model_info: { id: "q-1" },
      },
      0,
      ADMIN,
      null,
    );

    expect(row.kind).toBe("quality");
    expect(row.typeLabel).toBe("Quality");
    expect(row.targets).toEqual(["gpt-4o"]);
  });

  it("mirrors the backend prefix ordering, so a named strategy never reads as semantic", () => {
    const kindOf = (model: string) => autoRouterStrategy({ model }).kind;
    expect(kindOf("auto_router/complexity_router")).toBe("complexity");
    expect(kindOf("auto_router/adaptive_router")).toBe("adaptive");
    expect(kindOf("auto_router/quality_router")).toBe("quality");
    expect(kindOf("auto_router/my-own-router")).toBe("semantic");
  });

  // The capability matrix. Origin and strategy constrain DIFFERENT capabilities, and
  // collapsing them into one "editable" flag is what stranded DB-created adaptive routers
  // with no delete control. Live-verified: for a config row PATCH /model/{id}/update 404s
  // and POST /model/delete 400s.
  const rowFor = (model: string, dbModel: boolean) =>
    toAutoRouterRow(
      { model_name: "r", litellm_params: { model }, model_info: { id: "x", db_model: dbModel } },
      0,
      ADMIN,
      null,
    );

  it.each([
    { model: "auto_router/complexity_router", db: true, canEdit: true, canDelete: true, reason: null },
    { model: "auto_router/my-semantic", db: true, canEdit: true, canDelete: true, reason: null },
    // No editor for its shape, but deleting never reads the config, so delete stays.
    { model: "auto_router/adaptive_router", db: true, canEdit: false, canDelete: true, reason: "no-editor" },
    { model: "auto_router/quality_router", db: true, canEdit: false, canDelete: true, reason: "no-editor" },
    // config.yaml rows: the API refuses both, whatever the strategy.
    { model: "auto_router/complexity_router", db: false, canEdit: false, canDelete: false, reason: "config-managed" },
    { model: "auto_router/adaptive_router", db: false, canEdit: false, canDelete: false, reason: "config-managed" },
  ])("$model (db_model=$db) -> canEdit=$canEdit canDelete=$canDelete", (spec) => {
    const row = rowFor(spec.model, spec.db);
    expect(row.canEdit).toBe(spec.canEdit);
    expect(row.canDelete).toBe(spec.canDelete);
    expect(row.editBlockedReason).toBe(spec.reason);
  });

  it("treats a missing db_model as config-defined rather than assuming it is writable", () => {
    const row = toAutoRouterRow({ ...complexityDeployment, model_info: { id: "unknown-1" } }, 0, ADMIN, null);
    expect(row.canEdit).toBe(false);
    expect(row.canDelete).toBe(false);
  });
});

describe("autoRouterRows actor gating", () => {
  const TEAMS = [
    { team_id: "team-1", members_with_roles: [{ user_id: "u-team-admin", user_email: "t@t", role: "admin" }] },
  ] as never;

  const rowIn = (actor: { userRole: string; userID: string }, teamId: string | null) =>
    toAutoRouterRow(
      { ...complexityDeployment, model_info: { id: "cid-1", db_model: true, team_id: teamId } },
      0,
      actor,
      TEAMS,
    );

  // Opening the tab to team admins puts rows they cannot act on in the same list: other
  // teams' routers, and the proxy-level unscoped ones. PATCH and DELETE both 403 those, so
  // the affordance has to be per row rather than per tab.
  it("hides write affordances on another team's router", () => {
    const row = rowIn(TEAM_ADMIN, "other-team");
    expect(row.canEdit).toBe(false);
    expect(row.canDelete).toBe(false);
  });

  it("hides them on an unscoped router a proxy admin owns", () => {
    const row = rowIn(TEAM_ADMIN, null);
    expect(row.canEdit).toBe(false);
    expect(row.canDelete).toBe(false);
  });

  // Authorizing on created_by would fail this: the API lets any admin of the owning team act.
  it("keeps them on the team's router regardless of who created it", () => {
    const row = rowIn(TEAM_ADMIN, "team-1");
    expect(row.canEdit).toBe(true);
    expect(row.canDelete).toBe(true);
  });

  it("lets a proxy admin act on any team's router", () => {
    const row = rowIn(ADMIN, "other-team");
    expect(row.canEdit).toBe(true);
    expect(row.canDelete).toBe(true);
  });
});
