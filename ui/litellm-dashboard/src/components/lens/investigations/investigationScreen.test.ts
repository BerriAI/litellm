import { describe, expect, it } from "vitest";
import { investigationScreen, type ScreenInput } from "./investigationScreen";
import type { Lens, LensList } from "../model/types";

function makeLens(id: string, created_at: string): Lens {
  return {
    version: 0,
    spent: 0,
    id,
    scope: { all_teams: true, api_key_hash: "", team_id: "" },
    settings: {
      context: "",
      source: "traces",
      lookback_hours: 24,
      service: "",
      agent_name: "",
      filters: [],
      interval_minutes: 15,
      sample_size: 100,
      sample_percent: 100,
      concurrency: 8,
      team_id: "",
      execution_ids: [],
      monthly_budget: 20,
      name: `Lens ${id}`,
      model: "analysis",
      enabled: true,
      checks: [],
    },
    revision: 1,
    created_at,
    next_run_at: created_at,
    budget_month: "2026-09",
    findings: [],
    jobs: [],
  };
}

const older = makeLens("older", "2026-09-01T10:00:00Z");
const newer = makeLens("newer", "2026-09-30T10:00:00Z");
const list = (lenses: Lens[]): LensList => ({ lenses, workers: [], tracing_enabled: true });
const ready = (lenses: Lens[]): ScreenInput["list"] => ({ data: list(lenses), error: null });
const route = { lensId: null, dialog: null, target: null } as const;

describe("investigationScreen", () => {
  it("loads, then fails only while there is nothing cached to show", () => {
    expect(investigationScreen({ ...route, list: { data: undefined, error: null } })).toEqual({ kind: "loading" });
    const error = new Error("down");
    expect(investigationScreen({ ...route, list: { data: undefined, error } })).toEqual({ kind: "failed", error });
    expect(investigationScreen({ ...route, list: { data: list([newer]), error } })).toMatchObject({ kind: "list" });
  });

  it("welcomes an empty account and lists newest investigations first otherwise", () => {
    expect(investigationScreen({ ...route, list: ready([]) })).toEqual({ kind: "welcome" });
    expect(investigationScreen({ ...route, list: ready([older, newer]) })).toEqual({
      kind: "list",
      lenses: [newer, older],
    });
  });

  it("keeps the list on screen with the selected investigation, or reports it missing", () => {
    expect(investigationScreen({ ...route, list: ready([older, newer]), lensId: "older" })).toEqual({
      kind: "list",
      lenses: [newer, older],
      lens: older,
    });
    expect(investigationScreen({ ...route, list: ready([older]), lensId: "gone" })).toEqual({ kind: "missing" });
    expect(investigationScreen({ ...route, list: { data: undefined, error: null }, lensId: "older" })).toEqual({
      kind: "loading",
    });
  });

  it("enters setup for new, edit and duplicate, prefilled from the targeted investigation", () => {
    expect(investigationScreen({ ...route, list: ready([older]), dialog: "new" })).toEqual({
      kind: "setup",
      mode: "new",
    });
    const edit: ScreenInput = { ...route, list: ready([older]), lensId: "older", dialog: "edit" };
    const editing = { kind: "setup", mode: "edit", lens: older, initial: older.settings };
    expect(investigationScreen(edit)).toEqual(editing);
    const duplicate: ScreenInput = {
      ...route,
      list: ready([older, newer]),
      lensId: "older",
      dialog: "duplicate",
      target: "newer",
    };
    const duplicating = {
      kind: "setup",
      mode: "duplicate",
      lens: newer,
      initial: { ...newer.settings, name: "Lens newer copy", enabled: false },
    };
    expect(investigationScreen(duplicate)).toEqual(duplicating);
  });

  it("falls back to the plain screen when an edit names an investigation that is not loaded", () => {
    const editGone: ScreenInput = { ...route, list: ready([older]), lensId: "gone", dialog: "edit" };
    expect(investigationScreen(editGone)).toEqual({ kind: "missing" });
    const runNow: ScreenInput = { ...route, list: ready([older]), dialog: "run_now", target: "older" };
    expect(investigationScreen(runNow)).toMatchObject({ kind: "list" });
  });
});
