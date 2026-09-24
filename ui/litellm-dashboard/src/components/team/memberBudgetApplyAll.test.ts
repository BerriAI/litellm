import { describe, expect, it } from "vitest";
import {
  chunkMemberIds,
  customBudgetMemberUserIds,
  MEMBER_BUDGET_BULK_CHUNK_SIZE,
  shouldPromptMemberBudgetApplyAll,
} from "./memberBudgetApplyAll";

describe("customBudgetMemberUserIds", () => {
  it("returns only members whose budget_source is custom", () => {
    const memberships = [
      { user_id: "u-custom", budget_source: "custom" },
      { user_id: "u-default", budget_source: "team_default" },
      { user_id: "u-none", budget_source: "none" },
    ];

    expect(customBudgetMemberUserIds(memberships)).toEqual(["u-custom"]);
  });

  it("skips a custom-budget row that carries no user_id", () => {
    const memberships = [
      { user_id: null, budget_source: "custom" },
      { user_id: "u-custom", budget_source: "custom" },
    ];

    expect(customBudgetMemberUserIds(memberships)).toEqual(["u-custom"]);
  });
});

describe("shouldPromptMemberBudgetApplyAll", () => {
  it("prompts when the default changes and custom-budget members exist", () => {
    expect(shouldPromptMemberBudgetApplyAll(10, 5, ["u-1"])).toBe(true);
  });

  it("prompts when a default is set for the first time", () => {
    expect(shouldPromptMemberBudgetApplyAll(10, null, ["u-1"])).toBe(true);
    expect(shouldPromptMemberBudgetApplyAll(10, undefined, ["u-1"])).toBe(true);
  });

  it("does not prompt when the submitted budget is unchanged", () => {
    expect(shouldPromptMemberBudgetApplyAll(10, 10, ["u-1"])).toBe(false);
  });

  it("does not prompt when the budget is cleared or zero", () => {
    expect(shouldPromptMemberBudgetApplyAll(undefined, 10, ["u-1"])).toBe(false);
    expect(shouldPromptMemberBudgetApplyAll(0, 10, ["u-1"])).toBe(false);
  });

  it("does not prompt when no member has a custom budget", () => {
    expect(shouldPromptMemberBudgetApplyAll(10, 5, [])).toBe(false);
  });
});

describe("chunkMemberIds", () => {
  it("splits selections larger than the bulk endpoint limit", () => {
    const ids = Array.from({ length: MEMBER_BUDGET_BULK_CHUNK_SIZE + 1 }, (_, i) => `u-${i}`);

    const chunks = chunkMemberIds(ids);

    expect(chunks).toHaveLength(2);
    expect(chunks[0]).toHaveLength(MEMBER_BUDGET_BULK_CHUNK_SIZE);
    expect(chunks[1]).toEqual([`u-${MEMBER_BUDGET_BULK_CHUNK_SIZE}`]);
  });

  it("keeps a selection under the limit in a single chunk", () => {
    expect(chunkMemberIds(["u-1", "u-2"])).toEqual([["u-1", "u-2"]]);
  });

  it("returns no chunks for an empty selection", () => {
    expect(chunkMemberIds([])).toEqual([]);
  });
});
