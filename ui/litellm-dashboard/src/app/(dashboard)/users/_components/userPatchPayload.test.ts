import { describe, expect, it } from "vitest";
import { toUserPatch } from "./userPatchPayload";

describe("toUserPatch", () => {
  it("sends null for every control the operator emptied", () => {
    const emptied = {
      user_email: "",
      user_alias: "",
      budget_duration: undefined,
      max_budget: null,
      tpm_limit: "",
      rpm_limit: "",
      metadata: "",
    };

    expect(toUserPatch(emptied)).toEqual({
      user_email: null,
      user_alias: null,
      budget_duration: null,
      max_budget: null,
      tpm_limit: null,
      rpm_limit: null,
      metadata: null,
    });
  });

  it("keeps the values the operator did set", () => {
    const filled = {
      user_email: "someone@example.com",
      user_alias: "someone",
      user_role: "internal_user" as const,
      models: ["gpt-4o"],
      max_budget: "12.5",
      budget_duration: "30d",
      tpm_limit: "1000",
      rpm_limit: "60",
      metadata: { team: "core" },
      model_max_budget: { "gpt-4o": { budget_limit: 1, time_period: "1d" } },
    };

    expect(toUserPatch(filled)).toEqual({
      user_email: "someone@example.com",
      user_alias: "someone",
      user_role: "internal_user",
      models: ["gpt-4o"],
      max_budget: 12.5,
      budget_duration: "30d",
      tpm_limit: 1000,
      rpm_limit: 60,
      metadata: { team: "core" },
      model_max_budget: { "gpt-4o": { budget_limit: 1, time_period: "1d" } },
    });
  });

  it("converts numeric strings so the proxy is not handed a string limit", () => {
    const patch = toUserPatch({ tpm_limit: "1000", rpm_limit: "60", max_budget: "12.5" });

    expect(patch.tpm_limit).toBe(1000);
    expect(patch.rpm_limit).toBe(60);
    expect(patch.max_budget).toBe(12.5);
  });

  it("omits fields the form never rendered rather than clearing them", () => {
    expect(toUserPatch({ user_alias: "someone" })).toEqual({ user_alias: "someone" });
  });

  it("omits an empty role, since the dropdown offers no way to clear one", () => {
    expect(toUserPatch({ user_role: null })).toEqual({});
    expect(toUserPatch({ user_role: undefined })).toEqual({});
  });

  it("sends an empty model list, which is how personal models get revoked", () => {
    expect(toUserPatch({ models: [] })).toEqual({ models: [] });
  });

  it("omits model_max_budget when the editor reported no change", () => {
    expect(toUserPatch({ model_max_budget: undefined })).toEqual({});
  });

  it("drops the form-only keys the endpoint refuses with a 422", () => {
    const patch = toUserPatch({
      user_id: "u-1",
      mcp_servers_and_groups: { servers: [], accessGroups: [], toolsets: [] },
      mcp_tool_permissions: {},
      user_alias: "someone",
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } as any);

    expect(patch).toEqual({ user_alias: "someone" });
  });

  it("keeps a zero limit, which is a real setting and not an empty field", () => {
    expect(toUserPatch({ tpm_limit: 0, rpm_limit: "0", max_budget: 0 })).toEqual({
      tpm_limit: 0,
      rpm_limit: 0,
      max_budget: 0,
    });
  });
});
