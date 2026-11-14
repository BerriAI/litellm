import { afterEach, expect, it, vi } from "vitest";
import { registerBaseUrlGetter } from "@/lib/http/runtime";
import { customerHierarchyContext } from "./useCustomerHierarchy";

afterEach(() => vi.unstubAllGlobals());

it("should encode hierarchy IDs in paths and preserve their identity", async () => {
  registerBaseUrlGetter(() => "http://localhost");
  const requested: string[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn<typeof fetch>(async (input) => {
      const url = new URL(input instanceof Request ? input.url : String(input), "http://localhost");
      requested.push(url.pathname);
      let record: object = { customer_team_id: "team/one", parent_org_id: "org/one" };
      if (url.pathname.includes("customer_organizations")) record = { customer_org_id: "org/one", default_budgets: [] };
      if (url.pathname.includes("customer_members"))
        record = { customer_member_id: "member/one", parent_team_id: "team/one" };
      return new Response(JSON.stringify(record), { headers: { "Content-Type": "application/json" } });
    }),
  );
  const result = await customerHierarchyContext({ org: "org/one", team: null, member: "member/one" }, "test-session");
  expect(requested).toEqual([
    "/management/v1/customer_organizations/org%2Fone",
    "/management/v1/customer_members/member%2Fone",
    "/management/v1/customer_teams/team%2Fone",
  ]);
  expect(result.entity).toEqual({
    kind: "member",
    record: { customer_member_id: "member/one", parent_team_id: "team/one" },
  });
});
