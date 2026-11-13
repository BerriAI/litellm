import { describe, expect, it } from "vitest";
import { customerLogIdentity } from "./customerIdentity";

describe("customerLogIdentity", () => {
  it("prefers persisted names and summarizes a recognized customer org and customer member", () => {
    const metadata = {
      user_api_key_customer_org_id: "org-id",
      user_api_key_customer_org_alias: "Org display",
      user_api_key_customer_team_id: "team-id",
      user_api_key_customer_team_alias: "Team display",
      user_api_key_customer_member_id: "member-id",
      user_api_key_customer_member_alias: "Member display",
    };
    const identity = customerLogIdentity(metadata, "legacy-customer");
    expect(identity.recognized).toBe(true);
    expect(identity.rows.map((row) => row.name)).toEqual(["Org display", "Team display", "Member display"]);
    expect(identity.summary.map((row) => row.name)).toEqual(["Org display", "Member display"]);
  });

  it("treats a legacy end user without a hierarchy as asserted", () => {
    expect(customerLogIdentity({}, "legacy-customer")).toEqual({
      rows: [{ scope: "legacy", name: "legacy-customer" }],
      summary: [{ scope: "legacy", name: "legacy-customer" }],
      recognized: false,
    });
  });

  it("does not recognize an alias or lone customer member ID without a resolved customer org", () => {
    const identity = customerLogIdentity({
      user_api_key_customer_org_alias: "asserted-org",
      user_api_key_customer_member_id: "member-id",
    });
    expect(identity.recognized).toBe(false);
    expect(identity.summary.map((row) => row.name)).toEqual(["asserted-org", "member-id"]);
  });

  it.each([undefined, null, {}, { user_api_key_customer_org_id: 42, user_api_key_customer_member_id: " " }])(
    "ignores missing or malformed identities (%j)",
    (metadata) => {
      expect(customerLogIdentity(metadata, " ").rows).toEqual([]);
    },
  );
});
