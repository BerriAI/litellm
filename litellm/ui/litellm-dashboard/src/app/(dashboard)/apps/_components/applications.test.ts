import { describe, expect, it } from "vitest";
import { applicationInitials, applicationRows, grantedApplicationRows } from "./applications";
import type { App, AppResource } from "./appForms";

const resource = (alias: string, appId: string | null): AppResource => ({
  resource_type: "team",
  resource_id: `team-${alias}`,
  alias,
  app_id: appId,
});
const app = (appId: string): App =>
  ({ app_id: appId, resource_type: "team", resource_id: "", alias: "", spend: 3 }) as App;

describe("application rows", () => {
  it("should list every resource by alias and attach settings only to configured ones", () => {
    const rows = applicationRows([resource("search", null), resource("chat", "app-chat")], [app("app-chat")], "");
    expect(rows.map((row) => [row.resource.alias, row.app?.app_id ?? null])).toEqual([
      ["chat", "app-chat"],
      ["search", null],
    ]);
  });

  it("should match the search against the alias or the resource id, ignoring case", () => {
    const resources = [resource("Chat", null), resource("search", null)];
    expect(applicationRows(resources, [], " cha ").map((row) => row.resource.alias)).toEqual(["Chat"]);
    expect(applicationRows(resources, [], "TEAM-SEARCH").map((row) => row.resource.alias)).toEqual(["search"]);
    expect(applicationRows(resources, [], "billing")).toEqual([]);
  });

  it("should leave a resource unconfigured when its app is missing from the loaded apps", () => {
    expect(applicationRows([resource("chat", "app-gone")], [], "")[0].app).toBeNull();
  });

  it("should build initials from the first two words", () => {
    expect(applicationInitials("customer chat bot")).toBe("CC");
    expect(applicationInitials("search")).toBe("S");
  });
});

describe("granted application rows", () => {
  it("should list each granted app by alias using the resource the app is linked to", () => {
    const granted = (alias: string, resourceId: string | null): App =>
      ({ app_id: `app-${alias}`, resource_type: "team", resource_id: resourceId, alias, spend: 3 }) as App;
    const rows = grantedApplicationRows([granted("search", "team-search"), granted("chat", "team-chat")]);
    expect(rows.map((row) => [row.resource.alias, row.resource.resource_id, row.app?.app_id])).toEqual([
      ["chat", "team-chat", "app-chat"],
      ["search", "team-search", "app-search"],
    ]);
  });

  it("should leave out an app that is no longer linked to a resource", () => {
    const unlinked = { app_id: "app-old", resource_type: "team", resource_id: null, alias: "old", spend: 0 } as App;
    expect(grantedApplicationRows([unlinked])).toEqual([]);
  });
});
