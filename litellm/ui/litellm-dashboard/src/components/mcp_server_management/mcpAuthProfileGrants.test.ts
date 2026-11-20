import { describe, expect, it } from "vitest";
import type { MCPServer, MCPToolset } from "@/components/mcp_tools/types";
import {
  authProfileGrantEntry,
  grantChangeNeedsServers,
  grantedProfileFor,
  profileServers,
  reachableServers,
  withServerProfile,
} from "./mcpAuthProfileGrants";

const server = (id: string, profileIds: readonly string[]): MCPServer =>
  ({
    server_id: id,
    server_name: id,
    additional_auth: profileIds.map((profileId) => ({ id: profileId, auth_type: "api_key" })),
  }) as MCPServer;

const tickets = server("tickets", ["tickets-bots", "tickets-ci"]);
const wiki = server("wiki", ["wiki-bots"]);
const plain = server("plain", []);
const grouped = server("grouped", ["grp-bots"]);
const catalog = [tickets, wiki, plain, grouped];

describe("profileServers", () => {
  it("offers every server that has profiles, since a grant never depends on how the holder reaches it", () => {
    expect(profileServers(catalog)).toEqual([tickets, wiki, grouped]);
  });
});

describe("reachableServers", () => {
  const ticketsRead: MCPToolset = {
    toolset_id: "tickets-read",
    toolset_name: "tickets-read",
    tools: [{ server_id: "tickets", tool_name: "list_tickets" }],
  };
  const reach = { scoped: [wiki], all: catalog, serverIds: [], toolsets: [ticketsRead], toolsetIds: [] };

  it("adds a server the holder reaches only through a granted toolset", () => {
    expect(reachableServers({ ...reach, toolsetIds: ["tickets-read"] })).toEqual([wiki, tickets]);
  });

  it("adds a server granted directly outside the scoped list, once", () => {
    expect(reachableServers({ ...reach, serverIds: ["tickets", "wiki"], toolsetIds: ["tickets-read"] })).toEqual([
      wiki,
      tickets,
    ]);
  });

  it("leaves out servers of toolsets the holder was not granted", () => {
    expect(reachableServers(reach)).toEqual([wiki]);
  });
});

describe("withServerProfile", () => {
  it("replaces the earlier choice for the same server, so two ids for one server cannot happen", () => {
    expect(withServerProfile(["tickets-bots", "wiki-bots"], tickets, "tickets-ci")).toEqual([
      "wiki-bots",
      "tickets-ci",
    ]);
  });

  it("clears a server's choice when set back to the default profile", () => {
    expect(withServerProfile(["tickets-bots", "wiki-bots"], tickets, null)).toEqual(["wiki-bots"]);
  });
});

describe("grantedProfileFor", () => {
  it("finds the grant that belongs to the server, ignoring case", () => {
    expect(grantedProfileFor(["WIKI-Bots", "tickets-ci"], tickets)).toBe("tickets-ci");
    expect(grantedProfileFor(["WIKI-Bots"], wiki)).toBe("WIKI-Bots");
    expect(grantedProfileFor(["wiki-bots"], tickets)).toBeUndefined();
  });
});

describe("authProfileGrantEntry", () => {
  it("adds no key when nothing is granted, so the payload is today's", () => {
    expect(Object.hasOwn(authProfileGrantEntry({ grants: undefined, servers: catalog }), "mcp_auth_profiles")).toBe(
      false,
    );
    expect(Object.hasOwn(authProfileGrantEntry({ grants: [], servers: catalog }), "mcp_auth_profiles")).toBe(false);
  });

  it("sends every chosen grant on a create, whichever servers the selection names", () => {
    expect(authProfileGrantEntry({ grants: ["tickets-bots", "wiki-bots"], servers: catalog })).toEqual({
      mcp_auth_profiles: ["tickets-bots", "wiki-bots"],
    });
  });

  it("keeps a grant whose server the holder reaches outside its own selection", () => {
    const unrelatedSave = { grants: ["tickets-bots"], servers: catalog, loadedGrants: ["tickets-bots"] };
    expect(authProfileGrantEntry(unrelatedSave)).toEqual({});
  });

  it("leaves the key out when the form never carried the grant field", () => {
    const fieldAbsent = { grants: undefined, servers: catalog, loadedGrants: ["tickets-bots"] };
    expect(authProfileGrantEntry(fieldAbsent)).toEqual({});
  });

  it("leaves the key out when a non-admin resaves the grant their catalog hides", () => {
    const hiddenProfiles = catalog.map((server) => ({ ...server, additional_auth: null }));
    const nonAdminResave = { grants: ["tickets-bots"], servers: hiddenProfiles, loadedGrants: ["tickets-bots"] };
    expect(authProfileGrantEntry(nonAdminResave)).toEqual({});
  });

  it("keeps a stored grant the picker cannot show alongside a new choice", () => {
    const pickedAfterMove = { grants: ["wiki-bots"], servers: [wiki], loadedGrants: ["tickets-bots"] };
    expect(authProfileGrantEntry(pickedAfterMove)).toEqual({ mcp_auth_profiles: ["wiki-bots", "tickets-bots"] });
  });

  it("sends an empty list on edit when the admin cleared grants the key held", () => {
    const clearedOnEdit = { grants: [], servers: catalog, loadedGrants: ["tickets-bots"] };
    expect(authProfileGrantEntry(clearedOnEdit)).toEqual({ mcp_auth_profiles: [] });
  });

  it("leaves the key out when a non-admin's catalog hides the profile a key holds", () => {
    const hiddenProfiles = catalog.map((server) => ({ ...server, additional_auth: null }));
    const nonAdminEdit = { grants: [], servers: hiddenProfiles, loadedGrants: ["tickets-bots"] };
    expect(authProfileGrantEntry(nonAdminEdit)).toEqual({});
  });

  it("leaves the key out when an edit keeps the grants the key held", () => {
    const unchangedEdit = { grants: ["TICKETS-Bots"], servers: catalog, loadedGrants: ["tickets-bots"] };
    expect(authProfileGrantEntry(unchangedEdit)).toEqual({});
  });

  it("leaves the key out when the server catalog is unavailable, so stored grants stay", () => {
    const catalogUnavailable = { grants: ["tickets-bots"], servers: undefined, loadedGrants: ["tickets-bots"] };
    expect(authProfileGrantEntry(catalogUnavailable)).toEqual({});
  });
});

describe("grantChangeNeedsServers", () => {
  it("flags a grant change only when the server catalog is missing", () => {
    expect(grantChangeNeedsServers({ grants: [], servers: undefined, loadedGrants: ["tickets-bots"] })).toBe(true);
    expect(grantChangeNeedsServers({ grants: [], servers: catalog, loadedGrants: ["tickets-bots"] })).toBe(false);
  });

  it("lets a save through when the grants are unchanged or the form never carried them", () => {
    expect(
      grantChangeNeedsServers({ grants: ["TICKETS-Bots"], servers: undefined, loadedGrants: ["tickets-bots"] }),
    ).toBe(false);
    expect(grantChangeNeedsServers({ grants: undefined, servers: undefined, loadedGrants: ["tickets-bots"] })).toBe(
      false,
    );
  });
});
