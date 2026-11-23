import { describe, expect, it } from "vitest";
import { eventServers, sameEventPermissions, withServerEvents } from "./eventPermissions";

describe("withServerEvents", () => {
  it("restricts one server without touching the others", () => {
    expect(withServerEvents({ a: ["x"] }, "b", ["y", "z"])).toEqual({ a: ["x"], b: ["y", "z"] });
  });

  it("drops the entry when nothing is selected, which means every event type is allowed", () => {
    expect(withServerEvents({ a: ["x"], b: ["y"] }, "b", [])).toEqual({ a: ["x"] });
  });

  it("replaces an existing restriction", () => {
    expect(withServerEvents({ a: ["x"] }, "a", ["q"])).toEqual({ a: ["q"] });
  });
});

describe("sameEventPermissions", () => {
  it("ignores key and event order", () => {
    expect(sameEventPermissions({ a: ["x", "y"], b: ["z"] }, { b: ["z"], a: ["y", "x"] })).toBe(true);
  });

  it("notices a changed restriction", () => {
    expect(sameEventPermissions({ a: ["x"] }, { a: ["x", "y"] })).toBe(false);
    expect(sameEventPermissions({ a: ["x"] }, {})).toBe(false);
  });
});

describe("eventServers", () => {
  it("lists only events-enabled servers, labelled by name, then alias, then id", () => {
    expect(
      eventServers([
        { server_id: "a", server_name: "CI", events: { enabled: true } },
        { server_id: "b", alias: "deploys", events: { enabled: true } },
        { server_id: "c", events: { enabled: true } },
        { server_id: "d", server_name: "off", events: { enabled: false } },
        { server_id: "e", server_name: "never" },
      ]),
    ).toEqual([
      { serverId: "a", label: "CI" },
      { serverId: "b", label: "deploys" },
      { serverId: "c", label: "c" },
    ]);
  });
});
