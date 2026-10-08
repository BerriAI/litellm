import { describe, expect, it } from "vitest";
import type { MenuGroup } from "@/components/leftnav";
import { flattenNavItems, matchNavItems, paletteScopeForRoute, shortcutLabel, type PaletteNavItem } from "./utils";

describe("paletteScopeForRoute", () => {
  it.each([
    ["api-keys", "keys"],
    ["", "keys"],
    ["teams", "global"],
  ] as const)("uses the expected palette scope for %s", (routeSegment, expected) => {
    expect(paletteScopeForRoute(routeSegment)).toBe(expected);
  });
});

describe("flattenNavItems", () => {
  it("filters admin-only groups and items for non-admin roles", () => {
    const groups: MenuGroup[] = [
      {
        groupLabel: "ADMIN",
        roles: ["Admin"],
        items: [{ key: "users", page: "users", label: "Users" }],
      },
      {
        groupLabel: "TOOLS",
        items: [
          { key: "logs", page: "logs", label: "Logs", roles: ["Admin"] },
          { key: "teams", page: "teams", label: "Teams" },
        ],
      },
    ];

    expect(flattenNavItems(groups, "Internal User").map(({ label }) => label)).toEqual(["Teams"]);
  });
});

describe("matchNavItems", () => {
  const items: PaletteNavItem[] = [
    { key: "section", label: "Teams", section: "Logs", route: "teams" },
    { key: "substring", label: "Audit Logs", section: "Access Control", route: "audit-logs" },
    { key: "prefix", label: "Logging", section: "Observability", route: "logging" },
  ];

  it("ranks label prefixes above label substrings and section matches", () => {
    expect(matchNavItems(items, "log").map(({ key }) => key)).toEqual(["prefix", "substring", "section"]);
  });

  it("returns every item in its original order for an empty query", () => {
    expect(matchNavItems(items, "  ")).toEqual(items);
  });
});

describe("shortcutLabel", () => {
  it("uses the command key on Mac and Ctrl elsewhere", () => {
    expect(shortcutLabel("MacIntel", "Mozilla/5.0")).toBe("⌘K");
    expect(shortcutLabel("Linux x86_64", "Mozilla/5.0 (Windows NT 10.0)")).toBe("Ctrl K");
  });
});
