import moment from "moment";
import { createLoader, createSerializer } from "nuqs";
import { describe, expect, it } from "vitest";

import { UsageClient } from "./components/overview/agentCatalog";
import type { UsageOption } from "./components/UsageViewSelect/UsageViewSelect";
import { USAGE_OPTIONS } from "./components/UsageViewSelect/UsageViewSelect";
import {
  cleanUsageUrl,
  clientDateRangeFromParams,
  clientDateRangePatch,
  clientPanelPatch,
  dateRangeFromParams,
  dateRangePatch,
  entitySelectionPatch,
  selectedEntitiesFromParams,
  usageHrefForUser,
  usageTabFromParams,
  usageTabPatch,
  usageUrlParsers,
  usageViewFromParams,
  usageViewPatch,
  type UsageUrlParams,
  type UsageUrlPatch,
} from "./usageUrlState";

const load = createLoader(usageUrlParsers);
const serialize = createSerializer(usageUrlParsers);

const applyPatch = (query: string, patch: UsageUrlPatch): string => serialize(query, patch);

const withTimezone = <T>(timezone: string, callback: () => T): T => {
  const originalTimezone = process.env.TZ;
  process.env.TZ = timezone;
  try {
    return callback();
  } finally {
    if (originalTimezone === undefined) {
      delete process.env.TZ;
    } else {
      process.env.TZ = originalTimezone;
    }
  }
};

const adminAccess = { allowedViews: USAGE_OPTIONS, isAdmin: true };
const internalUserAccess = {
  allowedViews: ["global", "tag"] as readonly UsageOption[],
  isAdmin: false,
};

const clean = (query: string, access = adminAccess): { query: string; deniedAccess: boolean } | null => {
  const cleanup = cleanUsageUrl(load(query), access);
  return cleanup && { query: applyPatch(query, cleanup.patch), deniedAccess: cleanup.deniedAccess };
};

describe("usage view and tab", () => {
  it("reads a known view and tab from the URL", () => {
    const params = load("?view=team&tab=models");

    expect(usageViewFromParams(params)).toBe("team");
    expect(usageTabFromParams(params)).toBe("models");
  });

  it("falls back to the global view and cost tab when the URL names neither", () => {
    const params = load("?view=bogus&tab=bogus");

    expect(usageViewFromParams(params)).toBe("global");
    expect(usageTabFromParams(params)).toBe("cost");
  });

  it("switching views drops the previous view's filters and tab but keeps the date range", () => {
    expect(applyPatch("?user=u1&tab=keys&range=30d", usageViewPatch("team"))).toBe("?range=30d&view=team");
    expect(applyPatch("?view=team&team=t1", usageViewPatch("global"))).toBe("");
  });

  it("round-trips the User Agent Activity view", () => {
    const query = applyPatch("", usageViewPatch("user-agent-activity"));

    expect(query).toBe("?view=user-agent-activity");
    expect(usageViewFromParams(load(query))).toBe("user-agent-activity");
  });

  it.each(["users", "errors"] as const)("round-trips the %s tab", (tab) => {
    const query = applyPatch("", usageTabPatch(tab));

    expect(query).toBe(`?tab=${tab}`);
    expect(usageTabFromParams(load(query))).toBe(tab);
  });

  it("falls back from the admin-only Errors tab for an internal user", () => {
    expect(usageTabFromParams(load("?tab=errors"), false)).toBe("cost");
    expect(clean("?tab=errors", internalUserAccess)).toEqual({ query: "", deniedAccess: false });
  });

  it("keeps the default tab out of the URL", () => {
    expect(applyPatch("?tab=models", usageTabPatch("cost"))).toBe("");
    expect(applyPatch("", usageTabPatch("mcp"))).toBe("?tab=mcp");
  });
});

describe("entity selection", () => {
  it("stores teams as repeated keys so ids with commas survive", () => {
    const query = applyPatch("?view=tag", entitySelectionPatch("tag", ["Credential: a,b", "prod"]));

    expect(query).toBe("?view=tag&tag=Credential:+a,b&tag=prod");
    expect(selectedEntitiesFromParams(load(query), "tag")).toEqual(["Credential: a,b", "prod"]);
  });

  it("maps the organization view to the org key", () => {
    const query = applyPatch("?view=organization", entitySelectionPatch("organization", ["o1"]));

    expect(query).toBe("?view=organization&org=o1");
    expect(selectedEntitiesFromParams(load(query), "organization")).toEqual(["o1"]);
  });

  it("stores the user view's selection in the single user key", () => {
    const query = applyPatch("?view=user", entitySelectionPatch("user", ["u1"]));

    expect(query).toBe("?view=user&user=u1");
    expect(selectedEntitiesFromParams(load(query), "user")).toEqual(["u1"]);
  });

  it("keeps User Agent Activity tag filters in the URL", () => {
    const query = applyPatch("?view=user-agent-activity", entitySelectionPatch("tag", ["User-Agent: Chrome/1.0"]));

    expect(query).toBe("?view=user-agent-activity&tag=User-Agent:+Chrome/1.0");
    expect(clean(query)).toBeNull();
    expect(selectedEntitiesFromParams(load(query), "tag")).toEqual(["User-Agent: Chrome/1.0"]);
  });

  it("removes the key once the selection is cleared", () => {
    expect(applyPatch("?view=team&team=t1", entitySelectionPatch("team", []))).toBe("?view=team");
    expect(applyPatch("?view=user&user=u1", entitySelectionPatch("user", []))).toBe("?view=user");
  });
});

describe("date range", () => {
  it("stores a preset as a rolling range key", () => {
    const lastThirtyDays = { from: moment().subtract(30, "days").startOf("day").toDate(), to: new Date() };

    expect(applyPatch("?from=a&to=b", dateRangePatch(lastThirtyDays, "30d"))).toBe("?range=30d");
  });

  it("stores a custom range matching today as explicit dates", () => {
    const today = moment();
    const value = { from: today.clone().startOf("day").toDate(), to: today.clone().endOf("day").toDate() };

    expect(dateRangePatch(value, null)).toEqual({
      range: null,
      from: today.format("YYYY-MM-DD"),
      to: today.format("YYYY-MM-DD"),
    });
  });

  it("stores an unknown preset label as explicit dates", () => {
    const today = moment();
    const value = { from: today.clone().startOf("day").toDate(), to: today.clone().endOf("day").toDate() };

    expect(dateRangePatch(value, "not-a-preset")).toEqual({
      range: null,
      from: today.format("YYYY-MM-DD"),
      to: today.format("YYYY-MM-DD"),
    });
  });

  it("keeps the custom calendar day when opened in a different timezone", () => {
    const query = withTimezone("Australia/Sydney", () => {
      const value = {
        from: new Date(2026, 0, 5, 0, 0, 0, 0),
        to: new Date(2026, 0, 5, 23, 59, 59, 999),
      };

      expect(value.from.getTimezoneOffset()).toBe(-660);
      expect(moment(value.from).format("YYYY-MM-DD")).toBe("2026-01-05");
      expect(moment(value.to).format("YYYY-MM-DD")).toBe("2026-01-05");

      const serialized = applyPatch("", dateRangePatch(value, null));
      expect(serialized).toBe("?from=2026-01-05&to=2026-01-05");
      return serialized;
    });

    withTimezone("America/New_York", () => {
      expect(new Date(2026, 0, 5).getTimezoneOffset()).toBe(300);
      const range = dateRangeFromParams(load(query));
      expect(range).toBeDefined();
      if (!range) return;

      expect(moment(range.from).format("YYYY-MM-DD")).toBe("2026-01-05");
      expect(moment(range.to).format("YYYY-MM-DD")).toBe("2026-01-05");
    });
  });

  it("re-resolves a rolling range against the current day", () => {
    const range = dateRangeFromParams(load("?range=7d"));

    expect(moment(range?.from).isSame(moment().subtract(7, "days"), "day")).toBe(true);
    expect(moment(range?.to).isSame(moment(), "day")).toBe(true);
  });

  it("round-trips a custom range as whole local calendar days", () => {
    const custom = { from: new Date(2026, 0, 5, 8, 0, 0, 0), to: new Date(2026, 0, 9, 20, 30, 0, 0) };
    const query = applyPatch("?range=7d", dateRangePatch(custom, null));

    expect(query).toBe("?from=2026-01-05&to=2026-01-09");
    expect(dateRangeFromParams(load(query))).toEqual({
      from: new Date(2026, 0, 5, 0, 0, 0, 0),
      to: new Date(2026, 0, 9, 23, 59, 59, 999),
    });
  });

  it.each([
    ["an unknown preset", "?range=90d"],
    ["an ISO timestamp", "?from=2026-01-04T13:00:00.000Z&to=2026-01-05"],
    ["an impossible date", "?from=2026-02-30&to=2026-03-01"],
    ["a malformed date", "?from=yesterday&to=2026-01-09"],
    ["a range that ends before it starts", "?from=2026-01-09&to=2026-01-05"],
    ["only one end of a custom range", "?from=2026-01-05"],
  ])("ignores %s", (_label, query) => {
    expect(dateRangeFromParams(load(query))).toBeUndefined();
  });
});

describe("cleanUsageUrl", () => {
  it("leaves a valid URL alone", () => {
    expect(clean("?view=team&team=t1&team=t2&range=30d")).toBeNull();
    expect(clean("?user=u1&tab=keys&from=2026-01-05&to=2026-01-09")).toBeNull();
    expect(clean("?view=tag&tag=prod", internalUserAccess)).toBeNull();
  });

  it("silently drops params that name nothing real", () => {
    expect(clean("?view=bogus&tab=bogus&range=90d")).toEqual({ query: "", deniedAccess: false });
    expect(clean("?from=yesterday&to=today&team=t1")).toEqual({ query: "", deniedAccess: false });
  });

  it("drops a tab on a view that has no tabs", () => {
    expect(clean("?view=team&tab=models")).toEqual({ query: "?view=team", deniedAccess: false });
  });

  it("drops filters that belong to a different view", () => {
    expect(clean("?view=team&team=t1&org=o1&user=u1")).toEqual({ query: "?view=team&team=t1", deniedAccess: false });
    expect(clean("?view=my-usage&user=u1")).toEqual({ query: "?view=my-usage", deniedAccess: false });
  });

  it("prefers the preset when a URL carries both a preset and a custom range", () => {
    expect(clean("?range=7d&from=2026-01-05&to=2026-01-09")).toEqual({
      query: "?range=7d",
      deniedAccess: false,
    });
  });

  it("sends a viewer to the default view, keeping the dates, when the linked view is not theirs", () => {
    expect(clean("?view=team&team=t1&range=30d", internalUserAccess)).toEqual({
      query: "?range=30d",
      deniedAccess: true,
    });
  });

  it("denies another user's usage to a non-admin", () => {
    expect(clean("?user=someone-else&tab=models", internalUserAccess)).toEqual({ query: "", deniedAccess: true });
  });
});

describe("cross-page links", () => {
  it("encodes the user id so it survives the round trip into the usage URL", () => {
    const href = usageHrefForUser("alice+ops@example.com");
    const params: UsageUrlParams = load(href.slice(href.indexOf("?")));

    expect(params.user).toBe("alice+ops@example.com");
  });
});

describe("client usage filters", () => {
  it("should preserve a recognized client and person in a share link", () => {
    const params = load("?user=u1&client=claude-desktop&range=7d");
    expect(params.client).toBe("claude-desktop");
    expect(clean("?user=u1&client=claude-desktop&range=7d")).toBeNull();
    expect(load(serialize(params)).user).toBe("u1");
  });

  it("should reject an unknown client", () => {
    expect(load("?client=made-up").client).toBeNull();
  });

  it("should clear the client when changing to another usage view", () => {
    expect(load(applyPatch("?client=claude-desktop&user=u1", usageViewPatch("team"))).client).toBeNull();
    expect(clean("?view=organization&client=claude-desktop")).toEqual({
      query: "?view=organization",
      deniedAccess: false,
    });
    expect(clean("?view=team&client=claude-desktop")).toBeNull();
  });

  it("should preserve the underlying tab and person when a client panel opens", () => {
    const query = applyPatch("?user=u1&tab=keys&range=7d", { client: UsageClient.ClaudeDesktop, client_user: "u2" });
    expect(usageTabFromParams(load(query))).toBe("keys");
    expect(clean(query)).toBeNull();
    expect(load(query).user).toBe("u1");
    expect(load(query).client_user).toBe("u2");
    expect(applyPatch(query, { client: null, client_user: null })).toBe("?user=u1&tab=keys&range=7d");
  });

  it("should remove stray panel person filters", () => {
    expect(clean("?client_user=u2&range=7d")).toEqual({ query: "?range=7d", deniedAccess: false });
    expect(clean("?client=claude-desktop&client_user=u2", internalUserAccess)).toEqual({
      query: "?client=claude-desktop",
      deniedAccess: false,
    });
  });
});

describe("client panel date range", () => {
  it("should change the panel dates without changing overview filters or dates", () => {
    const query = applyPatch("?user=u1&tab=models&range=30d&client=claude-desktop", clientDateRangePatch({}, "7d"));
    const params = load(query);
    expect(params.range).toBe("30d");
    expect(params.user).toBe("u1");
    expect(params.tab).toBe("models");
    expect(params.client_range).toBe("7d");
    expect(clientDateRangeFromParams(params)).toBeDefined();
    expect(clean(query)).toBeNull();
  });

  it("should round-trip custom panel dates independently", () => {
    const value = { from: new Date(2026, 9, 20), to: new Date(2026, 9, 22) };
    const query = applyPatch("?client=claude-desktop&client_range=7d&range=30d", clientDateRangePatch(value, null));
    const params = load(query);
    expect(params.client_range).toBeNull();
    expect(params.client_from).toBe("2026-10-20");
    expect(params.client_to).toBe("2026-10-22");
    expect(moment(clientDateRangeFromParams(params)?.from).format("YYYY-MM-DD")).toBe("2026-10-20");
    expect(params.range).toBe("30d");
  });

  it("should clear only panel state on close or when opening a different client", () => {
    const query = "?client=claude-desktop&client_user=u2&client_range=7d&range=30d&user=u1&tab=keys";
    expect(applyPatch(query, clientPanelPatch(null))).toBe("?range=30d&user=u1&tab=keys");
    const next = load(applyPatch(query, clientPanelPatch(UsageClient.ClaudeCode)));
    expect(next.client).toBe("claude-code");
    expect(next.client_range).toBeNull();
    expect(next.range).toBe("30d");
  });

  it("should remove invalid or orphaned panel dates without clearing overview dates", () => {
    expect(clean("?range=7d&client_range=30d")).toEqual({ query: "?range=7d", deniedAccess: false });
    expect(clean("?range=7d&client=claude-desktop&client_from=bad&client_to=2026-10-22")).toEqual({
      query: "?range=7d&client=claude-desktop",
      deniedAccess: false,
    });
  });
});

describe("client team scope", () => {
  it("should preserve an administered-team member filter and independent dates", () => {
    const query =
      "?view=team&team=engineering&client=claude-desktop&client_team=engineering&client_user=member&client_range=30d";
    const access = { allowedViews: USAGE_OPTIONS, isAdmin: false, canViewTeamClients: true };
    expect(cleanUsageUrl(load(query), access)).toBeNull();
  });

  it("should discard an unscoped person filter for an internal user", () => {
    const cleanup = cleanUsageUrl(load("?client=claude-desktop&client_user=other"), internalUserAccess);
    expect(cleanup?.patch).toMatchObject({ client_user: null });
  });

  it("should clear panel team and member without changing the overview's selected teams", () => {
    const query = applyPatch(
      "?view=team&team=engineering&client=claude-desktop&client_team=design&client_user=member",
      clientPanelPatch(null),
    );
    expect(query).toBe("?view=team&team=engineering");
  });
});
