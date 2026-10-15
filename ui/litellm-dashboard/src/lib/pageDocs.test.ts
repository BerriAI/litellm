import { describe, expect, it } from "vitest";
import { menuGroups, routeOf, type MenuItem } from "@/components/leftnav";
import { uiHref } from "@/utils/uiHref";
import { PAGE_DOCS, pageDocsFor } from "./pageDocs";

const sidebarRoutes = (): ReadonlySet<string> => {
  const flatten = (items: readonly MenuItem[]): readonly MenuItem[] =>
    items.flatMap((item) => [item, ...(item.children ?? [])]);
  return new Set(menuGroups.flatMap((group) => flatten(group.items)).map(routeOf));
};

describe("PAGE_DOCS", () => {
  it("only links to pages under https://docs.litellm.ai/docs", () => {
    for (const [route, topics] of PAGE_DOCS) {
      for (const topic of topics) {
        expect(topic.href, `${route}: ${topic.title}`).toMatch(/^https:\/\/docs\.litellm\.ai\/docs\//);
      }
    }
  });

  it("only covers routes that exist in the sidebar", () => {
    const routes = sidebarRoutes();
    for (const route of PAGE_DOCS.keys()) {
      expect(routes.has(route), route).toBe(true);
    }
  });
});

describe("pageDocsFor", () => {
  it("returns the teams topics for the teams route", () => {
    const topics = pageDocsFor(uiHref("teams"));
    expect(topics.map((topic) => topic.title)).toEqual(["Team budgets", "Let teams add models", "Team logging"]);
  });

  it("returns no topics for a route outside the map", () => {
    expect(pageDocsFor(uiHref("transform-request"))).toEqual([]);
  });
});
