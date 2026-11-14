import { describe, it, expect } from "vitest";
import {
  deselectTools,
  filterServerTools,
  isToolSelected,
  matchesServerName,
  selectTools,
  selectionState,
  toggleTool,
} from "./toolsetSelection";

const t = (server_id: string, tool_name: string) => ({ server_id, tool_name });

const githubTools = [
  { name: "create_issue", description: "Open a new issue" },
  { name: "list_pulls", description: "List pull requests" },
  { name: "search_code", description: "Search code across repos" },
];

describe("filterServerTools", () => {
  it("returns every tool for an empty or whitespace query", () => {
    expect(filterServerTools("", ["github"], githubTools)).toBe(githubTools);
    expect(filterServerTools("   ", ["github"], githubTools)).toBe(githubTools);
  });

  it("returns every tool when the query matches a server name, case-insensitively", () => {
    expect(filterServerTools("GitH", ["github_prod", "GitHub"], githubTools)).toBe(githubTools);
  });

  it("matches tool names and descriptions when the server name does not match", () => {
    expect(filterServerTools("issue", ["github"], githubTools).map((tool) => tool.name)).toEqual(["create_issue"]);
    expect(filterServerTools("PULL REQ", ["github"], githubTools).map((tool) => tool.name)).toEqual(["list_pulls"]);
  });

  it("returns nothing when neither the server nor any tool matches", () => {
    expect(filterServerTools("slack", ["github"], githubTools)).toEqual([]);
  });
});

describe("matchesServerName", () => {
  it("matches any of the server names, case-insensitively, and never for an empty query", () => {
    expect(matchesServerName("PROD", ["github_prod", "GitHub"])).toBe(true);
    expect(matchesServerName("slack", ["github_prod", "GitHub"])).toBe(false);
    expect(matchesServerName("  ", ["github"])).toBe(false);
  });
});

describe("selectTools", () => {
  it("appends new tools after the existing selection in order", () => {
    expect(selectTools([t("a", "x")], [t("b", "y"), t("b", "z")])).toEqual([t("a", "x"), t("b", "y"), t("b", "z")]);
  });

  it("skips tools already selected and duplicates within the input", () => {
    expect(selectTools([t("a", "x")], [t("a", "x"), t("a", "y"), t("a", "y")])).toEqual([t("a", "x"), t("a", "y")]);
  });

  it("treats the same tool name on different servers as different tools", () => {
    expect(selectTools([t("a", "search")], [t("b", "search")])).toEqual([t("a", "search"), t("b", "search")]);
  });
});

describe("deselectTools", () => {
  it("removes only the given tools and keeps the rest of the selection", () => {
    const selected = [t("a", "x"), t("b", "y"), t("gone", "z")];
    expect(deselectTools(selected, [t("b", "y"), t("b", "not-selected")])).toEqual([t("a", "x"), t("gone", "z")]);
  });
});

describe("toggleTool and isToolSelected", () => {
  it("adds an unselected tool and removes a selected one", () => {
    const added = toggleTool([t("a", "x")], t("a", "y"));
    expect(added).toEqual([t("a", "x"), t("a", "y")]);
    expect(isToolSelected(added, t("a", "y"))).toBe(true);
    expect(toggleTool(added, t("a", "x"))).toEqual([t("a", "y")]);
  });
});

describe("selectionState", () => {
  const tools = [t("a", "x"), t("a", "y")];

  it("is none for an empty tool list", () => {
    expect(selectionState([t("a", "x")], [])).toBe("none");
  });

  it("reports none, some and all against the given tools only", () => {
    expect(selectionState([t("b", "x")], tools)).toBe("none");
    expect(selectionState([t("a", "x"), t("b", "z")], tools)).toBe("some");
    expect(selectionState([t("a", "y"), t("a", "x")], tools)).toBe("all");
  });
});
