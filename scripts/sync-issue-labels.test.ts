import { describe, expect, test } from "bun:test";

import type { GitHubApi } from "./auto-close-duplicates";
import { MANIFEST, manifestLabels, type Manifest } from "./issue-labels";
import { readConfig, syncLabels, syncPlan, type GitHubLabel } from "./sync-issue-labels";

const small: Manifest = {
  domain: { caching: { color: "1C6E5B", description: "Response cache" } },
  provider: {},
  kind: {},
  priority: { p0: { color: "B60205", description: "Bleeding" } },
  lift: {},
  needs: { template: { color: "E99695", description: "Template sections missing" } },
  dup: {},
  repro: { skip: { color: "C5DEF5", description: "Opted out" } },
};

describe("syncPlan", () => {
  test("creates what is missing, updates what drifted, leaves the rest", () => {
    const existing: readonly GitHubLabel[] = [
      { name: "Domain:Caching", color: "1c6e5b", description: "Response cache" },
      { name: "priority:p0", color: "000000", description: "Bleeding" },
      { name: "bug", color: "d73a4a", description: "Something isn't working" },
    ];
    expect(syncPlan(existing, small).map((action) => `${action.kind} ${action.name}`)).toEqual([
      "unchanged domain:caching",
      "update priority:p0",
      "create needs:template",
      "create repro:skip",
    ]);
  });

  test("a missing description counts as drift", () => {
    const existing: readonly GitHubLabel[] = [{ name: "domain:caching", color: "1C6E5B", description: null }];
    expect(syncPlan(existing, small)[0]?.kind).toBe("update");
  });

  test("the real manifest is 50 labels across eight namespaces", () => {
    expect(manifestLabels(MANIFEST)).toHaveLength(50);
    expect(syncPlan([], MANIFEST).every((action) => action.kind === "create")).toBe(true);
  });
});

describe("syncLabels", () => {
  function fakeApi(existing: readonly GitHubLabel[]): { readonly api: GitHubApi; readonly writes: string[] } {
    const writes: string[] = [];
    const api: GitHubApi = {
      request: async <T>(method: string, path: string, body?: object): Promise<T> => {
        if (method === "GET" && path.startsWith("/repos/BerriAI/litellm/labels")) {
          return existing as T;
        }
        if (method === "GET") {
          throw new Error(`unexpected GET ${path}`);
        }
        writes.push(`${method} ${path} ${JSON.stringify(body)}`);
        return {} as T;
      },
    };
    return { api, writes };
  }

  test("a real run creates and patches, and never deletes", async () => {
    const { api, writes } = fakeApi([{ name: "priority:p0", color: "000000", description: "Bleeding" }, { name: "stale", color: "ededed", description: null }]);
    await syncLabels(api, { repo: "BerriAI/litellm", dryRun: false }, small);
    expect(writes).toEqual([
      'POST /repos/BerriAI/litellm/labels {"name":"domain:caching","color":"1C6E5B","description":"Response cache"}',
      'PATCH /repos/BerriAI/litellm/labels/priority%3Ap0 {"color":"B60205","description":"Bleeding"}',
      'POST /repos/BerriAI/litellm/labels {"name":"needs:template","color":"E99695","description":"Template sections missing"}',
      'POST /repos/BerriAI/litellm/labels {"name":"repro:skip","color":"C5DEF5","description":"Opted out"}',
    ]);
  });

  test("a dry run returns the plan and writes nothing", async () => {
    const { api, writes } = fakeApi([]);
    const plan = await syncLabels(api, { repo: "BerriAI/litellm", dryRun: true }, small);
    expect(plan.map((action) => action.kind)).toEqual(["create", "create", "create", "create"]);
    expect(writes).toEqual([]);
  });
});

describe("readConfig", () => {
  test("reads the repo and the dry-run flag", () => {
    expect(readConfig({ GITHUB_TOKEN: "t", GITHUB_REPOSITORY: "BerriAI/litellm", DRY_RUN: "true" })).toEqual({
      token: "t",
      repo: "BerriAI/litellm",
      dryRun: true,
    });
    expect(() => readConfig({ GITHUB_TOKEN: "t", GITHUB_REPOSITORY: "nope" })).toThrow("GITHUB_REPOSITORY");
  });
});
