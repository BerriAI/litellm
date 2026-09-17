import { describe, expect, test } from "bun:test";

import type { Comment, GitHubApi } from "./auto-close-duplicates";
import type { Classification, GateVerdict } from "./classify-issue";
import {
  TEMPLATE_MARKER,
  desiredLabels,
  labelIssue,
  labelPlan,
  parseVerdict,
  readConfig,
  templateComment,
  type LabelConfig,
} from "./label-issue";

const classified = (overrides: Partial<Classification> = {}): Classification => ({
  gate: "pass",
  domain: "caching",
  provider: null,
  kind: "bug",
  priority: "p0",
  lift: "small",
  route: "chat_completions",
  version: "v1.100.0",
  needs: [],
  reason: "Cache returns another key's response.",
  ...overrides,
});

const gated: GateVerdict = { gate: "template", template: "bug", missing: ["Config", "Steps to Repro"] };

const config: LabelConfig = { repo: "BerriAI/litellm", issueNumber: 41700, dryRun: false };

describe("desiredLabels", () => {
  test("a classification is one label per namespace, provider and needs only when present", () => {
    expect(desiredLabels(classified())).toEqual(["domain:caching", "kind:bug", "priority:p0", "lift:small"]);
    expect(desiredLabels(classified({ provider: "bedrock", needs: ["version", "repro"] }))).toEqual([
      "domain:caching",
      "provider:bedrock",
      "kind:bug",
      "priority:p0",
      "lift:small",
      "needs:version",
      "needs:repro",
    ]);
  });

  test("a gated issue wants needs:template and nothing else", () => {
    expect(desiredLabels(gated)).toEqual(["needs:template"]);
  });
});

describe("labelPlan", () => {
  test("a fresh issue gets every label added and nothing removed", () => {
    expect(labelPlan(["bug"], classified())).toEqual({
      add: ["domain:caching", "kind:bug", "priority:p0", "lift:small"],
      remove: [],
    });
  });

  test("a rerun replaces within each namespace and leaves labels outside them alone", () => {
    const current = ["bug", "potential-duplicate", "domain:routing", "provider:openai", "kind:bug", "priority:p2", "lift:small", "needs:template"];
    expect(labelPlan(current, classified())).toEqual({
      add: ["domain:caching", "priority:p0"],
      remove: ["domain:routing", "provider:openai", "priority:p2", "needs:template"],
    });
  });

  test("the same verdict twice is a no-op", () => {
    const current = ["bug", ...desiredLabels(classified({ provider: "azure" }))];
    expect(labelPlan(current, classified({ provider: "azure" }))).toEqual({ add: [], remove: [] });
  });

  test("a gate failure touches only the needs namespace", () => {
    expect(labelPlan(["bug", "domain:caching", "needs:repro"], gated)).toEqual({
      add: ["needs:template"],
      remove: ["needs:repro"],
    });
    expect(labelPlan(["needs:template"], gated)).toEqual({ add: [], remove: [] });
  });
});

describe("templateComment", () => {
  test("names the missing sections, links the right template, and carries the marker", () => {
    const body = templateComment(gated);
    expect(body.startsWith(`${TEMPLATE_MARKER}\n`)).toBe(true);
    expect(body).toContain("missing **Config**, **Steps to Repro** from the [bug template](https://github.com/BerriAI/litellm/issues/new?template=bug_report.yml)");
    expect(body).toContain("add them and it will be labelled automatically");
    expect(body.split("\n")[1]?.split(" ").length).toBeLessThanOrEqual(30);
  });

  test("a single missing section reads naturally and a feature links the feature template", () => {
    const body = templateComment({ gate: "template", template: "feature", missing: ["User Flow"] });
    expect(body).toContain("missing **User Flow** from the [feature template](https://github.com/BerriAI/litellm/issues/new?template=feature_request.yml)");
    expect(body).toContain("add it and");
  });
});

describe("parseVerdict", () => {
  test("accepts both verdict shapes the classify step writes", () => {
    expect(parseVerdict(JSON.stringify(classified()))).toEqual({ kind: "verdict", verdict: classified() });
    expect(parseVerdict(JSON.stringify(gated))).toEqual({ kind: "verdict", verdict: gated });
  });

  test("refuses a label the manifest does not know, so a typo never creates a label", () => {
    expect(parseVerdict(JSON.stringify(classified({ domain: "cache" })))).toMatchObject({ kind: "invalid" });
    expect(parseVerdict(JSON.stringify(classified({ needs: ["screenshots"] })))).toMatchObject({ kind: "invalid" });
    expect(parseVerdict(JSON.stringify(classified({ provider: "groq" })))).toMatchObject({ kind: "invalid" });
  });

  test("refuses junk", () => {
    expect(parseVerdict("")).toMatchObject({ kind: "invalid" });
    expect(parseVerdict("[]")).toMatchObject({ kind: "invalid" });
    expect(parseVerdict('{"gate":"maybe"}')).toMatchObject({ kind: "invalid" });
    expect(parseVerdict('{"gate":"template","template":"bug","missing":[]}')).toMatchObject({ kind: "invalid" });
    expect(parseVerdict('{"gate":"template","template":"docs","missing":["Config"]}')).toMatchObject({ kind: "invalid" });
  });
});

describe("labelIssue", () => {
  const notice: Comment = {
    id: 77,
    body: templateComment(gated),
    created_at: "2026-09-10T00:00:00Z",
    user: { type: "Bot", login: "github-actions[bot]" },
  };

  function fakeApi(
    labels: readonly string[],
    comments: readonly Comment[] = [],
  ): { readonly api: GitHubApi; readonly writes: string[] } {
    const writes: string[] = [];
    const api: GitHubApi = {
      request: async <T>(method: string, path: string, body?: object): Promise<T> => {
        if (method !== "GET") {
          writes.push(`${method} ${path}${body === undefined ? "" : ` ${JSON.stringify(body)}`}`);
          return undefined as T;
        }
        if (path.startsWith("/repos/BerriAI/litellm/issues/41700/comments")) {
          return comments as T;
        }
        if (path === "/repos/BerriAI/litellm/issues/41700") {
          return { labels: labels.map((name) => ({ name })) } as T;
        }
        throw new Error(`unexpected GET ${path}`);
      },
    };
    return { api, writes };
  }

  test("a classification removes stale namespace labels one by one, then adds the new set in one call", async () => {
    const { api, writes } = fakeApi(["bug", "priority:p2", "needs:template"], [notice]);
    const outcome = await labelIssue(api, config, classified());
    expect(writes).toEqual([
      "DELETE /repos/BerriAI/litellm/issues/41700/labels/priority%3Ap2",
      "DELETE /repos/BerriAI/litellm/issues/41700/labels/needs%3Atemplate",
      'POST /repos/BerriAI/litellm/issues/41700/labels {"labels":["domain:caching","kind:bug","priority:p0","lift:small"]}',
      "DELETE /repos/BerriAI/litellm/issues/comments/77",
    ]);
    expect(outcome).toEqual({ plan: { add: ["domain:caching", "kind:bug", "priority:p0", "lift:small"], remove: ["priority:p2", "needs:template"] }, comment: null, removedNotices: 1 });
  });

  test("a gate failure labels first, then posts one comment with the marker", async () => {
    const { api, writes } = fakeApi(["bug"]);
    const outcome = await labelIssue(api, config, gated);
    expect(writes.map((write) => write.split(" ").slice(0, 2).join(" "))).toEqual([
      "POST /repos/BerriAI/litellm/issues/41700/labels",
      "POST /repos/BerriAI/litellm/issues/41700/comments",
    ]);
    expect(writes[0]).toContain('{"labels":["needs:template"]}');
    expect(writes[1]).toContain(TEMPLATE_MARKER);
    expect(outcome.comment).toContain("**Config**, **Steps to Repro**");
  });

  test("a second gate failure on an issue that already carries the notice writes nothing", async () => {
    const { api, writes } = fakeApi(["bug", "needs:template"], [notice]);
    const outcome = await labelIssue(api, config, gated);
    expect(writes).toEqual([]);
    expect(outcome).toEqual({ plan: { add: [], remove: [] }, comment: null, removedNotices: 0 });
  });

  test("a dry run reports the plan and the comment and touches nothing", async () => {
    const { api, writes } = fakeApi(["bug"]);
    const outcome = await labelIssue(api, { ...config, dryRun: true }, gated);
    expect(writes).toEqual([]);
    expect(outcome.plan.add).toEqual(["needs:template"]);
    expect(outcome.comment).toContain(TEMPLATE_MARKER);
  });

  test("a notice is only removed once the issue passes the gate", async () => {
    const stillGated = fakeApi(["needs:template"], [notice]);
    await labelIssue(stillGated.api, config, gated);
    expect(stillGated.writes).toEqual([]);

    const passed = fakeApi(["needs:template"], [notice]);
    await labelIssue(passed.api, config, classified());
    expect(passed.writes).toContain("DELETE /repos/BerriAI/litellm/issues/comments/77");
  });
});

describe("readConfig", () => {
  const env = { GITHUB_TOKEN: "t", GITHUB_REPOSITORY: "BerriAI/litellm", ISSUE_NUMBER: "41700" };

  test("defaults to a real run and honors DRY_RUN", () => {
    expect(readConfig(env)).toEqual({ token: "t", repo: "BerriAI/litellm", issueNumber: 41700, dryRun: false });
    expect(readConfig({ ...env, DRY_RUN: "true" }).dryRun).toBe(true);
  });

  test("refuses a missing token, a malformed repository, or a bad issue number", () => {
    expect(() => readConfig({ ...env, GITHUB_TOKEN: undefined })).toThrow("GITHUB_TOKEN");
    expect(() => readConfig({ ...env, GITHUB_REPOSITORY: "not a repo" })).toThrow("GITHUB_REPOSITORY");
    expect(() => readConfig({ ...env, ISSUE_NUMBER: "1.5" })).toThrow("ISSUE_NUMBER");
  });
});
