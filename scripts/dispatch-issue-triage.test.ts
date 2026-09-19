import { describe, expect, test } from "bun:test";

import type { Comment, GitHubApi } from "./auto-close-duplicates";
import {
  DISPATCH_COMMENT,
  DISPATCH_MARKER,
  RUNNING_LABEL,
  dispatchIssue,
  dispatchTarget,
  openPullsReferencing,
  type DispatchVerdict,
  type IssueForDispatch,
  type TimelineEvent,
} from "./dispatch-issue-triage";
import type { FlagConfig } from "./flag-duplicate-issue";

const ready = (labels: readonly string[] = ["kind:bug", "dup:clear", "lift:small", "domain:caching"], overrides: Partial<IssueForDispatch> = {}): IssueForDispatch => ({
  number: 41900,
  state: "open",
  labels: labels.map((name) => ({ name })),
  ...overrides,
});

const crossRef = (number: number, state: string, pull = true): TimelineEvent => ({
  event: "cross-referenced",
  source: { issue: { number, state, ...(pull ? { pull_request: {} } : {}) } },
});

const marker: Comment = {
  id: 1,
  body: DISPATCH_COMMENT,
  created_at: "2026-09-18T00:00:00Z",
  user: { type: "Bot", login: "github-actions[bot]" },
};

const config: FlagConfig = { repo: "BerriAI/litellm", issueNumber: 41900, dryRun: false };

const reasonOf = (verdict: DispatchVerdict): string => (verdict.kind === "skip" ? verdict.reason : "");

describe("dispatchTarget", () => {
  test("an open bug cleared of duplicates with no needs, no repro state, no marker and no PR is dispatched", () => {
    expect(dispatchTarget(ready(), [], [])).toEqual({ kind: "dispatch" });
  });

  test("both positive signals are required, label absence never counts", () => {
    expect(reasonOf(dispatchTarget(ready(["dup:clear"]), [], []))).toBe("missing kind:bug");
    expect(reasonOf(dispatchTarget(ready(["kind:bug"]), [], []))).toBe("missing dup:clear");
    expect(reasonOf(dispatchTarget(ready(["kind:feature", "dup:clear"]), [], []))).toBe("missing kind:bug");
    expect(reasonOf(dispatchTarget(ready([]), [], []))).toBe("missing kind:bug and dup:clear");
  });

  test("closed issues and pull requests are left alone", () => {
    expect(reasonOf(dispatchTarget(ready(undefined, { state: "closed" }), [], []))).toBe("is closed");
    expect(reasonOf(dispatchTarget(ready(undefined, { pull_request: {} }), [], []))).toBe("is a pull request");
  });

  test("any needs:*, any repro:* or a duplicate marker blocks, and every blocker is named", () => {
    const base = ["kind:bug", "dup:clear"];
    expect(reasonOf(dispatchTarget(ready([...base, "needs:repro"]), [], []))).toBe("carries needs:repro");
    expect(reasonOf(dispatchTarget(ready([...base, "needs:version"]), [], []))).toBe("carries needs:version");
    expect(reasonOf(dispatchTarget(ready([...base, "repro:running"]), [], []))).toBe("carries repro:running");
    expect(reasonOf(dispatchTarget(ready([...base, "repro:skip"]), [], []))).toBe("carries repro:skip");
    expect(reasonOf(dispatchTarget(ready([...base, "repro:fail"]), [], []))).toBe("carries repro:fail");
    expect(reasonOf(dispatchTarget(ready([...base, "potential-duplicate"]), [], []))).toBe("carries potential-duplicate");
    expect(reasonOf(dispatchTarget(ready([...base, "duplicate"]), [], []))).toBe("carries duplicate");
    expect(reasonOf(dispatchTarget(ready([...base, "needs:repro", "repro:skip"]), [], []))).toBe("carries needs:repro, repro:skip");
  });

  test("a marker comment left by an earlier dispatch blocks even after the labels were cleaned up", () => {
    expect(reasonOf(dispatchTarget(ready(), [marker], []))).toBe("was already dispatched once");
    const unrelated: Comment = { ...marker, body: "I hit this too" };
    expect(dispatchTarget(ready(), [unrelated], [])).toEqual({ kind: "dispatch" });
  });

  test("only the workflow's own marker counts, a commenter pasting it cannot opt the issue out", () => {
    const forged: Comment = { ...marker, user: { type: "User", login: "someone" } };
    expect(dispatchTarget(ready(), [forged], [])).toEqual({ kind: "dispatch" });
    const otherBot: Comment = { ...marker, user: { type: "Bot", login: "some-other-app[bot]" } };
    expect(dispatchTarget(ready(), [otherBot], [])).toEqual({ kind: "dispatch" });
  });

  test("an open pull request that references the issue blocks, a merged one or a plain issue does not", () => {
    expect(reasonOf(dispatchTarget(ready(), [], [crossRef(41950, "open")]))).toBe("open pull request #41950 already references it");
    expect(reasonOf(dispatchTarget(ready(), [], [crossRef(41950, "open"), crossRef(41960, "open")]))).toBe(
      "open pull requests #41950, #41960 already reference it",
    );
    expect(dispatchTarget(ready(), [], [crossRef(41950, "closed")])).toEqual({ kind: "dispatch" });
    expect(dispatchTarget(ready(), [], [crossRef(41950, "open", false)])).toEqual({ kind: "dispatch" });
  });
});

describe("openPullsReferencing", () => {
  test("keeps open pull requests once each, sorted, and ignores other timeline events", () => {
    const timeline: readonly TimelineEvent[] = [
      { event: "labeled" },
      crossRef(300, "open"),
      crossRef(200, "open"),
      crossRef(300, "open"),
      crossRef(100, "closed"),
      crossRef(50, "open", false),
      { event: "cross-referenced" },
    ];
    expect(openPullsReferencing(timeline)).toEqual([200, 300]);
  });
});

describe("dispatchIssue", () => {
  function fakeApi(
    issue: IssueForDispatch = ready(),
    comments: readonly Comment[] = [],
    timeline: readonly TimelineEvent[] = [],
  ): { readonly api: GitHubApi; readonly writes: string[] } {
    const writes: string[] = [];
    const api: GitHubApi = {
      request: async <T>(method: string, path: string, body?: object): Promise<T> => {
        if (method !== "GET") {
          writes.push(`${method} ${path} ${JSON.stringify(body)}`);
          return {} as T;
        }
        if (path === "/repos/BerriAI/litellm/issues/41900") {
          return issue as T;
        }
        if (path.startsWith("/repos/BerriAI/litellm/issues/41900/comments")) {
          return comments as T;
        }
        if (path.startsWith("/repos/BerriAI/litellm/issues/41900/timeline")) {
          return timeline as T;
        }
        throw new Error(`unexpected GET ${path}`);
      },
    };
    return { api, writes };
  }

  test("a real run adds the lock label first, then the marker comment", async () => {
    const { api, writes } = fakeApi();
    expect(await dispatchIssue(api, config)).toEqual({ kind: "dispatch" });
    expect(writes).toEqual([
      `POST /repos/BerriAI/litellm/issues/41900/labels {"labels":["${RUNNING_LABEL}"]}`,
      `POST /repos/BerriAI/litellm/issues/41900/comments ${JSON.stringify({ body: DISPATCH_COMMENT })}`,
    ]);
    expect(DISPATCH_COMMENT.startsWith(DISPATCH_MARKER)).toBe(true);
  });

  test("a dry run decides the same way and writes nothing", async () => {
    const { api, writes } = fakeApi();
    expect(await dispatchIssue(api, { ...config, dryRun: true })).toEqual({ kind: "dispatch" });
    expect(writes).toEqual([]);
  });

  test("a blocked issue is never written to, whatever blocked it", async () => {
    const labelled = fakeApi(ready(["kind:bug", "dup:clear", "repro:running"]));
    expect((await dispatchIssue(labelled.api, config)).kind).toBe("skip");
    expect(labelled.writes).toEqual([]);

    const marked = fakeApi(ready(), [marker]);
    expect((await dispatchIssue(marked.api, config)).kind).toBe("skip");
    expect(marked.writes).toEqual([]);

    const referenced = fakeApi(ready(), [], [crossRef(41950, "open")]);
    expect((await dispatchIssue(referenced.api, config)).kind).toBe("skip");
    expect(referenced.writes).toEqual([]);
  });

  test("the second run after a successful dispatch is a no-op because the label it just added blocks it", async () => {
    const first = fakeApi();
    await dispatchIssue(first.api, config);
    const second = fakeApi(ready(["kind:bug", "dup:clear", RUNNING_LABEL]), [marker]);
    expect(reasonOf(await dispatchIssue(second.api, config))).toBe(`carries ${RUNNING_LABEL}`);
    expect(second.writes).toEqual([]);
  });

  test("the reporter-facing comment stays within the 15 to 25 word limit", () => {
    const words = DISPATCH_COMMENT.split("\n").slice(1).join(" ").trim().split(/\s+/);
    expect(words.length).toBeGreaterThanOrEqual(15);
    expect(words.length).toBeLessThanOrEqual(25);
  });
});
