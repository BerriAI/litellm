import { describe, expect, test } from "bun:test";

import {
  CLOSED_MARKER,
  REOPEN_COMMENT,
  candidateNumbers,
  duplicateTarget,
  normalizeTitle,
  pendingNotice,
  readConfig,
  reopenTarget,
  sweepClosedIssue,
  sweepIssue,
  type Comment,
  type GitHubApi,
  type Issue,
  type Reaction,
  type SweepConfig,
} from "./auto-close-duplicates";

const NOW = new Date("2026-09-04T09:00:00Z");
const DAY_MS = 24 * 60 * 60 * 1000;
const daysAgo = (days: number): string => new Date(NOW.getTime() - days * DAY_MS).toISOString();

const issue = (number: number, title: string, overrides: Partial<Issue> = {}): Issue => ({
  number,
  title,
  state: "open",
  user: { login: "reporter" },
  ...overrides,
});

const notice = (candidates: readonly number[], createdAt: string, overrides: Partial<Comment> = {}): Comment => ({
  id: 900,
  body: `<!-- litellm:potential-duplicate candidates=${candidates.join(",")}, -->\n**Potential duplicate detected**`,
  created_at: createdAt,
  user: { type: "Bot", login: "github-actions[bot]" },
  ...overrides,
});

const humanComment = (createdAt: string, body = "It is not the same thing", login = "reporter"): Comment => ({
  id: 901,
  body,
  created_at: createdAt,
  user: { type: "User", login },
});

const config: SweepConfig = { repo: "BerriAI/litellm", graceDays: 3, dryRun: false, now: NOW };

describe("normalizeTitle", () => {
  test("drops the template prefix, case, and punctuation", () => {
    expect(normalizeTitle("[Bug]: Gemma 4-e4b fails on Vertex!")).toBe("gemma 4 e4b fails on vertex");
    expect(normalizeTitle("[Feature]:   ")).toBe("");
  });
});

describe("candidateNumbers", () => {
  test("reads only the marker field, keeps older issues, sorted ascending and deduplicated", () => {
    const body = "<!-- litellm:potential-duplicate candidates=40,10,30,10, -->\n- #1 - see #1 (100% similar)";
    expect(candidateNumbers(body, 35)).toEqual([10, 30]);
  });

  test("returns nothing without the marker", () => {
    expect(candidateNumbers("- #1 - looks like #1", 35)).toEqual([]);
  });
});

describe("pendingNotice", () => {
  test("waits out the grace period from the latest notice", () => {
    const fresh = pendingNotice(issue(35, "t"), [notice([10], daysAgo(2.9))], config);
    expect(fresh.kind).toBe("skip");
    const aged = pendingNotice(issue(35, "t"), [notice([10], daysAgo(3.1))], config);
    expect(aged.kind).toBe("pending");
    const reposted = pendingNotice(
      issue(35, "t"),
      [notice([10], daysAgo(6)), notice([10], daysAgo(1), { id: 902 })],
      config,
    );
    expect(reposted.kind).toBe("skip");
  });

  test("an objection posted before a re-posted notice still keeps the issue open", () => {
    const verdict = pendingNotice(
      issue(35, "t"),
      [notice([10], daysAgo(10)), humanComment(daysAgo(7)), notice([10], daysAgo(4), { id: 902 })],
      config,
    );
    expect(verdict).toEqual({ kind: "skip", reason: "someone replied after the notice" });
  });

  test("a zero-day grace period acts on the notice at once", () => {
    const verdict = pendingNotice(issue(35, "t"), [notice([10], daysAgo(0.01))], { ...config, graceDays: 0 });
    expect(verdict.kind).toBe("pending");
  });

  test("a human reply after the notice keeps the issue open, a bot reply does not", () => {
    const human = pendingNotice(issue(35, "t"), [notice([10], daysAgo(5)), humanComment(daysAgo(4))], config);
    expect(human).toEqual({ kind: "skip", reason: "someone replied after the notice" });
    const bot = pendingNotice(
      issue(35, "t"),
      [notice([10], daysAgo(5)), { id: 903, body: "triage", created_at: daysAgo(4), user: { type: "Bot", login: "triage[bot]" } }],
      config,
    );
    expect(bot.kind).toBe("pending");
  });

  test("a human quoting the marker is not a notice", () => {
    const quoted = pendingNotice(issue(35, "t"), [notice([10], daysAgo(5), { user: { type: "User", login: "reporter" } })], config);
    expect(quoted).toEqual({ kind: "skip", reason: "carries no duplicate notice" });
  });

  test("never closes an issue twice: a reopened issue is left alone", () => {
    const reopened = pendingNotice(
      issue(35, "t"),
      [notice([10], daysAgo(9)), { id: 904, body: `Closed automatically\n\n${CLOSED_MARKER}`, created_at: daysAgo(5), user: { type: "Bot", login: "github-actions[bot]" } }],
      config,
    );
    expect(reopened).toEqual({ kind: "skip", reason: "was reopened after an automatic close" });
  });

  test("skips pull requests and issues whose only candidates are newer", () => {
    expect(pendingNotice(issue(35, "t", { pull_request: {} }), [notice([10], daysAgo(5))], config).kind).toBe("skip");
    expect(pendingNotice(issue(35, "t"), [notice([40], daysAgo(5))], config)).toEqual({
      kind: "skip",
      reason: "no candidate is older than this issue",
    });
  });
});

describe("duplicateTarget", () => {
  const reporter = issue(35, "[Bug]: Gemma 4-e4b fails on Vertex");

  test("closes only against the earliest open issue with the identical normalized title", () => {
    const verdict = duplicateTarget(
      reporter,
      [issue(10, "[Bug]: Gemma 4-e4n fails on Vertex"), issue(20, "[bug]: gemma 4-e4b fails on vertex"), issue(30, "[Bug]: Gemma 4-e4b fails on Vertex")],
      [],
    );
    expect(verdict).toEqual({ kind: "close", duplicateOf: 20 });
  });

  test("a near miss in the title is not a duplicate", () => {
    const verdict = duplicateTarget(reporter, [issue(10, "[Bug]: Gemma 4-e4n fails on Vertex")], []);
    expect(verdict).toEqual({ kind: "skip", reason: "no older open issue has the identical title" });
  });

  test("bare template titles never match each other", () => {
    const verdict = duplicateTarget(issue(35, "[Bug]: "), [issue(10, "[Bug]: ")], []);
    expect(verdict.kind).toBe("skip");
    expect(verdict.kind === "skip" && verdict.reason).toContain("too short");
  });

  test("a closed candidate or a pull request is never the target", () => {
    expect(duplicateTarget(reporter, [issue(10, reporter.title, { state: "closed" })], []).kind).toBe("skip");
    expect(duplicateTarget(reporter, [issue(10, reporter.title, { pull_request: {} })], []).kind).toBe("skip");
  });

  test("a thumbs down on the notice keeps the issue open", () => {
    const verdict = duplicateTarget(reporter, [issue(10, reporter.title)], [{ content: "+1" }, { content: "-1" }]);
    expect(verdict).toEqual({ kind: "skip", reason: "someone gave the notice a thumbs down" });
  });
});

describe("sweepIssue", () => {
  const reporter = issue(35, "[Bug]: Gemma 4-e4b fails on Vertex");
  const original = issue(10, "[Bug]: Gemma 4-e4b fails on Vertex");

  function fakeApi(
    comments: readonly Comment[] = [notice([10], daysAgo(5))],
    reactionsByNotice: Readonly<Record<number, readonly Reaction[]>> = {},
  ): { readonly api: GitHubApi; readonly writes: readonly string[] } {
    const writes: string[] = [];
    const api: GitHubApi = {
      request: async <T>(method: string, path: string, body?: object): Promise<T> => {
        if (method !== "GET") {
          writes.push(`${method} ${path} ${JSON.stringify(body)}`);
          return {} as T;
        }
        if (path.startsWith("/repos/BerriAI/litellm/issues/35/comments")) {
          return comments as T;
        }
        const reactionsPath = path.match(/^\/repos\/BerriAI\/litellm\/issues\/comments\/(\d+)\/reactions/);
        if (reactionsPath) {
          return (reactionsByNotice[Number(reactionsPath[1])] ?? []) as T;
        }
        if (path === "/repos/BerriAI/litellm/issues/10") {
          return original as T;
        }
        throw new Error(`unexpected GET ${path}`);
      },
    };
    return { api, writes };
  }

  test("a dry run reports the close and writes nothing", async () => {
    const { api, writes } = fakeApi();
    const verdict = await sweepIssue(api, { ...config, dryRun: true }, reporter);
    expect(verdict).toEqual({ kind: "close", duplicateOf: 10 });
    expect(writes).toEqual([]);
  });

  test("a thumbs down on an earlier notice still keeps the issue open", async () => {
    const { api, writes } = fakeApi([notice([10], daysAgo(9)), notice([10], daysAgo(5), { id: 902 })], { 900: [{ content: "-1" }] });
    const verdict = await sweepIssue(api, config, reporter);
    expect(verdict).toEqual({ kind: "skip", reason: "someone gave the notice a thumbs down" });
    expect(writes).toEqual([]);
  });

  test("a real run comments, labels, then closes with the duplicate reason", async () => {
    const { api, writes } = fakeApi();
    const verdict = await sweepIssue(api, config, reporter);
    expect(verdict).toEqual({ kind: "close", duplicateOf: 10 });
    expect(writes.map((write) => write.split(" ").slice(0, 2).join(" "))).toEqual([
      "POST /repos/BerriAI/litellm/issues/35/comments",
      "POST /repos/BerriAI/litellm/issues/35/labels",
      "PATCH /repos/BerriAI/litellm/issues/35",
    ]);
    expect(writes[0]).toContain("duplicate of #10");
    expect(writes[0]).toContain("unanswered for 3 days");
    expect(writes[0]).toContain(CLOSED_MARKER);
    expect(writes[1]).toContain('{"labels":["duplicate"]}');
    expect(writes[2]).toContain('{"state":"closed","state_reason":"duplicate"}');
  });
});

describe("reopenTarget", () => {
  const closedByBot = (overrides: Partial<Issue> = {}): Issue =>
    issue(35, "t", { state: "closed", closed_by: { type: "Bot" }, ...overrides });
  const closeMarker = (createdAt: string): Comment => ({
    id: 905,
    body: `Closed automatically as a duplicate of #10.\n\n${CLOSED_MARKER}`,
    created_at: createdAt,
    user: { type: "Bot", login: "github-actions[bot]" },
  });

  test("a reporter reply after the automatic close reopens", () => {
    const verdict = reopenTarget(closedByBot(), [closeMarker(daysAgo(2)), humanComment(daysAgo(1))]);
    expect(verdict).toEqual({ kind: "reopen" });
  });

  test("an issue closed by a person stays closed", () => {
    const verdict = reopenTarget(closedByBot({ closed_by: { type: "User" } }), [
      closeMarker(daysAgo(2)),
      humanComment(daysAgo(1)),
    ]);
    expect(verdict).toEqual({ kind: "skip", reason: "was closed by a person" });
  });

  test("without the automatic-close marker nothing reopens", () => {
    const verdict = reopenTarget(closedByBot(), [humanComment(daysAgo(1))]);
    expect(verdict).toEqual({ kind: "skip", reason: "carries no automatic-close marker" });
  });

  test("a maintainer reply alone does not reopen", () => {
    const verdict = reopenTarget(closedByBot(), [
      closeMarker(daysAgo(2)),
      humanComment(daysAgo(1), "Confirmed duplicate", "maintainer"),
    ]);
    expect(verdict).toEqual({ kind: "skip", reason: "the reporter has not replied since the close" });
  });

  test("a reporter comment from before the close does not reopen", () => {
    const verdict = reopenTarget(closedByBot(), [humanComment(daysAgo(3)), closeMarker(daysAgo(2))]);
    expect(verdict).toEqual({ kind: "skip", reason: "the reporter has not replied since the close" });
  });

  test("a pull request never reopens", () => {
    const verdict = reopenTarget(closedByBot({ pull_request: {} }), [closeMarker(daysAgo(2)), humanComment(daysAgo(1))]);
    expect(verdict).toEqual({ kind: "skip", reason: "is a pull request" });
  });
});

describe("sweepClosedIssue", () => {
  function fakeApi(issueBody: Issue, comments: readonly Comment[]): { readonly api: GitHubApi; readonly writes: readonly string[] } {
    const writes: string[] = [];
    const api: GitHubApi = {
      request: async <T>(method: string, path: string, body?: object): Promise<T> => {
        if (method !== "GET") {
          writes.push(`${method} ${path} ${JSON.stringify(body)}`);
          return {} as T;
        }
        if (path.startsWith("/repos/BerriAI/litellm/issues/35/comments")) {
          return comments as T;
        }
        if (path === "/repos/BerriAI/litellm/issues/35") {
          return issueBody as T;
        }
        throw new Error(`unexpected GET ${path}`);
      },
    };
    return { api, writes };
  }

  const closedByBot = issue(35, "t", { state: "closed", closed_by: { type: "Bot" } });
  const closeMarker: Comment = {
    id: 905,
    body: `Closed automatically as a duplicate of #10.\n\n${CLOSED_MARKER}`,
    created_at: daysAgo(2),
    user: { type: "Bot", login: "github-actions[bot]" },
  };

  test("a real run unlabels, reopens, then explains", async () => {
    const { api, writes } = fakeApi(closedByBot, [closeMarker, humanComment(daysAgo(1))]);
    const verdict = await sweepClosedIssue(api, config, 35);
    expect(verdict).toEqual({ kind: "reopen" });
    expect(writes).toEqual([
      "DELETE /repos/BerriAI/litellm/issues/35/labels/duplicate undefined",
      'PATCH /repos/BerriAI/litellm/issues/35 {"state":"open"}',
      `POST /repos/BerriAI/litellm/issues/35/comments {"body":"${REOPEN_COMMENT}"}`,
    ]);
  });

  test("a dry run reports the reopen and writes nothing", async () => {
    const { api, writes } = fakeApi(closedByBot, [closeMarker, humanComment(daysAgo(1))]);
    const verdict = await sweepClosedIssue(api, { ...config, dryRun: true }, 35);
    expect(verdict).toEqual({ kind: "reopen" });
    expect(writes).toEqual([]);
  });
});

describe("readConfig", () => {
  test("defaults to a real run with a 3-day grace period", () => {
    const parsed = readConfig({ GITHUB_TOKEN: "t", GITHUB_REPOSITORY: "BerriAI/litellm" }, NOW);
    expect(parsed).toEqual({ token: "t", repo: "BerriAI/litellm", graceDays: 3, dryRun: false, now: NOW });
  });

  test("honors DRY_RUN and GRACE_PERIOD_DAYS overrides", () => {
    const parsed = readConfig(
      { GITHUB_TOKEN: "t", GITHUB_REPOSITORY: "o/r", DRY_RUN: "true", GRACE_PERIOD_DAYS: "0" },
      NOW,
    );
    expect(parsed.dryRun).toBe(true);
    expect(parsed.graceDays).toBe(0);
  });

  test("an empty GRACE_PERIOD_DAYS, as a schedule run renders it, means the default", () => {
    const parsed = readConfig({ GITHUB_TOKEN: "t", GITHUB_REPOSITORY: "o/r", GRACE_PERIOD_DAYS: "" }, NOW);
    expect(parsed.graceDays).toBe(3);
  });

  test("refuses a missing token, a malformed repository, or a bad grace period", () => {
    expect(() => readConfig({ GITHUB_REPOSITORY: "o/r" }, NOW)).toThrow("GITHUB_TOKEN");
    expect(() => readConfig({ GITHUB_TOKEN: "t", GITHUB_REPOSITORY: "litellm" }, NOW)).toThrow("owner/repo");
    expect(() => readConfig({ GITHUB_TOKEN: "t", GITHUB_REPOSITORY: "o/r", GRACE_PERIOD_DAYS: "-1" }, NOW)).toThrow(
      "GRACE_PERIOD_DAYS",
    );
  });
});
