import { describe, expect, test } from "bun:test";

import { candidateNumbers, duplicateTarget, type Comment, type GitHubApi, type Issue } from "./auto-close-duplicates";
import {
  MIN_CONFIDENCE,
  flagIssue,
  flagTarget,
  noticeBody,
  parseVerdict,
  readConfig,
  type FlagConfig,
  type Verdict,
} from "./flag-duplicate-issue";

const issue = (number: number, title: string, overrides: Partial<Issue> = {}): Issue => ({
  number,
  title,
  state: "open",
  user: { login: "reporter" },
  ...overrides,
});

const verdict = (overrides: Partial<Verdict> = {}): Verdict => ({
  duplicate_of: 10,
  confidence: 0.99,
  evidence: "Both report the same traceback from the same function.",
  ...overrides,
});

const config: FlagConfig = { repo: "BerriAI/litellm", issueNumber: 35, dryRun: false };

describe("parseVerdict", () => {
  test("accepts the schema's shape, with a null duplicate_of", () => {
    const parsed = parseVerdict('{"duplicate_of": null, "confidence": 0.9, "evidence": "Nothing matches.", "considered": [1]}');
    expect(parsed).toEqual({ kind: "verdict", verdict: { duplicate_of: null, confidence: 0.9, evidence: "Nothing matches." } });
  });

  test("rejects non-JSON, a non-object, a non-integer target, a missing confidence and empty evidence", () => {
    expect(parseVerdict("not json").kind).toBe("skip");
    expect(parseVerdict('"just a string"').kind).toBe("skip");
    expect(parseVerdict('{"duplicate_of": "10", "confidence": 0.99, "evidence": "x"}').kind).toBe("skip");
    expect(parseVerdict('{"duplicate_of": 10.5, "confidence": 0.99, "evidence": "x"}').kind).toBe("skip");
    expect(parseVerdict('{"duplicate_of": 10, "evidence": "x"}').kind).toBe("skip");
    expect(parseVerdict('{"duplicate_of": 10, "confidence": 0.99, "evidence": "  "}').kind).toBe("skip");
  });
});

describe("flagTarget", () => {
  test("flags at the gate and not one hundredth below it", () => {
    expect(flagTarget(verdict({ confidence: MIN_CONFIDENCE }), 35)).toEqual({ kind: "target", original: 10 });
    expect(flagTarget(verdict({ confidence: 0.94 }), 35).kind).toBe("skip");
  });

  test("never flags nothing, itself, or a newer issue", () => {
    expect(flagTarget(verdict({ duplicate_of: null }), 35).kind).toBe("skip");
    expect(flagTarget(verdict({ duplicate_of: 35 }), 35).kind).toBe("skip");
    expect(flagTarget(verdict({ duplicate_of: 36 }), 35).kind).toBe("skip");
  });
});

describe("noticeBody", () => {
  const reporter = issue(35, "[Bug]: Gemma 4-e4b fails on Vertex");

  test("an open original gets the thumbs-up ask, and the marker the sweep reads", () => {
    const body = noticeBody(reporter, issue(10, "Vertex Gemma 4 crash"), "Same stack.");
    expect(body).toContain("**Possible duplicate of #10**");
    expect(body).toContain("add a thumbs-up to #10");
    expect(body).toContain("Same stack.");
    expect(body).not.toContain("closes automatically");
    expect(candidateNumbers(body, 35)).toEqual([10]);
  });

  test("a closed original gets the follow-up-there ask", () => {
    const body = noticeBody(reporter, issue(10, "Vertex Gemma 4 crash", { state: "closed" }), "Same stack.");
    expect(body).toContain("**Already reported in #10**, which is closed");
    expect(body).toContain("follow up there");
  });

  test("warns about the automatic close exactly when the sweep would close", () => {
    const twin = issue(10, "[bug] gemma 4-e4b fails on vertex!");
    const body = noticeBody(reporter, twin, "Same stack.");
    expect(body).toContain("closes automatically in 3 days");
    expect(duplicateTarget(reporter, [twin], []).kind).toBe("close");

    const closedTwin = issue(10, "[bug] gemma 4-e4b fails on vertex!", { state: "closed" });
    expect(noticeBody(reporter, closedTwin, "Same stack.")).not.toContain("closes automatically");
    expect(duplicateTarget(reporter, [closedTwin], []).kind).toBe("skip");
  });
});

describe("flagIssue", () => {
  const reporter = issue(35, "[Bug]: Gemma 4-e4b fails on Vertex");

  function fakeApi(
    prior: Issue = issue(10, "Vertex Gemma 4 crash"),
    comments: readonly Comment[] = [],
    failing: readonly string[] = [],
  ): { readonly api: GitHubApi; readonly writes: string[] } {
    const writes: string[] = [];
    const api: GitHubApi = {
      request: async <T>(method: string, path: string, body?: object): Promise<T> => {
        if (method !== "GET") {
          if (failing.includes(path)) {
            throw new Error(`${method} ${path} failed: 502`);
          }
          writes.push(`${method} ${path} ${JSON.stringify(body)}`);
          return {} as T;
        }
        if (path.startsWith("/repos/BerriAI/litellm/issues/35/comments")) {
          return comments as T;
        }
        if (path === "/repos/BerriAI/litellm/issues/35") {
          return reporter as T;
        }
        if (path === `/repos/BerriAI/litellm/issues/${prior.number}`) {
          return prior as T;
        }
        throw new Error(`unexpected GET ${path}`);
      },
    };
    return { api, writes };
  }

  test("a real run labels first, then comments with the marker", async () => {
    const { api, writes } = fakeApi();
    const result = await flagIssue(api, config, verdict());
    expect(result.kind).toBe("flagged");
    expect(writes.map((write) => write.split(" ").slice(0, 2).join(" "))).toEqual([
      "POST /repos/BerriAI/litellm/issues/35/labels",
      "POST /repos/BerriAI/litellm/issues/35/comments",
    ]);
    expect(writes[0]).toContain('{"labels":["potential-duplicate"]}');
    expect(writes[1]).toContain("<!-- litellm:potential-duplicate candidates=10, -->");
  });

  test("a dry run renders the comment and writes nothing", async () => {
    const { api, writes } = fakeApi();
    const result = await flagIssue(api, { ...config, dryRun: true }, verdict());
    expect(result.kind).toBe("flagged");
    expect(result.kind === "flagged" && result.body).toContain("**Possible duplicate of #10**");
    expect(writes).toEqual([]);
  });

  test("a verdict naming a pull request is dropped without a write", async () => {
    const { api, writes } = fakeApi(issue(10, "fix: Vertex Gemma 4 crash", { pull_request: {} }));
    expect(await flagIssue(api, config, verdict())).toEqual({ kind: "skip", reason: "#10 is a pull request" });
    expect(writes).toEqual([]);
  });

  test("a verdict below the gate never touches the API", async () => {
    const { api, writes } = fakeApi();
    expect((await flagIssue(api, config, verdict({ confidence: 0.9 }))).kind).toBe("skip");
    expect(writes).toEqual([]);
  });

  test("an issue that already carries a notice is not flagged twice", async () => {
    const existing: Comment = {
      id: 1,
      body: "<!-- litellm:potential-duplicate candidates=10, -->\n**Possible duplicate of #10**",
      created_at: "2026-09-10T00:00:00Z",
      user: { type: "Bot", login: "github-actions[bot]" },
    };
    const { api, writes } = fakeApi(undefined, [existing]);
    expect(await flagIssue(api, config, verdict())).toEqual({ kind: "skip", reason: "already carries a duplicate notice" });
    expect(writes).toEqual([]);
  });

  test("a failed comment leaves no marker, so the rerun finishes the job", async () => {
    const commentsPath = "/repos/BerriAI/litellm/issues/35/comments";
    const first = fakeApi(undefined, [], [commentsPath]);
    await expect(flagIssue(first.api, config, verdict())).rejects.toThrow("failed: 502");
    expect(first.writes).toEqual(['POST /repos/BerriAI/litellm/issues/35/labels {"labels":["potential-duplicate"]}']);

    const rerun = fakeApi();
    expect((await flagIssue(rerun.api, config, verdict())).kind).toBe("flagged");
    expect(rerun.writes.map((write) => write.split(" ")[1])).toEqual([
      "/repos/BerriAI/litellm/issues/35/labels",
      commentsPath,
    ]);
  });
});

describe("readConfig", () => {
  const env = { GITHUB_TOKEN: "t", GITHUB_REPOSITORY: "BerriAI/litellm", ISSUE_NUMBER: "35" };

  test("defaults to a real run", () => {
    expect(readConfig(env)).toEqual({ token: "t", repo: "BerriAI/litellm", issueNumber: 35, dryRun: false });
  });

  test("honors DRY_RUN", () => {
    expect(readConfig({ ...env, DRY_RUN: "true" }).dryRun).toBe(true);
  });

  test("refuses a missing token, a malformed repository, or a bad issue number", () => {
    expect(() => readConfig({ ...env, GITHUB_TOKEN: undefined })).toThrow("GITHUB_TOKEN");
    expect(() => readConfig({ ...env, GITHUB_REPOSITORY: "not a repo" })).toThrow("GITHUB_REPOSITORY");
    expect(() => readConfig({ ...env, ISSUE_NUMBER: "" })).toThrow("ISSUE_NUMBER");
    expect(() => readConfig({ ...env, ISSUE_NUMBER: "1.5" })).toThrow("ISSUE_NUMBER");
  });
});
