import { describe, expect, test } from "bun:test";

import type { Comment, GitHubApi } from "./auto-close-duplicates";
import {
  FIXED_MARKER,
  closerOf,
  commentFixedIssue,
  fixedBody,
  nextMinor,
  parseVersion,
  placement,
  readConfig,
  releaseCandidate,
  type ClosedIssue,
  type FixedConfig,
} from "./comment-fixed-issue";

const MERGE_COMMIT = "68c4c82ac977b48b2b81ee8d633d5771307c6162";

const mergedPr = {
  __typename: "PullRequest" as const,
  number: 41767,
  merged: true,
  baseRefName: "main",
  mergeCommit: { oid: MERGE_COMMIT },
};

type Closer = ClosedIssue["timelineItems"]["nodes"][number]["closer"];

const closedBy = (closer: Closer, state: ClosedIssue["state"] = "CLOSED"): ClosedIssue => ({
  state,
  timelineItems: { nodes: [{ closer }] },
});

const pyproject = (version: string): string =>
  `[project]\nname = "litellm"\nversion = "${version}"\n\n[tool.commitizen]\nversion = "${version}"\n`;

const config: FixedConfig = { repo: "BerriAI/litellm", issueNumber: 41750, defaultBranch: "main", dryRun: false };

interface World {
  readonly issue?: ClosedIssue | null;
  readonly comments?: readonly Comment[];
  readonly version?: string;
  // Which existing rc.1 tags contain the merge commit; a tag absent from the map does not exist
  readonly tags?: Readonly<Record<string, boolean>>;
}

function fakeApi(world: World = {}): { readonly api: GitHubApi; readonly writes: string[] } {
  const writes: string[] = [];
  const tags = world.tags ?? {};
  const api: GitHubApi = {
    request: async <T>(method: string, path: string, body?: object): Promise<T> => {
      if (method === "POST" && path === "/graphql") {
        return { data: { repository: { issue: world.issue === undefined ? closedBy(mergedPr) : world.issue } } } as T;
      }
      if (method !== "GET") {
        writes.push(`${method} ${path} ${JSON.stringify(body)}`);
        return {} as T;
      }
      if (path.startsWith("/repos/BerriAI/litellm/issues/41750/comments")) {
        return (world.comments ?? []) as T;
      }
      if (path === `/repos/BerriAI/litellm/contents/pyproject.toml?ref=${MERGE_COMMIT}`) {
        return { content: btoa(pyproject(world.version ?? "1.103.0")).replace(/(.{60})/g, "$1\n") } as T;
      }
      const matching = /^\/repos\/BerriAI\/litellm\/git\/matching-refs\/tags\/(.+)$/.exec(path);
      if (matching !== null) {
        return (matching[1] in tags ? [{ ref: `refs/tags/${matching[1]}` }] : []) as T;
      }
      const compare = /^\/repos\/BerriAI\/litellm\/compare\/(.+)\.\.\.(.+)$/.exec(path);
      if (compare !== null && compare[2] === MERGE_COMMIT) {
        return { status: tags[compare[1]] ? "behind" : "ahead" } as T;
      }
      throw new Error(`unexpected ${method} ${path}`);
    },
  };
  return { api, writes };
}

describe("closerOf", () => {
  test("a pull request merged into the default branch is the fix", () => {
    expect(closerOf(closedBy(mergedPr), "main")).toEqual({ kind: "pull_request", number: 41767, mergeCommit: MERGE_COMMIT });
  });

  test("an issue closed by hand, by a commit, or by an unmerged pull request gets no comment", () => {
    expect(closerOf(closedBy(null), "main")).toEqual({ kind: "skip", reason: "closed by hand, not by a pull request" });
    expect(closerOf(closedBy({ __typename: "Commit", oid: MERGE_COMMIT }), "main").kind).toBe("skip");
    expect(closerOf(closedBy({ ...mergedPr, merged: false }), "main").kind).toBe("skip");
    expect(closerOf(closedBy({ ...mergedPr, mergeCommit: null }), "main").kind).toBe("skip");
  });

  test("a pull request merged into a release branch is not a fix on main", () => {
    const verdict = closerOf(closedBy({ ...mergedPr, baseRefName: "release/1.102.0rc2" }), "main");
    expect(verdict).toEqual({ kind: "skip", reason: "#41767 merged into release/1.102.0rc2, not main" });
  });

  test("an issue reopened after the close event is left alone", () => {
    expect(closerOf(closedBy(mergedPr, "OPEN"), "main")).toEqual({ kind: "skip", reason: "the issue is open again" });
  });
});

describe("version helpers", () => {
  test("parseVersion reads the project version and ignores everything else", () => {
    expect(parseVersion(pyproject("1.103.0"))).toBe("1.103.0");
    expect(parseVersion('[project]\nversion = "1.103.0rc1"\n')).toBeUndefined();
    expect(parseVersion("[project]\nname = 'litellm'\n")).toBeUndefined();
  });

  test("the first rc of a version is the release that carries a fix merged under it", () => {
    expect(releaseCandidate("1.103.0")).toBe("v1.103.0-rc.1");
  });

  test("nextMinor bumps the minor and resets the patch", () => {
    expect(nextMinor("1.103.0")).toBe("1.104.0");
    expect(nextMinor("1.99.4")).toBe("1.100.0");
  });
});

describe("placement", () => {
  test("no rc yet: the fix ships in the rc.1 of the version at the merge commit", async () => {
    const { api } = fakeApi({ version: "1.103.0" });
    expect(await placement(api, "BerriAI/litellm", MERGE_COMMIT)).toEqual({ kind: "release", tag: "v1.103.0-rc.1", shipped: false });
  });

  test("rc.1 already cut with the commit in it: the fix is out", async () => {
    const { api } = fakeApi({ version: "1.102.0", tags: { "v1.102.0-rc.1": true } });
    expect(await placement(api, "BerriAI/litellm", MERGE_COMMIT)).toEqual({ kind: "release", tag: "v1.102.0-rc.1", shipped: true });
  });

  test("rc.1 cut before the merge while main still said that version: the fix waits for the next minor", async () => {
    const { api } = fakeApi({ version: "1.102.0", tags: { "v1.102.0-rc.1": false } });
    expect(await placement(api, "BerriAI/litellm", MERGE_COMMIT)).toEqual({ kind: "release", tag: "v1.103.0-rc.1", shipped: false });
  });

  test("keeps walking minors while each rc.1 exists without the commit, then gives up", async () => {
    const twoTaken = fakeApi({ version: "1.102.0", tags: { "v1.102.0-rc.1": false, "v1.103.0-rc.1": false } });
    expect(await placement(twoTaken.api, "BerriAI/litellm", MERGE_COMMIT)).toEqual({ kind: "release", tag: "v1.104.0-rc.1", shipped: false });

    const allTaken = fakeApi({
      version: "1.102.0",
      tags: { "v1.102.0-rc.1": false, "v1.103.0-rc.1": false, "v1.104.0-rc.1": false, "v1.105.0-rc.1": false },
    });
    expect((await placement(allTaken.api, "BerriAI/litellm", MERGE_COMMIT)).kind).toBe("skip");
  });

  test("a pyproject without a version line is a skip, not a comment", async () => {
    const { api } = fakeApi({ version: "not-a-version" });
    expect((await placement(api, "BerriAI/litellm", MERGE_COMMIT)).kind).toBe("skip");
  });
});

describe("fixedBody", () => {
  test("names the pull request and the first release, and carries the marker the rerun looks for", () => {
    const body = fixedBody(41767, { tag: "v1.103.0-rc.1", shipped: false });
    expect(body.startsWith(FIXED_MARKER)).toBe(true);
    expect(body).toContain("Fixed by #41767.");
    expect(body).toContain("ships in v1.103.0-rc.1 and up");
    expect(body).toContain("dev pre-release");
  });

  test("a release that is already out says so instead of promising one", () => {
    const body = fixedBody(41767, { tag: "v1.102.0-rc.1", shipped: true });
    expect(body).toContain("is in v1.102.0-rc.1 and up");
    expect(body).not.toContain("ships in");
  });

  test("stays within the 25-word comment rule either way", () => {
    for (const shipped of [true, false]) {
      const words = fixedBody(41767, { tag: "v1.103.0-rc.1", shipped }).replace(FIXED_MARKER, "").trim().split(/\s+/);
      expect(words.length).toBeGreaterThanOrEqual(15);
      expect(words.length).toBeLessThanOrEqual(25);
    }
  });
});

describe("commentFixedIssue", () => {
  test("a real run posts one comment naming the pull request and the release", async () => {
    const { api, writes } = fakeApi();
    const verdict = await commentFixedIssue(api, config);
    expect(verdict).toMatchObject({ kind: "commented", pullRequest: 41767, tag: "v1.103.0-rc.1" });
    expect(writes).toHaveLength(1);
    expect(writes[0]).toContain("POST /repos/BerriAI/litellm/issues/41750/comments");
    expect(writes[0]).toContain("Fixed by #41767. This ships in v1.103.0-rc.1 and up");
  });

  test("a dry run renders the comment and writes nothing", async () => {
    const { api, writes } = fakeApi();
    const verdict = await commentFixedIssue(api, { ...config, dryRun: true });
    expect(verdict.kind).toBe("commented");
    expect(writes).toEqual([]);
  });

  test("an issue that already carries the comment is not commented twice", async () => {
    const existing: Comment = {
      id: 1,
      body: `${FIXED_MARKER}\nFixed by #41767. This ships in v1.103.0-rc.1 and up.`,
      created_at: "2026-09-18T00:00:00Z",
      user: { type: "Bot", login: "github-actions[bot]" },
    };
    const { api, writes } = fakeApi({ comments: [existing] });
    expect(await commentFixedIssue(api, config)).toEqual({ kind: "skip", reason: "already carries a fixed-in comment" });
    expect(writes).toEqual([]);
  });

  test("a hand-closed issue never reaches the release lookup or the API writes", async () => {
    const { api, writes } = fakeApi({ issue: closedBy(null) });
    expect((await commentFixedIssue(api, config)).kind).toBe("skip");
    expect(writes).toEqual([]);
  });

  test("a number that is not an issue in the repository is a skip", async () => {
    const { api, writes } = fakeApi({ issue: null });
    expect(await commentFixedIssue(api, config)).toEqual({ kind: "skip", reason: "not an issue in this repository" });
    expect(writes).toEqual([]);
  });
});

describe("readConfig", () => {
  const env = { GITHUB_TOKEN: "t", GITHUB_REPOSITORY: "BerriAI/litellm", ISSUE_NUMBER: "41750", DEFAULT_BRANCH: "main" };

  test("reads the four inputs and treats anything but the literal true as a real run", () => {
    expect(readConfig(env)).toEqual({ token: "t", repo: "BerriAI/litellm", issueNumber: 41750, defaultBranch: "main", dryRun: false });
    expect(readConfig({ ...env, DRY_RUN: "true" }).dryRun).toBe(true);
    expect(readConfig({ ...env, DRY_RUN: "false" }).dryRun).toBe(false);
  });

  test("refuses a missing token, repo, branch or a bad issue number", () => {
    expect(() => readConfig({ ...env, GITHUB_TOKEN: undefined })).toThrow("GITHUB_TOKEN");
    expect(() => readConfig({ ...env, GITHUB_REPOSITORY: "litellm" })).toThrow("owner/repo");
    expect(() => readConfig({ ...env, DEFAULT_BRANCH: "" })).toThrow("DEFAULT_BRANCH");
    expect(() => readConfig({ ...env, ISSUE_NUMBER: "0" })).toThrow("ISSUE_NUMBER");
    expect(() => readConfig({ ...env, ISSUE_NUMBER: "abc" })).toThrow("ISSUE_NUMBER");
  });
});
