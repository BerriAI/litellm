import { describe, expect, test } from "bun:test";

import type { Comment, GitHubApi } from "./auto-close-duplicates";
import {
  FIXED_MARKER,
  OPEN_PULL_REQUESTS_QUERY,
  SUPERSEDED_MARKER,
  closeVerdict,
  closerOf,
  describeIssue,
  describeSweep,
  fixOf,
  fixedBody,
  handleFixedIssue,
  nextMinor,
  parseVersion,
  placement,
  readConfig,
  releaseCandidate,
  supersededBody,
  sweep,
  type ClosedIssue,
  type FixedConfig,
  type IssueClosure,
  type LinkedIssue,
  type LinkedPullRequest,
  type PullRequestsPage,
} from "./comment-fixed-issue";

const MERGE_COMMIT = "68c4c82ac977b48b2b81ee8d633d5771307c6162";
const ISSUE = 41750;

const mergedPr = {
  __typename: "PullRequest" as const,
  number: 41767,
  merged: true,
  baseRefName: "main",
  repository: { nameWithOwner: "BerriAI/litellm" },
  mergeCommit: { oid: MERGE_COMMIT },
};

const commitCloser = { __typename: "Commit" as const, oid: MERGE_COMMIT, repository: { nameWithOwner: "BerriAI/litellm" } };

type Closer = IssueClosure["timelineItems"]["nodes"][number]["closer"];

const closure = (closer: Closer, state: IssueClosure["state"] = "CLOSED"): IssueClosure => ({
  state,
  timelineItems: { nodes: [{ closer }] },
});

const page = (pages: readonly (readonly LinkedPullRequest[])[], index: number): PullRequestsPage => ({
  pageInfo: { hasNextPage: index + 1 < pages.length, endCursor: String(index + 1) },
  nodes: pages[index] ?? [],
});

const cursorIndex = (after: string | null): number => (after === null ? 0 : Number(after));

const closedBy = (closer: Closer, state?: IssueClosure["state"], linked: readonly LinkedPullRequest[] = []): ClosedIssue => ({
  ...closure(closer, state),
  closedByPullRequestsReferences: page([linked], 0),
});

const reopenedAt = (createdAt: string): LinkedPullRequest["reopens"] => ({ nodes: [{ createdAt }] });

const linkedIssue = (number: number, closer: Closer = mergedPr, state?: IssueClosure["state"]): LinkedIssue => ({
  number,
  repository: { nameWithOwner: "BerriAI/litellm" },
  ...closure(closer, state),
});

const links = (...issues: readonly LinkedIssue[]): LinkedPullRequest["closingIssuesReferences"] => ({ totalCount: issues.length, nodes: issues });

const openPr = (number: number, overrides: Partial<LinkedPullRequest> = {}): LinkedPullRequest => ({
  number,
  state: "OPEN",
  baseRefName: "main",
  repository: { nameWithOwner: "BerriAI/litellm" },
  closingIssuesReferences: links(linkedIssue(ISSUE)),
  reopens: { nodes: [] },
  ...overrides,
});

const pyproject = (version: string): string =>
  `[project]\nname = "litellm"\nversion = "${version}"\n\n[tool.commitizen]\nversion = "${version}"\n`;

const config: FixedConfig = { repo: "BerriAI/litellm", defaultBranch: "main", commentDryRun: false, closeDryRun: false };

const prFix = (issue = ISSUE, number = 41767) => ({ issue, source: { kind: "pull_request" as const, number, oid: MERGE_COMMIT } });
const commitFix = (issue = ISSUE) => ({ issue, source: { kind: "commit" as const, oid: MERGE_COMMIT } });
const oneFixBody = supersededBody([prFix()], "main");

const noPause = async (): Promise<void> => {};

const supersededComment: Comment = {
  id: 2,
  body: `${SUPERSEDED_MARKER}\n#41750 was fixed by #41767 on main, so this pull request is closed. Reopen it if something was missed.`,
  created_at: "2026-09-18T00:00:00Z",
  user: { type: "Bot", login: "github-actions[bot]" },
};

interface World {
  readonly issue?: ClosedIssue | null;
  readonly comments?: readonly Comment[];
  readonly pullRequestComments?: Readonly<Record<number, readonly Comment[]>>;
  readonly openPullRequests?: readonly (readonly LinkedPullRequest[])[];
  readonly linkedPullRequests?: readonly (readonly LinkedPullRequest[])[];
  readonly version?: string;
  // Which existing rc.1 tags contain the merge commit; a tag absent from the map does not exist
  readonly tags?: Readonly<Record<string, boolean>>;
  readonly reachable?: Readonly<Record<string, readonly string[]>>;
}

function fakeApi(world: World = {}): { readonly api: GitHubApi; readonly writes: string[] } {
  const writes: string[] = [];
  const tags = world.tags ?? {};
  const reachable = world.reachable ?? { main: [MERGE_COMMIT] };
  const pages = world.openPullRequests ?? [];
  const api: GitHubApi = {
    request: async <T>(method: string, path: string, body?: object): Promise<T> => {
      if (method === "POST" && path === "/graphql") {
        const { query, variables } = body as { query: string; variables: { after: string | null } };
        if (query === OPEN_PULL_REQUESTS_QUERY) {
          return { data: { repository: { pullRequests: page(pages, cursorIndex(variables.after)) } } } as T;
        }
        const issue = world.issue === undefined ? closedBy(mergedPr) : world.issue;
        if (issue === null || world.linkedPullRequests === undefined) {
          return { data: { repository: { issue } } } as T;
        }
        const closedByPullRequestsReferences = page(world.linkedPullRequests, cursorIndex(variables.after));
        return { data: { repository: { issue: { ...issue, closedByPullRequestsReferences } } } } as T;
      }
      if (method !== "GET") {
        writes.push(`${method} ${path} ${JSON.stringify(body)}`);
        return {} as T;
      }
      const comments = /^\/repos\/BerriAI\/litellm\/issues\/(\d+)\/comments/.exec(path);
      if (comments !== null) {
        const number = Number(comments[1]);
        return ((number === ISSUE ? world.comments : world.pullRequestComments?.[number]) ?? []) as T;
      }
      if (path === `/repos/BerriAI/litellm/contents/pyproject.toml?ref=${MERGE_COMMIT}`) {
        return { content: btoa(pyproject(world.version ?? "1.103.0")).replace(/(.{60})/g, "$1\n") } as T;
      }
      const matching = /^\/repos\/BerriAI\/litellm\/git\/matching-refs\/tags\/(.+)$/.exec(path);
      if (matching !== null) {
        return (matching[1] in tags ? [{ ref: `refs/tags/${matching[1]}` }] : []) as T;
      }
      const compare = /^\/repos\/BerriAI\/litellm\/compare\/(.+)\.\.\.(.+)$/.exec(path);
      const branch = compare === null ? undefined : reachable[compare[1] ?? ""];
      if (compare !== null && branch !== undefined) {
        return { status: branch.includes(compare[2] ?? "") ? "behind" : "diverged" } as T;
      }
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
    expect(closerOf(closedBy(mergedPr), "BerriAI/litellm", "main")).toEqual({ kind: "pull_request", number: 41767, mergeCommit: MERGE_COMMIT });
  });

  test("an issue closed by hand, by a commit, or by an unmerged pull request gets no comment", () => {
    expect(closerOf(closedBy(null), "BerriAI/litellm", "main")).toEqual({ kind: "skip", reason: "closed by hand, not by a pull request" });
    expect(closerOf(closedBy(commitCloser), "BerriAI/litellm", "main").kind).toBe("skip");
    expect(closerOf(closedBy({ ...mergedPr, merged: false }), "BerriAI/litellm", "main").kind).toBe("skip");
    expect(closerOf(closedBy({ ...mergedPr, mergeCommit: null }), "BerriAI/litellm", "main").kind).toBe("skip");
  });

  test("a pull request merged into a release branch is not a fix on main", () => {
    const verdict = closerOf(closedBy({ ...mergedPr, baseRefName: "release/1.102.0rc2" }), "BerriAI/litellm", "main");
    expect(verdict).toEqual({ kind: "skip", reason: "#41767 merged into release/1.102.0rc2, not main" });
  });

  test("an issue reopened after the close event is left alone", () => {
    expect(closerOf(closedBy(mergedPr, "OPEN"), "BerriAI/litellm", "main")).toEqual({ kind: "skip", reason: "the issue is open again" });
  });

  test("a pull request merged in a fork closes the issue on GitHub but is no fix here", () => {
    const forkPr = { ...mergedPr, number: 9, repository: { nameWithOwner: "someone/litellm" } };
    expect(closerOf(closedBy(forkPr), "BerriAI/litellm", "main")).toEqual({ kind: "skip", reason: "closed by someone/litellm#9, a pull request in another repository" });
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

describe("fixOf", () => {
  test("a merged pull request or a commit is a fix, whatever branch it was merged into", () => {
    expect(fixOf(closure(mergedPr), "BerriAI/litellm")).toEqual({ kind: "pull_request", number: 41767, oid: MERGE_COMMIT });
    expect(fixOf(closure({ ...mergedPr, baseRefName: "release_branch" }), "BerriAI/litellm")).toEqual({ kind: "pull_request", number: 41767, oid: MERGE_COMMIT });
    expect(fixOf(closure(commitCloser), "BerriAI/litellm")).toEqual({ kind: "commit", oid: MERGE_COMMIT });
  });

  test("a hand close, a fork's pull request, an unmerged pull request, or a reopened issue is no fix", () => {
    expect(fixOf(closure(null), "BerriAI/litellm")).toEqual({ kind: "skip", reason: "was closed by hand" });
    expect(fixOf(closure({ ...mergedPr, repository: { nameWithOwner: "someone/litellm" } }), "BerriAI/litellm")).toEqual({ kind: "skip", reason: "was closed from someone/litellm" });
    expect(fixOf(closure({ ...commitCloser, repository: { nameWithOwner: "someone/litellm" } }), "BerriAI/litellm")).toEqual({ kind: "skip", reason: "was closed from someone/litellm" });
    expect(fixOf(closure({ ...mergedPr, merged: false }), "BerriAI/litellm")).toEqual({ kind: "skip", reason: "was closed by #41767, which is not merged" });
    expect(fixOf(closure({ ...mergedPr, mergeCommit: null }), "BerriAI/litellm").kind).toBe("skip");
    expect(fixOf(closure(mergedPr, "OPEN"), "BerriAI/litellm")).toEqual({ kind: "skip", reason: "is open again" });
  });
});

describe("closeVerdict", () => {
  test("an open pull request whose only linked issue was fixed by a merged pull request is a candidate", () => {
    expect(closeVerdict(openPr(41760), config)).toEqual({ kind: "candidate", fixes: [prFix()] });
  });

  test("every linked issue has to be fixed, and each fix is named", () => {
    const both = openPr(41760, { closingIssuesReferences: links(linkedIssue(ISSUE), linkedIssue(41751, commitCloser)) });
    expect(closeVerdict(both, config)).toEqual({ kind: "candidate", fixes: [prFix(), commitFix(41751)] });
  });

  test("a pull request that still links an open issue keeps its work", () => {
    const stillOpen = openPr(41760, { closingIssuesReferences: links(linkedIssue(ISSUE), linkedIssue(41751, null, "OPEN")) });
    expect(closeVerdict(stillOpen, config)).toEqual({ kind: "skip", reason: "still linked to open #41751" });
  });

  test("a linked issue closed by hand or by an unmerged pull request is not a fix that supersedes the pull request", () => {
    expect(closeVerdict(openPr(41760, { closingIssuesReferences: links(linkedIssue(ISSUE, null)) }), config)).toEqual({
      kind: "skip",
      reason: `#${ISSUE} was closed by hand`,
    });
    const unmerged = openPr(41760, { closingIssuesReferences: links(linkedIssue(ISSUE, { ...mergedPr, merged: false })) });
    expect(closeVerdict(unmerged, config)).toEqual({ kind: "skip", reason: `#${ISSUE} was closed by #41767, which is not merged` });
  });

  test("a pull request linking an issue in another repository, or more issues than the query reads, is left alone", () => {
    const foreign = { ...linkedIssue(41751), repository: { nameWithOwner: "mlflow/mlflow" } };
    expect(closeVerdict(openPr(41760, { closingIssuesReferences: links(linkedIssue(ISSUE), foreign) }), config)).toEqual({
      kind: "skip",
      reason: "links mlflow/mlflow#41751",
    });
    const truncated = openPr(41760, { closingIssuesReferences: { totalCount: 11, nodes: [linkedIssue(ISSUE)] } });
    expect(closeVerdict(truncated, config)).toEqual({ kind: "skip", reason: "links 11 issues, more than the 10 this workflow reads" });
  });

  test("a pull request against a release line is a backport and stays open, one against a retired development branch does not", () => {
    for (const base of ["release/v1.102.0-rc.2", "stable/v1.83.14", "v_1_83_3_stable_patch"]) {
      expect(closeVerdict(openPr(41760, { baseRefName: base }), config)).toEqual({ kind: "skip", reason: `targets the release line ${base}` });
    }
    expect(closeVerdict(openPr(41760, { baseRefName: "release_branch" }), config).kind).toBe("candidate");
  });

  test("a pull request in a fork, one that is not open, or one linking no issue is left alone", () => {
    expect(closeVerdict(openPr(1, { repository: { nameWithOwner: "someone/litellm" } }), config)).toEqual({ kind: "skip", reason: "lives in someone/litellm" });
    expect(closeVerdict(openPr(41767, { state: "MERGED" }), config)).toEqual({ kind: "skip", reason: "is merged" });
    expect(closeVerdict(openPr(41760, { closingIssuesReferences: links() }), config)).toEqual({ kind: "skip", reason: "links no issue" });
  });
});

describe("supersededBody", () => {
  test("names each issue with the pull request or commit that fixed it, invites a reopen, and carries the marker the rerun looks for", () => {
    expect(oneFixBody.startsWith(SUPERSEDED_MARKER)).toBe(true);
    expect(oneFixBody).toContain("#41750 was fixed by #41767 on main");
    expect(oneFixBody).toContain("Reopen it");
    expect(supersededBody([prFix(), commitFix(41751)], "main")).toContain("#41750 was fixed by #41767 and #41751 by commit 68c4c82ac9 on main");
  });

  test("stays within the 25-word comment rule for one and two fixes", () => {
    for (const fixes of [[prFix()], [commitFix()], [prFix(), commitFix(41751)]]) {
      const words = supersededBody(fixes, "main").replace(SUPERSEDED_MARKER, "").trim().split(/\s+/);
      expect(words.length).toBeGreaterThanOrEqual(15);
      expect(words.length).toBeLessThanOrEqual(25);
    }
  });
});

describe("handleFixedIssue", () => {
  test("a real run posts one comment naming the pull request and the release", async () => {
    const { api, writes } = fakeApi();
    const { comment, pullRequests } = await handleFixedIssue(api, config, ISSUE, noPause);
    expect(comment).toMatchObject({ kind: "commented", pullRequest: 41767, tag: "v1.103.0-rc.1" });
    expect(pullRequests).toEqual([]);
    expect(writes).toHaveLength(1);
    expect(writes[0]).toContain("POST /repos/BerriAI/litellm/issues/41750/comments");
    expect(writes[0]).toContain("Fixed by #41767. This ships in v1.103.0-rc.1 and up");
  });

  test("every other open pull request linked to the fixed issue is commented on and then closed, a pause before every write", async () => {
    const { api, writes } = fakeApi({ issue: closedBy(mergedPr, "CLOSED", [openPr(41760), openPr(41761)]) });
    let pauses = 0;
    const { pullRequests } = await handleFixedIssue(api, config, ISSUE, async () => {
      pauses += 1;
    });
    expect(pullRequests.map((pullRequest) => pullRequest.kind)).toEqual(["closed", "closed"]);
    expect(writes).toEqual([
      expect.stringContaining("POST /repos/BerriAI/litellm/issues/41750/comments"),
      `POST /repos/BerriAI/litellm/issues/41760/comments ${JSON.stringify({ body: oneFixBody })}`,
      'PATCH /repos/BerriAI/litellm/pulls/41760 {"state":"closed"}',
      expect.stringContaining("POST /repos/BerriAI/litellm/issues/41761/comments"),
      'PATCH /repos/BerriAI/litellm/pulls/41761 {"state":"closed"}',
    ]);
    expect(pauses).toBe(4);
  });

  test("the fixed-in comment and the closing are gated separately: a close dry run still posts the comment and closes nothing", async () => {
    const { api, writes } = fakeApi({ issue: closedBy(mergedPr, "CLOSED", [openPr(41760)]) });
    let pauses = 0;
    const outcome = await handleFixedIssue(api, { ...config, closeDryRun: true }, ISSUE, async () => {
      pauses += 1;
    });
    expect(outcome.comment.kind).toBe("commented");
    expect(outcome.pullRequests).toEqual([{ kind: "closed", number: 41760, body: oneFixBody }]);
    expect(writes).toHaveLength(1);
    expect(writes[0]).toContain("/issues/41750/comments");
    expect(pauses).toBe(0);
  });

  test("a comment dry run still closes the pull requests for real", async () => {
    const { api, writes } = fakeApi({ issue: closedBy(mergedPr, "CLOSED", [openPr(41760)]) });
    const outcome = await handleFixedIssue(api, { ...config, commentDryRun: true }, ISSUE, noPause);
    expect(outcome.comment.kind).toBe("commented");
    expect(writes).toEqual([expect.stringContaining("/issues/41760/comments"), 'PATCH /repos/BerriAI/litellm/pulls/41760 {"state":"closed"}']);
  });

  test("an issue that already carries the fixed-in comment is not commented twice, and its linked pull requests still get closed", async () => {
    const existing: Comment = {
      id: 1,
      body: `${FIXED_MARKER}\nFixed by #41767. This ships in v1.103.0-rc.1 and up.`,
      created_at: "2026-09-18T00:00:00Z",
      user: { type: "Bot", login: "github-actions[bot]" },
    };
    const { api, writes } = fakeApi({ comments: [existing], issue: closedBy(mergedPr, "CLOSED", [openPr(41760)]) });
    const outcome = await handleFixedIssue(api, config, ISSUE, noPause);
    expect(outcome.comment).toEqual({ kind: "skip", reason: "already carries a fixed-in comment" });
    expect(writes).toEqual([expect.stringContaining("/issues/41760/comments"), 'PATCH /repos/BerriAI/litellm/pulls/41760 {"state":"closed"}']);
  });

  test("a fix merged into the retired development branch or pushed as a commit counts once it is on the default branch", async () => {
    const stagingPr = { ...mergedPr, baseRefName: "release_branch" };
    const staging = openPr(41760, { baseRefName: "release_branch", closingIssuesReferences: links(linkedIssue(ISSUE, stagingPr)) });
    const byCommit = openPr(41761, { closingIssuesReferences: links(linkedIssue(41751, commitCloser)) });
    const { api, writes } = fakeApi({ issue: closedBy(mergedPr, "CLOSED", [staging, byCommit]) });
    const { pullRequests } = await handleFixedIssue(api, config, ISSUE, noPause);
    expect(pullRequests).toEqual([
      { kind: "closed", number: 41760, body: oneFixBody },
      { kind: "closed", number: 41761, body: supersededBody([commitFix(41751)], "main") },
    ]);
    expect(writes).toHaveLength(5);
    expect(writes[3]).toContain("#41751 was fixed by commit 68c4c82ac9 on main");
  });

  test("a fix whose commit never reached the default branch supersedes nothing", async () => {
    const { api, writes } = fakeApi({ issue: closedBy(mergedPr, "CLOSED", [openPr(41760)]), reachable: { main: [] } });
    const { pullRequests } = await handleFixedIssue(api, config, ISSUE, noPause);
    expect(pullRequests).toEqual([{ kind: "skip", number: 41760, reason: "#41750 was fixed by #41767, which is not on main" }]);
    expect(writes).toHaveLength(1);
    expect(writes[0]).toContain("/issues/41750/comments");
  });

  test("the default branch comes from the config for the containment check and the comment alike", async () => {
    const stagingConfig = { ...config, defaultBranch: "release_branch" };
    const closer = { ...mergedPr, baseRefName: "release_branch" };
    const { api, writes } = fakeApi({
      issue: closedBy(closer, "CLOSED", [openPr(41760, { closingIssuesReferences: links(linkedIssue(ISSUE, closer)) })]),
      reachable: { release_branch: [MERGE_COMMIT] },
    });
    const { pullRequests } = await handleFixedIssue(api, stagingConfig, ISSUE, noPause);
    expect(pullRequests).toEqual([{ kind: "closed", number: 41760, body: supersededBody([prFix()], "release_branch") }]);
    expect(writes[1]).toContain("on release_branch, so this pull request is closed");
  });

  test("the closer sits in the linked list as merged and gets neither a line nor a write", async () => {
    const { api, writes } = fakeApi({ issue: closedBy(mergedPr, "CLOSED", [openPr(41767, { state: "MERGED" }), openPr(41760)]) });
    const { pullRequests } = await handleFixedIssue(api, config, ISSUE, noPause);
    expect(pullRequests).toEqual([{ kind: "closed", number: 41760, body: oneFixBody }]);
    expect(writes.map((write) => write.split(" ")[1])).toEqual([
      "/repos/BerriAI/litellm/issues/41750/comments",
      "/repos/BerriAI/litellm/issues/41760/comments",
      "/repos/BerriAI/litellm/pulls/41760",
    ]);
  });

  test("a pull request this workflow closed once and its author reopened stays open", async () => {
    const reopened = openPr(41760, { reopens: reopenedAt("2026-09-19T00:00:00Z") });
    const { api, writes } = fakeApi({
      issue: closedBy(mergedPr, "CLOSED", [reopened, openPr(41761)]),
      pullRequestComments: { 41760: [supersededComment] },
    });
    const { pullRequests } = await handleFixedIssue(api, config, ISSUE, noPause);
    expect(pullRequests[0]).toEqual({ kind: "skip", number: 41760, reason: "was closed by this workflow once and reopened" });
    expect(pullRequests[1]?.kind).toBe("closed");
    expect(writes.filter((write) => write.includes("41760"))).toEqual([]);
  });

  test("a pull request whose comment landed but whose close failed is closed on the next run without a second comment", async () => {
    const reopenedBeforeTheComment = openPr(41761, { reopens: reopenedAt("2026-09-17T00:00:00Z") });
    const { api, writes } = fakeApi({
      issue: closedBy(mergedPr, "CLOSED", [openPr(41760), reopenedBeforeTheComment]),
      pullRequestComments: { 41760: [supersededComment], 41761: [supersededComment] },
    });
    const { pullRequests } = await handleFixedIssue(api, config, ISSUE, noPause);
    expect(pullRequests).toEqual([
      { kind: "closed", number: 41760, body: supersededComment.body },
      { kind: "closed", number: 41761, body: supersededComment.body },
    ]);
    expect(writes.filter((write) => write.includes("/4176"))).toEqual([
      'PATCH /repos/BerriAI/litellm/pulls/41760 {"state":"closed"}',
      'PATCH /repos/BerriAI/litellm/pulls/41761 {"state":"closed"}',
    ]);
  });

  test("a superseded marker pasted by anyone but the workflow neither keeps a pull request open nor replaces its comment", async () => {
    const forged: Comment = { ...supersededComment, id: 3, user: { type: "User", login: "someone" } };
    const { api, writes } = fakeApi({
      issue: closedBy(mergedPr, "CLOSED", [openPr(41760, { reopens: reopenedAt("2026-09-19T00:00:00Z") })]),
      pullRequestComments: { 41760: [forged] },
    });
    const { pullRequests } = await handleFixedIssue(api, config, ISSUE, noPause);
    expect(pullRequests).toEqual([{ kind: "closed", number: 41760, body: oneFixBody }]);
    expect(writes.filter((write) => write.includes("/41760"))).toEqual([
      `POST /repos/BerriAI/litellm/issues/41760/comments ${JSON.stringify({ body: oneFixBody })}`,
      'PATCH /repos/BerriAI/litellm/pulls/41760 {"state":"closed"}',
    ]);
  });

  test("every page of linked pull requests is read, not just the first", async () => {
    const { api } = fakeApi({ linkedPullRequests: [[openPr(41760)], [openPr(41761)], [openPr(41762)]] });
    const { pullRequests } = await handleFixedIssue(api, config, ISSUE, noPause);
    expect(pullRequests).toEqual([
      { kind: "closed", number: 41760, body: oneFixBody },
      { kind: "closed", number: 41761, body: oneFixBody },
      { kind: "closed", number: 41762, body: oneFixBody },
    ]);
  });

  test("a linked pull request from a fork or one still tied to another open issue is reported, not closed", async () => {
    const fork = openPr(1, { repository: { nameWithOwner: "someone/litellm" } });
    const busy = openPr(41762, { closingIssuesReferences: links(linkedIssue(ISSUE), linkedIssue(41751, null, "OPEN")) });
    const { api, writes } = fakeApi({ issue: closedBy(mergedPr, "CLOSED", [fork, busy]) });
    const { pullRequests } = await handleFixedIssue(api, config, ISSUE, noPause);
    expect(pullRequests).toEqual([
      { kind: "skip", number: 1, reason: "lives in someone/litellm" },
      { kind: "skip", number: 41762, reason: "still linked to open #41751" },
    ]);
    expect(writes).toHaveLength(1);
  });

  test("a hand-closed issue gets no comment and leaves its linked pull requests open with the reason on each", async () => {
    const byHand = openPr(41760, { closingIssuesReferences: links(linkedIssue(ISSUE, null)) });
    const { api, writes } = fakeApi({ issue: closedBy(null, "CLOSED", [byHand]) });
    const outcome = await handleFixedIssue(api, config, ISSUE, noPause);
    expect(outcome.comment).toEqual({ kind: "skip", reason: "closed by hand, not by a pull request" });
    expect(outcome.pullRequests).toEqual([{ kind: "skip", number: 41760, reason: `#${ISSUE} was closed by hand` }]);
    expect(writes).toEqual([]);
  });

  test("an issue closed by a commit on the default branch gets no comment but still closes its linked pull requests", async () => {
    const byCommit = openPr(41760, { closingIssuesReferences: links(linkedIssue(ISSUE, commitCloser)) });
    const { api, writes } = fakeApi({ issue: closedBy(commitCloser, "CLOSED", [byCommit]) });
    const { comment, pullRequests } = await handleFixedIssue(api, config, ISSUE, noPause);
    expect(comment).toEqual({ kind: "skip", reason: "closed by commit 68c4c82ac9, not by a pull request" });
    expect(pullRequests).toEqual([{ kind: "closed", number: 41760, body: supersededBody([commitFix()], "main") }]);
    expect(writes).toEqual([expect.stringContaining("/issues/41760/comments"), 'PATCH /repos/BerriAI/litellm/pulls/41760 {"state":"closed"}']);
    expect(writes[0]).toContain("#41750 was fixed by commit 68c4c82ac9 on main");
  });

  test("an issue closed from the retired development branch gets no comment but still closes its linked pull requests once the fix is on the default branch", async () => {
    const stagingPr = { ...mergedPr, baseRefName: "release_branch" };
    const staging = openPr(41760, { closingIssuesReferences: links(linkedIssue(ISSUE, stagingPr)) });
    const { api, writes } = fakeApi({ issue: closedBy(stagingPr, "CLOSED", [staging]) });
    const { comment, pullRequests } = await handleFixedIssue(api, config, ISSUE, noPause);
    expect(comment).toEqual({ kind: "skip", reason: "#41767 merged into release_branch, not main" });
    expect(pullRequests).toEqual([{ kind: "closed", number: 41760, body: oneFixBody }]);
    expect(writes.map((write) => write.split(" ")[1])).toEqual(["/repos/BerriAI/litellm/issues/41760/comments", "/repos/BerriAI/litellm/pulls/41760"]);
  });

  test("an issue that is open again gets no comment and its linked pull requests are neither read nor touched", async () => {
    const { api, writes } = fakeApi({ issue: closedBy(mergedPr, "OPEN", [openPr(41760)]) });
    const outcome = await handleFixedIssue(api, config, ISSUE, noPause);
    expect(outcome).toEqual({ comment: { kind: "skip", reason: "the issue is open again" }, pullRequests: [] });
    expect(writes).toEqual([]);
  });

  test("a number that is not an issue in the repository is a skip", async () => {
    const { api, writes } = fakeApi({ issue: null });
    const outcome = await handleFixedIssue(api, config, ISSUE, noPause);
    expect(outcome.comment).toEqual({ kind: "skip", reason: "not an issue in this repository" });
    expect(writes).toEqual([]);
  });
});

describe("sweep", () => {
  test("walks every page of open pull requests and closes the ones whose linked issues were all fixed", async () => {
    const unlinked = openPr(41700, { closingIssuesReferences: links() });
    const busy = openPr(41701, { closingIssuesReferences: links(linkedIssue(41751, null, "OPEN")) });
    const { api, writes } = fakeApi({ openPullRequests: [[unlinked, openPr(41760)], [busy, openPr(41761)]] });
    const outcome = await sweep(api, { ...config, commentDryRun: true }, noPause);
    expect(outcome.considered).toBe(4);
    expect(outcome.pullRequests).toEqual([
      { kind: "closed", number: 41760, body: oneFixBody },
      { kind: "skip", number: 41701, reason: "still linked to open #41751" },
      { kind: "closed", number: 41761, body: oneFixBody },
    ]);
    expect(writes).toEqual([
      expect.stringContaining("POST /repos/BerriAI/litellm/issues/41760/comments"),
      'PATCH /repos/BerriAI/litellm/pulls/41760 {"state":"closed"}',
      expect.stringContaining("POST /repos/BerriAI/litellm/issues/41761/comments"),
      'PATCH /repos/BerriAI/litellm/pulls/41761 {"state":"closed"}',
    ]);
  });

  test("a sweep dry run lists what it would close and writes nothing", async () => {
    const { api, writes } = fakeApi({ openPullRequests: [[openPr(41760)]] });
    const outcome = await sweep(api, { ...config, closeDryRun: true }, noPause);
    expect(outcome.pullRequests.map((pullRequest) => pullRequest.kind)).toEqual(["closed"]);
    expect(writes).toEqual([]);
  });
});

describe("readConfig", () => {
  const env = { GITHUB_TOKEN: "t", GITHUB_REPOSITORY: "BerriAI/litellm", ISSUE_NUMBER: "41750", DEFAULT_BRANCH: "main" };

  test("reads the inputs and treats anything but the literal true as a real run for each gate", () => {
    expect(readConfig(env)).toEqual({
      token: "t",
      repo: "BerriAI/litellm",
      defaultBranch: "main",
      commentDryRun: false,
      closeDryRun: false,
      run: { kind: "issue", number: 41750 },
    });
    expect(readConfig({ ...env, DRY_RUN: "true" })).toMatchObject({ commentDryRun: true, closeDryRun: false });
    expect(readConfig({ ...env, CLOSE_PRS_DRY_RUN: "true" })).toMatchObject({ commentDryRun: false, closeDryRun: true });
    expect(readConfig({ ...env, DRY_RUN: "false", CLOSE_PRS_DRY_RUN: "false" })).toMatchObject({ commentDryRun: false, closeDryRun: false });
  });

  test("a sweep needs no issue number and anything but the literal true is an issue run", () => {
    expect(readConfig({ ...env, ISSUE_NUMBER: undefined, SWEEP: "true" }).run).toEqual({ kind: "sweep" });
    expect(readConfig({ ...env, SWEEP: "false" }).run).toEqual({ kind: "issue", number: 41750 });
    expect(() => readConfig({ ...env, ISSUE_NUMBER: "", SWEEP: "false" })).toThrow("dispatch with an issue_number or with sweep ticked");
  });

  test("refuses a missing token, repo, branch or a bad issue number", () => {
    expect(() => readConfig({ ...env, GITHUB_TOKEN: undefined })).toThrow("GITHUB_TOKEN");
    expect(() => readConfig({ ...env, GITHUB_REPOSITORY: "litellm" })).toThrow("owner/repo");
    expect(() => readConfig({ ...env, DEFAULT_BRANCH: "" })).toThrow("DEFAULT_BRANCH");
    expect(() => readConfig({ ...env, ISSUE_NUMBER: "0" })).toThrow("ISSUE_NUMBER");
    expect(() => readConfig({ ...env, ISSUE_NUMBER: "abc" })).toThrow("ISSUE_NUMBER");
  });
});

describe("step summary", () => {
  const closed = { kind: "closed" as const, number: 41760, body: oneFixBody };
  const left = { kind: "skip" as const, number: 41762, reason: "still linked to open #41751" };
  const commented = { kind: "commented" as const, pullRequest: 41767, tag: "v1.103.0-rc.1", body: fixedBody(41767, { tag: "v1.103.0-rc.1", shipped: false }) };

  test("an issue run names the comment, each close, and each pull request left open", () => {
    const summary = describeIssue(config, ISSUE, { comment: commented, pullRequests: [closed, left] });
    expect(summary).toContain("#41750: commented, fixed by #41767 in v1.103.0-rc.1");
    expect(summary).toContain("#41760: closed with: #41750 was fixed by #41767 on main");
    expect(summary).toContain("#41762: left open, still linked to open #41751");
    expect(summary).not.toContain("DRY RUN");
  });

  test("a close dry run names the repo variable that turns closing on", () => {
    const summary = describeIssue({ ...config, closeDryRun: true }, ISSUE, { comment: commented, pullRequests: [closed] });
    expect(summary).toContain("ISSUE_FIXED_CLOSE_PRS_ENABLED");
    expect(summary).toContain("#41760: DRY RUN, would close with: #41750 was fixed by #41767 on main");
  });

  test("a sweep summary counts what it saw and lists only the closes", () => {
    const summary = describeSweep(config, { considered: 3522, pullRequests: [closed, left] });
    expect(summary).toContain("Swept 3522 open pull requests, 2 linked to an issue, 1 closed");
    expect(summary).toContain("#41760: closed with:");
    expect(summary).not.toContain("#41762");
    expect(describeSweep({ ...config, closeDryRun: true }, { considered: 3522, pullRequests: [closed] })).toContain("1 would be closed, DRY RUN, set the ISSUE_FIXED_CLOSE_PRS_ENABLED");
  });
});
