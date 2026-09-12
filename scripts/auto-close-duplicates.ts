#!/usr/bin/env bun

declare const process: { readonly env: Readonly<Record<string, string | undefined>> };

export interface Issue {
  readonly number: number;
  readonly title: string;
  readonly state: string;
  readonly user: { readonly login: string };
  readonly closed_by?: { readonly type: string } | null;
  readonly pull_request?: unknown;
}

export interface Comment {
  readonly id: number;
  readonly body: string;
  readonly created_at: string;
  readonly user: { readonly type: string; readonly login: string };
}

export interface Reaction {
  readonly content: string;
}

export interface GitHubApi {
  readonly request: <T>(method: "GET" | "POST" | "PATCH" | "DELETE", path: string, body?: object) => Promise<T>;
}

export interface SweepConfig {
  readonly repo: string;
  readonly graceDays: number;
  readonly dryRun: boolean;
  readonly now: Date;
}

export type NoticeVerdict =
  | { readonly kind: "pending"; readonly notices: readonly Comment[]; readonly candidates: readonly number[] }
  | { readonly kind: "skip"; readonly reason: string };

export type CloseVerdict =
  | { readonly kind: "close"; readonly duplicateOf: number }
  | { readonly kind: "skip"; readonly reason: string };

export type ReopenVerdict =
  | { readonly kind: "reopen" }
  | { readonly kind: "skip"; readonly reason: string };

export const FLAG_LABEL = "potential-duplicate";
export const CLOSED_MARKER = "<!-- litellm:closed-as-duplicate -->";
export const DEFAULT_GRACE_DAYS = 3;
export const REOPEN_COMMENT =
  "Reopened automatically: the reporter replied after the duplicate close, so this needs a human look.";
const NOTICE_MARKER = /<!-- litellm:potential-duplicate candidates=([\d,]*) -->/;
const MIN_TITLE_WORDS = 3;
const PAGE_SIZE = 100;
const DAY_MS = 24 * 60 * 60 * 1000;
const REOPEN_LOOKBACK_DAYS = 30;

const skip = (reason: string): { readonly kind: "skip"; readonly reason: string } => ({ kind: "skip", reason });

export function normalizeTitle(title: string): string {
  return title
    .toLowerCase()
    .replace(/^\s*\[[^\]]*\]\s*:?/, "")
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

export function candidateNumbers(noticeBody: string, issueNumber: number): readonly number[] {
  const field = noticeBody.match(NOTICE_MARKER);
  if (!field) {
    return [];
  }
  const older = field[1]
    .split(",")
    .filter((value) => value !== "")
    .map(Number)
    .filter((candidate) => candidate < issueNumber);
  return [...new Set(older)].sort((a, b) => a - b);
}

export function pendingNotice(
  issue: Issue,
  comments: readonly Comment[],
  config: Pick<SweepConfig, "graceDays" | "now">,
): NoticeVerdict {
  if (issue.pull_request !== undefined) {
    return skip("is a pull request");
  }
  if (comments.some((comment) => comment.body.includes(CLOSED_MARKER))) {
    return skip("was reopened after an automatic close");
  }
  const notices = comments.filter((comment) => comment.user.type === "Bot" && NOTICE_MARKER.test(comment.body));
  const first = notices[0];
  const latest = notices[notices.length - 1];
  if (first === undefined || latest === undefined) {
    return skip("carries no duplicate notice");
  }
  const ageDays = (config.now.getTime() - new Date(latest.created_at).getTime()) / DAY_MS;
  if (ageDays < config.graceDays) {
    return skip(`notice is ${ageDays.toFixed(1)} days old, grace period is ${config.graceDays}`);
  }
  const firstNoticeAt = new Date(first.created_at);
  if (comments.some((comment) => comment.user.type !== "Bot" && new Date(comment.created_at) > firstNoticeAt)) {
    return skip("someone replied after the notice");
  }
  const candidates = candidateNumbers(latest.body, issue.number);
  if (candidates.length === 0) {
    return skip("no candidate is older than this issue");
  }
  return { kind: "pending", notices, candidates };
}

export function duplicateTarget(
  issue: Issue,
  candidates: readonly Issue[],
  reactions: readonly Reaction[],
): CloseVerdict {
  if (reactions.some((reaction) => reaction.content === "-1")) {
    return skip("someone gave the notice a thumbs down");
  }
  const title = normalizeTitle(issue.title);
  if (title.split(" ").length < MIN_TITLE_WORDS) {
    return skip(`title "${issue.title}" is too short to match on`);
  }
  const original = candidates.find(
    (candidate) =>
      candidate.state === "open" && candidate.pull_request === undefined && normalizeTitle(candidate.title) === title,
  );
  if (original === undefined) {
    return skip("no older open issue has the identical title");
  }
  return { kind: "close", duplicateOf: original.number };
}

export function reopenTarget(issue: Issue, comments: readonly Comment[]): ReopenVerdict {
  if (issue.pull_request !== undefined) {
    return skip("is a pull request");
  }
  if (issue.closed_by?.type !== "Bot") {
    return skip("was closed by a person");
  }
  const marker = comments.find((comment) => comment.body.includes(CLOSED_MARKER));
  if (marker === undefined) {
    return skip("carries no automatic-close marker");
  }
  const markerAt = new Date(marker.created_at);
  if (!comments.some((comment) => comment.user.login === issue.user.login && new Date(comment.created_at) > markerAt)) {
    return skip("the reporter has not replied since the close");
  }
  return { kind: "reopen" };
}

export function closingComment(duplicateOf: number, graceDays: number): string {
  return `Closed automatically as a duplicate of #${duplicateOf}. Its title is identical to that older open issue and the duplicate notice above went unanswered for ${graceDays} days. If this is wrong, comment here with how it differs from #${duplicateOf} and this issue will be reopened automatically within a day.

${CLOSED_MARKER}`;
}

async function listAll<T>(api: GitHubApi, path: string, page = 1): Promise<readonly T[]> {
  const separator = path.includes("?") ? "&" : "?";
  const batch = await api.request<readonly T[]>("GET", `${path}${separator}per_page=${PAGE_SIZE}&page=${page}`);
  return batch.length < PAGE_SIZE ? batch : [...batch, ...(await listAll<T>(api, path, page + 1))];
}

async function closeAsDuplicate(
  api: GitHubApi,
  config: SweepConfig,
  issueNumber: number,
  duplicateOf: number,
): Promise<void> {
  const issuePath = `/repos/${config.repo}/issues/${issueNumber}`;
  await api.request("POST", `${issuePath}/comments`, { body: closingComment(duplicateOf, config.graceDays) });
  await api.request("POST", `${issuePath}/labels`, { labels: ["duplicate"] });
  await api.request("PATCH", issuePath, { state: "closed", state_reason: "duplicate" });
}

async function reopenForReporter(api: GitHubApi, config: SweepConfig, issueNumber: number): Promise<void> {
  const issuePath = `/repos/${config.repo}/issues/${issueNumber}`;
  await api.request("DELETE", `${issuePath}/labels/duplicate`);
  await api.request("PATCH", issuePath, { state: "open" });
  await api.request("POST", `${issuePath}/comments`, { body: REOPEN_COMMENT });
}

export async function sweepClosedIssue(api: GitHubApi, config: SweepConfig, issueNumber: number): Promise<ReopenVerdict> {
  const issue = await api.request<Issue>("GET", `/repos/${config.repo}/issues/${issueNumber}`);
  const comments = await listAll<Comment>(api, `/repos/${config.repo}/issues/${issueNumber}/comments`);
  const verdict = reopenTarget(issue, comments);
  if (verdict.kind === "reopen" && !config.dryRun) {
    await reopenForReporter(api, config, issueNumber);
  }
  return verdict;
}

export async function sweepIssue(api: GitHubApi, config: SweepConfig, issue: Issue): Promise<CloseVerdict> {
  const comments = await listAll<Comment>(api, `/repos/${config.repo}/issues/${issue.number}/comments`);
  const pending = pendingNotice(issue, comments, config);
  if (pending.kind === "skip") {
    return pending;
  }
  const reactions = (
    await Promise.all(
      pending.notices.map((notice) => listAll<Reaction>(api, `/repos/${config.repo}/issues/comments/${notice.id}/reactions`)),
    )
  ).flat();
  const candidates = await Promise.all(
    pending.candidates.map((candidate) => api.request<Issue>("GET", `/repos/${config.repo}/issues/${candidate}`)),
  );
  const verdict = duplicateTarget(issue, candidates, reactions);
  if (verdict.kind === "close" && !config.dryRun) {
    await closeAsDuplicate(api, config, issue.number, verdict.duplicateOf);
  }
  return verdict;
}

function describe(issue: Issue, verdict: CloseVerdict, dryRun: boolean): string {
  if (verdict.kind === "skip") {
    return `#${issue.number}: skipped, ${verdict.reason}`;
  }
  return `#${issue.number}: ${dryRun ? "would close" : "closed"} as a duplicate of #${verdict.duplicateOf}`;
}

export async function sweep(api: GitHubApi, config: SweepConfig): Promise<readonly CloseVerdict[]> {
  const issues = await listAll<Issue>(api, `/repos/${config.repo}/issues?state=open&labels=${FLAG_LABEL}`);
  console.log(`${issues.length} open issues carry the ${FLAG_LABEL} label in ${config.repo}${config.dryRun ? " (dry run)" : ""}`);
  return issues.reduce<Promise<readonly CloseVerdict[]>>(async (previous, issue) => {
    const verdicts = await previous;
    const verdict = await sweepIssue(api, config, issue);
    console.log(describe(issue, verdict, config.dryRun));
    return [...verdicts, verdict];
  }, Promise.resolve([]));
}

function describeReopen(issueNumber: number, verdict: ReopenVerdict, dryRun: boolean): string {
  if (verdict.kind === "skip") {
    return `#${issueNumber}: skipped, ${verdict.reason}`;
  }
  return `#${issueNumber}: ${dryRun ? "would reopen" : "reopened"} for the reporter's reply`;
}

export async function reopenSweep(api: GitHubApi, config: SweepConfig): Promise<readonly ReopenVerdict[]> {
  const since = new Date(config.now.getTime() - REOPEN_LOOKBACK_DAYS * DAY_MS).toISOString();
  const closedPath = `/repos/${config.repo}/issues?state=closed&labels=duplicate,${FLAG_LABEL}&since=${encodeURIComponent(since)}`;
  const issues = await listAll<Issue>(api, closedPath);
  console.log(`${issues.length} recently closed issues carry the duplicate and ${FLAG_LABEL} labels in ${config.repo}${config.dryRun ? " (dry run)" : ""}`);
  return issues.reduce<Promise<readonly ReopenVerdict[]>>(async (previous, issue) => {
    const verdicts = await previous;
    const verdict = await sweepClosedIssue(api, config, issue.number);
    console.log(describeReopen(issue.number, verdict, config.dryRun));
    return [...verdicts, verdict];
  }, Promise.resolve([]));
}

export function readConfig(env: Readonly<Record<string, string | undefined>>, now: Date): SweepConfig & { readonly token: string } {
  const token = env.GITHUB_TOKEN;
  const repo = env.GITHUB_REPOSITORY;
  if (!token || !repo || !/^[\w.-]+\/[\w.-]+$/.test(repo)) {
    throw new Error("GITHUB_TOKEN and GITHUB_REPOSITORY (owner/repo) are required");
  }
  const rawGraceDays = env.GRACE_PERIOD_DAYS?.trim();
  const graceDays = rawGraceDays === undefined || rawGraceDays === "" ? DEFAULT_GRACE_DAYS : Number(rawGraceDays);
  if (!Number.isFinite(graceDays) || graceDays < 0) {
    throw new Error(`GRACE_PERIOD_DAYS must be a non-negative number, got "${env.GRACE_PERIOD_DAYS}"`);
  }
  return { token, repo, graceDays, dryRun: env.DRY_RUN === "true", now };
}

export function githubApi(token: string): GitHubApi {
  return {
    request: async <T>(method: "GET" | "POST" | "PATCH" | "DELETE", path: string, body?: object): Promise<T> => {
      const response = await fetch(`https://api.github.com${path}`, {
        method,
        headers: {
          Authorization: `Bearer ${token}`,
          Accept: "application/vnd.github+json",
          "X-GitHub-Api-Version": "2022-11-28",
          "User-Agent": "litellm-auto-close-duplicates",
          ...(body === undefined ? {} : { "Content-Type": "application/json" }),
        },
        body: body === undefined ? undefined : JSON.stringify(body),
      });
      if (!response.ok) {
        throw new Error(`${method} ${path} failed: ${response.status} ${response.statusText}`);
      }
      return (await response.json()) as T;
    },
  };
}

if (import.meta.main) {
  const { token, ...config } = readConfig(process.env, new Date());
  const api = githubApi(token);
  const closeVerdicts = await sweep(api, config);
  const reopenVerdicts = await reopenSweep(api, config);
  const closed = closeVerdicts.filter((verdict) => verdict.kind === "close").length;
  const reopened = reopenVerdicts.filter((verdict) => verdict.kind === "reopen").length;
  console.log(
    `${config.dryRun ? "Would close" : "Closed"} ${closed} of ${closeVerdicts.length} flagged issues, ${config.dryRun ? "would reopen" : "reopened"} ${reopened}`,
  );
}
