#!/usr/bin/env bun

import { githubApi, listAll, type Comment, type GitHubApi } from "./auto-close-duplicates";

declare const process: { readonly env: Readonly<Record<string, string | undefined>> };

export interface FixedConfig {
  readonly repo: string;
  readonly defaultBranch: string;
  readonly commentDryRun: boolean;
  readonly closeDryRun: boolean;
}

export type Run = { readonly kind: "issue"; readonly number: number } | { readonly kind: "sweep" };

interface PullRequestCloser {
  readonly __typename: "PullRequest";
  readonly number: number;
  readonly merged: boolean;
  readonly baseRefName: string;
  readonly repository: { readonly nameWithOwner: string };
  readonly mergeCommit: { readonly oid: string } | null;
}

interface CommitCloser {
  readonly __typename: "Commit";
  readonly oid: string;
  readonly repository: { readonly nameWithOwner: string };
}

export interface IssueClosure {
  readonly state: "OPEN" | "CLOSED";
  readonly timelineItems: {
    readonly nodes: readonly { readonly closer: PullRequestCloser | CommitCloser | null }[];
  };
}

export interface LinkedIssue extends IssueClosure {
  readonly number: number;
  readonly repository: { readonly nameWithOwner: string };
}

export interface LinkedPullRequest {
  readonly number: number;
  readonly state: "OPEN" | "CLOSED" | "MERGED";
  readonly baseRefName: string;
  readonly repository: { readonly nameWithOwner: string };
  readonly closingIssuesReferences: { readonly totalCount: number; readonly nodes: readonly LinkedIssue[] };
  readonly reopens: { readonly nodes: readonly { readonly createdAt: string }[] };
}

export interface PullRequestsPage {
  readonly pageInfo: { readonly hasNextPage: boolean; readonly endCursor: string | null };
  readonly nodes: readonly LinkedPullRequest[];
}

export interface ClosedIssue extends IssueClosure {
  readonly closedByPullRequestsReferences: PullRequestsPage;
}

interface TimelineResponse {
  readonly data?: { readonly repository?: { readonly issue: ClosedIssue | null } };
}

interface OpenPullRequestsResponse {
  readonly data?: { readonly repository?: { readonly pullRequests: PullRequestsPage } };
}

interface MatchingRef {
  readonly ref: string;
}

interface Comparison {
  readonly status: "ahead" | "behind" | "identical" | "diverged";
}

interface FileContent {
  readonly content: string;
}

export type Closer =
  | { readonly kind: "pull_request"; readonly number: number; readonly mergeCommit: string }
  | { readonly kind: "skip"; readonly reason: string };

export type Placement =
  | { readonly kind: "release"; readonly tag: string; readonly shipped: boolean }
  | { readonly kind: "skip"; readonly reason: string };

export type FixedVerdict =
  | { readonly kind: "commented"; readonly pullRequest: number; readonly tag: string; readonly body: string }
  | { readonly kind: "skip"; readonly reason: string };

export type FixSource =
  | { readonly kind: "pull_request"; readonly number: number; readonly oid: string }
  | { readonly kind: "commit"; readonly oid: string };

export type FixVerdict = FixSource | { readonly kind: "skip"; readonly reason: string };

export interface Fix {
  readonly issue: number;
  readonly source: FixSource;
}

export type CloseVerdict =
  | { readonly kind: "candidate"; readonly fixes: readonly Fix[] }
  | { readonly kind: "skip"; readonly reason: string };

export type CloseOutcome =
  | { readonly kind: "closed"; readonly number: number; readonly body: string }
  | { readonly kind: "skip"; readonly number: number; readonly reason: string };

export interface IssueOutcome {
  readonly comment: FixedVerdict;
  readonly pullRequests: readonly CloseOutcome[];
}

export interface SweepOutcome {
  readonly considered: number;
  readonly pullRequests: readonly CloseOutcome[];
}

export const FIXED_MARKER = "<!-- litellm:fixed-in -->";
export const SUPERSEDED_MARKER = "<!-- litellm:superseded -->";
const MAX_MINOR_BUMPS = 3;
const MAX_LINKED_PULL_REQUESTS = 50;
const MAX_LINKED_ISSUES = 10;
const SWEEP_PAGE_SIZE = 100;
const CLOSE_PAUSE_MS = 1000;
const WORKFLOW_LOGIN = "github-actions[bot]";
const OPEN_AGAIN = "the issue is open again";
const isReleaseLine = (base: string): boolean => base.startsWith("release/") || base.includes("stable");

const CLOSURE_FRAGMENT = `fragment Closure on Issue {
  state
  timelineItems(last: 1, itemTypes: [CLOSED_EVENT]) {
    nodes {
      ... on ClosedEvent {
        closer {
          __typename
          ... on PullRequest { number merged baseRefName repository { nameWithOwner } mergeCommit { oid } }
          ... on Commit { oid repository { nameWithOwner } }
        }
      }
    }
  }
}`;

const LINKED_PULL_REQUEST_FRAGMENT = `fragment Linked on PullRequest {
  number
  state
  baseRefName
  repository { nameWithOwner }
  closingIssuesReferences(first: ${MAX_LINKED_ISSUES}) { totalCount nodes { number repository { nameWithOwner } ...Closure } }
  reopens: timelineItems(last: 1, itemTypes: [REOPENED_EVENT]) { nodes { ... on ReopenedEvent { createdAt } } }
}`;

export const CLOSER_QUERY = `query($owner: String!, $name: String!, $number: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    issue(number: $number) {
      ...Closure
      closedByPullRequestsReferences(first: ${MAX_LINKED_PULL_REQUESTS}, after: $after) {
        pageInfo { hasNextPage endCursor }
        nodes { ...Linked }
      }
    }
  }
}
${CLOSURE_FRAGMENT}
${LINKED_PULL_REQUEST_FRAGMENT}`;

export const OPEN_PULL_REQUESTS_QUERY = `query($owner: String!, $name: String!, $after: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(states: OPEN, first: ${SWEEP_PAGE_SIZE}, after: $after) {
      pageInfo { hasNextPage endCursor }
      nodes { ...Linked }
    }
  }
}
${CLOSURE_FRAGMENT}
${LINKED_PULL_REQUEST_FRAGMENT}`;

const skip = (reason: string): { readonly kind: "skip"; readonly reason: string } => ({ kind: "skip", reason });

export function closerOf(issue: IssueClosure, repo: string, defaultBranch: string): Closer {
  if (issue.state !== "CLOSED") {
    return skip(OPEN_AGAIN);
  }
  const closer = issue.timelineItems.nodes[0]?.closer ?? null;
  if (closer === null) {
    return skip("closed by hand, not by a pull request");
  }
  if (closer.__typename === "Commit") {
    return skip(`closed by commit ${closer.oid.slice(0, 10)}, not by a pull request`);
  }
  if (closer.repository.nameWithOwner !== repo) {
    return skip(`closed by ${closer.repository.nameWithOwner}#${closer.number}, a pull request in another repository`);
  }
  if (!closer.merged || closer.mergeCommit === null) {
    return skip(`closed by #${closer.number}, which is not merged`);
  }
  if (closer.baseRefName !== defaultBranch) {
    return skip(`#${closer.number} merged into ${closer.baseRefName}, not ${defaultBranch}`);
  }
  return { kind: "pull_request", number: closer.number, mergeCommit: closer.mergeCommit.oid };
}

export function parseVersion(pyproject: string): string | undefined {
  return /^version = "(\d+\.\d+\.\d+)"$/m.exec(pyproject)?.[1];
}

export function releaseCandidate(version: string): string {
  return `v${version}-rc.1`;
}

export function nextMinor(version: string): string {
  const [major, minor] = version.split(".").map(Number);
  return `${major}.${minor + 1}.0`;
}

async function tagExists(api: GitHubApi, repo: string, tag: string): Promise<boolean> {
  const refs = await api.request<readonly MatchingRef[]>("GET", `/repos/${repo}/git/matching-refs/tags/${tag}`);
  return refs.some((ref) => ref.ref === `refs/tags/${tag}`);
}

async function refContains(api: GitHubApi, repo: string, ref: string, sha: string): Promise<boolean> {
  const comparison = await api.request<Comparison>("GET", `/repos/${repo}/compare/${ref}...${sha}`);
  return comparison.status === "behind" || comparison.status === "identical";
}

// The first rc of a version is cut straight from main, so a fix merged while pyproject says X.Y.Z ships in
// vX.Y.Z-rc.1 unless that rc was already cut without it, in which case it waits for the next minor's rc.1
async function firstReleaseWith(
  api: GitHubApi,
  repo: string,
  sha: string,
  version: string,
  bumpsLeft: number,
): Promise<Placement> {
  const tag = releaseCandidate(version);
  if (!(await tagExists(api, repo, tag))) {
    return { kind: "release", tag, shipped: false };
  }
  if (await refContains(api, repo, tag, sha)) {
    return { kind: "release", tag, shipped: true };
  }
  if (bumpsLeft === 0) {
    return skip(`${tag} exists without ${sha.slice(0, 10)} and the next ${MAX_MINOR_BUMPS} rc.1 tags are taken too`);
  }
  return firstReleaseWith(api, repo, sha, nextMinor(version), bumpsLeft - 1);
}

export async function placement(api: GitHubApi, repo: string, mergeCommit: string): Promise<Placement> {
  const file = await api.request<FileContent>("GET", `/repos/${repo}/contents/pyproject.toml?ref=${mergeCommit}`);
  const version = parseVersion(atob(file.content.replace(/\n/g, "")));
  if (version === undefined) {
    return skip(`pyproject.toml at ${mergeCommit.slice(0, 10)} has no version line`);
  }
  return firstReleaseWith(api, repo, mergeCommit, version, MAX_MINOR_BUMPS);
}

export function fixedBody(pullRequest: number, release: { readonly tag: string; readonly shipped: boolean }): string {
  const availability = release.shipped
    ? `This is in ${release.tag} and up, so upgrading to that release or any newer one picks it up.`
    : `This ships in ${release.tag} and up, and the next dev pre-release cut from main will carry it too.`;
  return `${FIXED_MARKER}\nFixed by #${pullRequest}. ${availability}`;
}

export async function commentFixedIssue(
  api: GitHubApi,
  config: FixedConfig,
  issueNumber: number,
  closer: { readonly number: number; readonly mergeCommit: string },
): Promise<FixedVerdict> {
  const issuePath = `/repos/${config.repo}/issues/${issueNumber}`;
  const comments = await listAll<Comment>(api, `${issuePath}/comments`);
  if (comments.some((comment) => comment.body.includes(FIXED_MARKER))) {
    return skip("already carries a fixed-in comment");
  }
  const release = await placement(api, config.repo, closer.mergeCommit);
  if (release.kind === "skip") {
    return release;
  }
  const body = fixedBody(closer.number, release);
  if (!config.commentDryRun) {
    await api.request("POST", `${issuePath}/comments`, { body });
  }
  return { kind: "commented", pullRequest: closer.number, tag: release.tag, body };
}

export function fixOf(issue: IssueClosure, repo: string): FixVerdict {
  if (issue.state !== "CLOSED") {
    return skip("is open again");
  }
  const closer = issue.timelineItems.nodes[0]?.closer ?? null;
  if (closer === null) {
    return skip("was closed by hand");
  }
  if (closer.repository.nameWithOwner !== repo) {
    return skip(`was closed from ${closer.repository.nameWithOwner}`);
  }
  if (closer.__typename === "Commit") {
    return { kind: "commit", oid: closer.oid };
  }
  if (!closer.merged || closer.mergeCommit === null) {
    return skip(`was closed by #${closer.number}, which is not merged`);
  }
  return { kind: "pull_request", number: closer.number, oid: closer.mergeCommit.oid };
}

export function closeVerdict(pullRequest: LinkedPullRequest, config: FixedConfig): CloseVerdict {
  if (pullRequest.repository.nameWithOwner !== config.repo) {
    return skip(`lives in ${pullRequest.repository.nameWithOwner}`);
  }
  if (pullRequest.state !== "OPEN") {
    return skip(`is ${pullRequest.state.toLowerCase()}`);
  }
  if (isReleaseLine(pullRequest.baseRefName)) {
    return skip(`targets the release line ${pullRequest.baseRefName}`);
  }
  const { totalCount, nodes: linked } = pullRequest.closingIssuesReferences;
  if (linked.length === 0) {
    return skip("links no issue");
  }
  if (totalCount > linked.length) {
    return skip(`links ${totalCount} issues, more than the ${MAX_LINKED_ISSUES} this workflow reads`);
  }
  const foreign = linked.find((issue) => issue.repository.nameWithOwner !== config.repo);
  if (foreign !== undefined) {
    return skip(`links ${foreign.repository.nameWithOwner}#${foreign.number}`);
  }
  const stillOpen = linked.find((issue) => issue.state === "OPEN");
  if (stillOpen !== undefined) {
    return skip(`still linked to open #${stillOpen.number}`);
  }
  const verdicts = linked.map((issue) => ({ issue: issue.number, fix: fixOf(issue, config.repo) }));
  for (const { issue, fix } of verdicts) {
    if (fix.kind === "skip") {
      return skip(`#${issue} ${fix.reason}`);
    }
  }
  return {
    kind: "candidate",
    fixes: verdicts.flatMap(({ issue, fix }) => (fix.kind === "skip" ? [] : [{ issue, source: fix }])),
  };
}

function describeSource(source: FixSource): string {
  return source.kind === "pull_request" ? `#${source.number}` : `commit ${source.oid.slice(0, 10)}`;
}

async function fixOffDefaultBranch(api: GitHubApi, config: FixedConfig, fixes: readonly Fix[]): Promise<Fix | undefined> {
  const onBranch = await Promise.all(fixes.map((fix) => refContains(api, config.repo, config.defaultBranch, fix.source.oid)));
  return fixes.find((_, index) => !onBranch[index]);
}

export function supersededBody(fixes: readonly Fix[], defaultBranch: string): string {
  const pairs = fixes
    .map((fix, index) => `#${fix.issue} ${index === 0 ? "was fixed by" : "by"} ${describeSource(fix.source)}`)
    .join(" and ");
  return `${SUPERSEDED_MARKER}\n${pairs} on ${defaultBranch}, so this pull request is closed. Reopen it if something was missed.`;
}

function reopenedAfter(pullRequest: LinkedPullRequest, comment: Comment): boolean {
  const reopen = pullRequest.reopens.nodes[0];
  return reopen !== undefined && Date.parse(reopen.createdAt) > Date.parse(comment.created_at);
}

async function closePullRequest(
  api: GitHubApi,
  config: FixedConfig,
  pullRequest: LinkedPullRequest,
  pause: () => Promise<void>,
): Promise<CloseOutcome> {
  const verdict = closeVerdict(pullRequest, config);
  if (verdict.kind === "skip") {
    return { kind: "skip", number: pullRequest.number, reason: verdict.reason };
  }
  const offBranch = await fixOffDefaultBranch(api, config, verdict.fixes);
  if (offBranch !== undefined) {
    const fix = describeSource(offBranch.source);
    return { kind: "skip", number: pullRequest.number, reason: `#${offBranch.issue} was fixed by ${fix}, which is not on ${config.defaultBranch}` };
  }
  const issuePath = `/repos/${config.repo}/issues/${pullRequest.number}`;
  const comments = await listAll<Comment>(api, `${issuePath}/comments`);
  const marker = comments.find((comment) => comment.user.login === WORKFLOW_LOGIN && comment.body.includes(SUPERSEDED_MARKER));
  if (marker !== undefined && reopenedAfter(pullRequest, marker)) {
    return { kind: "skip", number: pullRequest.number, reason: "was closed by this workflow once and reopened" };
  }
  const body = marker?.body ?? supersededBody(verdict.fixes, config.defaultBranch);
  if (config.closeDryRun) {
    return { kind: "closed", number: pullRequest.number, body };
  }
  if (marker === undefined) {
    await pause();
    await api.request("POST", `${issuePath}/comments`, { body });
  }
  await pause();
  await api.request("PATCH", `/repos/${config.repo}/pulls/${pullRequest.number}`, { state: "closed" });
  return { kind: "closed", number: pullRequest.number, body };
}

export function closePullRequests(
  api: GitHubApi,
  config: FixedConfig,
  candidates: readonly LinkedPullRequest[],
  pause: () => Promise<void>,
): Promise<readonly CloseOutcome[]> {
  return candidates.reduce<Promise<readonly CloseOutcome[]>>(
    async (previous, candidate) => [...(await previous), await closePullRequest(api, config, candidate, pause)],
    Promise.resolve([]),
  );
}

type NextPage = (after: string | null) => Promise<PullRequestsPage>;

async function collectPages(page: PullRequestsPage, nextPage: NextPage): Promise<readonly LinkedPullRequest[]> {
  if (!page.pageInfo.hasNextPage) {
    return page.nodes;
  }
  return [...page.nodes, ...(await collectPages(await nextPage(page.pageInfo.endCursor), nextPage))];
}

async function closedIssue(api: GitHubApi, config: FixedConfig, issueNumber: number, after: string | null): Promise<ClosedIssue | null> {
  const [owner, name] = config.repo.split("/");
  const response = await api.request<TimelineResponse>("POST", "/graphql", {
    query: CLOSER_QUERY,
    variables: { owner, name, number: issueNumber, after },
  });
  return response.data?.repository?.issue ?? null;
}

export async function handleFixedIssue(
  api: GitHubApi,
  config: FixedConfig,
  issueNumber: number,
  pause: () => Promise<void>,
): Promise<IssueOutcome> {
  const issue = await closedIssue(api, config, issueNumber, null);
  if (issue === null) {
    return { comment: skip("not an issue in this repository"), pullRequests: [] };
  }
  if (issue.state !== "CLOSED") {
    return { comment: skip(OPEN_AGAIN), pullRequests: [] };
  }
  const closer = closerOf(issue, config.repo, config.defaultBranch);
  const comment = closer.kind === "skip" ? closer : await commentFixedIssue(api, config, issueNumber, closer);
  const nextPage: NextPage = async (after) => {
    const more = await closedIssue(api, config, issueNumber, after);
    if (more === null) {
      throw new Error(`#${issueNumber} came back without data while reading its linked pull requests after cursor ${after}`);
    }
    return more.closedByPullRequestsReferences;
  };
  const linked = await collectPages(issue.closedByPullRequestsReferences, nextPage);
  const open = linked.filter((pullRequest) => pullRequest.state === "OPEN");
  const pullRequests = await closePullRequests(api, config, open, pause);
  return { comment, pullRequests };
}

async function openPullRequests(api: GitHubApi, config: FixedConfig): Promise<readonly LinkedPullRequest[]> {
  const [owner, name] = config.repo.split("/");
  const nextPage: NextPage = async (after) => {
    const response = await api.request<OpenPullRequestsResponse>("POST", "/graphql", {
      query: OPEN_PULL_REQUESTS_QUERY,
      variables: { owner, name, after },
    });
    const page = response.data?.repository?.pullRequests;
    if (page === undefined) {
      throw new Error(`open pull requests after cursor ${after} came back without data: ${JSON.stringify(response)}`);
    }
    return page;
  };
  return collectPages(await nextPage(null), nextPage);
}

export async function sweep(api: GitHubApi, config: FixedConfig, pause: () => Promise<void>): Promise<SweepOutcome> {
  const open = await openPullRequests(api, config);
  const linked = open.filter((pullRequest) => pullRequest.closingIssuesReferences.nodes.length > 0);
  return { considered: open.length, pullRequests: await closePullRequests(api, config, linked, pause) };
}

export function readConfig(
  env: Readonly<Record<string, string | undefined>>,
): FixedConfig & { readonly token: string; readonly run: Run } {
  const token = env.GITHUB_TOKEN;
  const repo = env.GITHUB_REPOSITORY;
  const defaultBranch = env.DEFAULT_BRANCH;
  if (!token || !repo || !/^[\w.-]+\/[\w.-]+$/.test(repo) || !defaultBranch) {
    throw new Error("GITHUB_TOKEN, GITHUB_REPOSITORY (owner/repo) and DEFAULT_BRANCH are required");
  }
  const config = { token, repo, defaultBranch, commentDryRun: env.DRY_RUN === "true", closeDryRun: env.CLOSE_PRS_DRY_RUN === "true" };
  if (env.SWEEP === "true") {
    return { ...config, run: { kind: "sweep" } };
  }
  const issueNumber = Number(env.ISSUE_NUMBER);
  if (!Number.isInteger(issueNumber) || issueNumber <= 0) {
    throw new Error(`ISSUE_NUMBER must be a positive integer, got "${env.ISSUE_NUMBER}": dispatch with an issue_number or with sweep ticked`);
  }
  return { ...config, run: { kind: "issue", number: issueNumber } };
}

const CLOSE_DRY_RUN_HINT = "set the ISSUE_FIXED_CLOSE_PRS_ENABLED repo variable to true to close pull requests";

function describeClose(config: FixedConfig, outcome: CloseOutcome): string {
  if (outcome.kind === "skip") {
    return `#${outcome.number}: left open, ${outcome.reason}`;
  }
  const text = outcome.body.replace(`${SUPERSEDED_MARKER}\n`, "");
  return config.closeDryRun ? `#${outcome.number}: DRY RUN, would close with: ${text}` : `#${outcome.number}: closed with: ${text}`;
}

export function describeIssue(config: FixedConfig, issueNumber: number, outcome: IssueOutcome): string {
  const comment =
    outcome.comment.kind === "skip"
      ? `#${issueNumber}: skipped, ${outcome.comment.reason}`
      : config.commentDryRun
        ? `#${issueNumber}: DRY RUN, set the ISSUE_FIXED_COMMENT_ENABLED repo variable to true to post this:\n\n${outcome.comment.body}`
        : `#${issueNumber}: commented, fixed by #${outcome.comment.pullRequest} in ${outcome.comment.tag}`;
  const hint = config.closeDryRun && outcome.pullRequests.some((pullRequest) => pullRequest.kind === "closed") ? [`Closing is a DRY RUN, ${CLOSE_DRY_RUN_HINT}`] : [];
  return [comment, ...hint, ...outcome.pullRequests.map((pullRequest) => describeClose(config, pullRequest))].join("\n");
}

export function describeSweep(config: FixedConfig, outcome: SweepOutcome): string {
  const closed = outcome.pullRequests.filter((pullRequest) => pullRequest.kind === "closed");
  const verb = config.closeDryRun ? `would be closed, DRY RUN, ${CLOSE_DRY_RUN_HINT}` : "closed";
  const header = `Swept ${outcome.considered} open pull requests, ${outcome.pullRequests.length} linked to an issue, ${closed.length} ${verb}`;
  return [header, ...closed.map((pullRequest) => describeClose(config, pullRequest))].join("\n");
}

if (import.meta.main) {
  const { token, run, ...config } = readConfig(process.env);
  const api = githubApi(token);
  const pause = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, CLOSE_PAUSE_MS));
  console.log(
    run.kind === "sweep"
      ? describeSweep(config, await sweep(api, config, pause))
      : describeIssue(config, run.number, await handleFixedIssue(api, config, run.number, pause)),
  );
}
