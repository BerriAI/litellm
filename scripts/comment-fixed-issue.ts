#!/usr/bin/env bun

import { githubApi, listAll, type Comment, type GitHubApi } from "./auto-close-duplicates";

declare const process: { readonly env: Readonly<Record<string, string | undefined>> };

export interface FixedConfig {
  readonly repo: string;
  readonly issueNumber: number;
  readonly defaultBranch: string;
  readonly dryRun: boolean;
}

interface PullRequestCloser {
  readonly __typename: "PullRequest";
  readonly number: number;
  readonly merged: boolean;
  readonly baseRefName: string;
  readonly mergeCommit: { readonly oid: string } | null;
}

interface CommitCloser {
  readonly __typename: "Commit";
  readonly oid: string;
}

export interface ClosedIssue {
  readonly state: "OPEN" | "CLOSED";
  readonly timelineItems: {
    readonly nodes: readonly { readonly closer: PullRequestCloser | CommitCloser | null }[];
  };
}

interface TimelineResponse {
  readonly data?: { readonly repository?: { readonly issue: ClosedIssue | null } };
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

export const FIXED_MARKER = "<!-- litellm:fixed-in -->";
const MAX_MINOR_BUMPS = 3;

export const CLOSER_QUERY = `query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    issue(number: $number) {
      state
      timelineItems(last: 1, itemTypes: [CLOSED_EVENT]) {
        nodes {
          ... on ClosedEvent {
            closer {
              __typename
              ... on PullRequest { number merged baseRefName mergeCommit { oid } }
              ... on Commit { oid }
            }
          }
        }
      }
    }
  }
}`;

const skip = (reason: string): { readonly kind: "skip"; readonly reason: string } => ({ kind: "skip", reason });

export function closerOf(issue: ClosedIssue, defaultBranch: string): Closer {
  if (issue.state !== "CLOSED") {
    return skip("the issue is open again");
  }
  const closer = issue.timelineItems.nodes[0]?.closer ?? null;
  if (closer === null) {
    return skip("closed by hand, not by a pull request");
  }
  if (closer.__typename === "Commit") {
    return skip(`closed by commit ${closer.oid.slice(0, 10)}, not by a pull request`);
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

async function tagContains(api: GitHubApi, repo: string, tag: string, sha: string): Promise<boolean> {
  const comparison = await api.request<Comparison>("GET", `/repos/${repo}/compare/${tag}...${sha}`);
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
  if (await tagContains(api, repo, tag, sha)) {
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

export async function commentFixedIssue(api: GitHubApi, config: FixedConfig): Promise<FixedVerdict> {
  const [owner, name] = config.repo.split("/");
  const response = await api.request<TimelineResponse>("POST", "/graphql", {
    query: CLOSER_QUERY,
    variables: { owner, name, number: config.issueNumber },
  });
  const issue = response.data?.repository?.issue ?? null;
  if (issue === null) {
    return skip("not an issue in this repository");
  }
  const closer = closerOf(issue, config.defaultBranch);
  if (closer.kind === "skip") {
    return closer;
  }
  const issuePath = `/repos/${config.repo}/issues/${config.issueNumber}`;
  const comments = await listAll<Comment>(api, `${issuePath}/comments`);
  if (comments.some((comment) => comment.body.includes(FIXED_MARKER))) {
    return skip("already carries a fixed-in comment");
  }
  const release = await placement(api, config.repo, closer.mergeCommit);
  if (release.kind === "skip") {
    return release;
  }
  const body = fixedBody(closer.number, release);
  if (!config.dryRun) {
    await api.request("POST", `${issuePath}/comments`, { body });
  }
  return { kind: "commented", pullRequest: closer.number, tag: release.tag, body };
}

export function readConfig(env: Readonly<Record<string, string | undefined>>): FixedConfig & { readonly token: string } {
  const token = env.GITHUB_TOKEN;
  const repo = env.GITHUB_REPOSITORY;
  const defaultBranch = env.DEFAULT_BRANCH;
  if (!token || !repo || !/^[\w.-]+\/[\w.-]+$/.test(repo) || !defaultBranch) {
    throw new Error("GITHUB_TOKEN, GITHUB_REPOSITORY (owner/repo) and DEFAULT_BRANCH are required");
  }
  const issueNumber = Number(env.ISSUE_NUMBER);
  if (!Number.isInteger(issueNumber) || issueNumber <= 0) {
    throw new Error(`ISSUE_NUMBER must be a positive integer, got "${env.ISSUE_NUMBER}"`);
  }
  return { token, repo, issueNumber, defaultBranch, dryRun: env.DRY_RUN === "true" };
}

function describe(config: FixedConfig, verdict: FixedVerdict): string {
  if (verdict.kind === "skip") {
    return `#${config.issueNumber}: skipped, ${verdict.reason}`;
  }
  if (config.dryRun) {
    return `#${config.issueNumber}: DRY RUN, set the ISSUE_FIXED_COMMENT_ENABLED repo variable to true to post this:\n\n${verdict.body}`;
  }
  return `#${config.issueNumber}: commented, fixed by #${verdict.pullRequest} in ${verdict.tag}`;
}

if (import.meta.main) {
  const { token, ...config } = readConfig(process.env);
  console.log(describe(config, await commentFixedIssue(githubApi(token), config)));
}
