#!/usr/bin/env bun

import { FLAG_LABEL, githubApi, listAll, type Comment, type GitHubApi } from "./auto-close-duplicates";
import { CLEAR_LABEL, readConfig, type FlagConfig as DispatchConfig } from "./flag-duplicate-issue";
import { labelName } from "./issue-labels";

declare const process: { readonly env: Readonly<Record<string, string | undefined>> };

export interface IssueForDispatch {
  readonly number: number;
  readonly state: string;
  readonly labels: readonly { readonly name: string }[];
  readonly pull_request?: unknown;
}

export interface TimelineSource {
  readonly number: number;
  readonly state: string;
  readonly pull_request?: unknown;
}

export interface TimelineEvent {
  readonly event: string;
  readonly source?: { readonly issue?: TimelineSource };
}

export type DispatchVerdict =
  | { readonly kind: "dispatch" }
  | { readonly kind: "skip"; readonly reason: string };

export const BUG_LABEL = labelName("kind", "bug");
export const RUNNING_LABEL = labelName("repro", "running");
export const DISPATCH_MARKER = "<!-- litellm:repro-dispatched -->";
export const DISPATCH_COMMENT = [
  DISPATCH_MARKER,
  "Devin is going to try to reproduce this. A PR or a Linear ticket follows if it does. Add repro:skip to opt out",
].join("\n");
const BLOCKING_LABELS: readonly string[] = [FLAG_LABEL, "duplicate"];
const BLOCKING_PREFIXES: readonly string[] = ["needs:", "repro:"];

const skip = (reason: string): DispatchVerdict => ({ kind: "skip", reason });

export function openPullsReferencing(timeline: readonly TimelineEvent[]): readonly number[] {
  const numbers = timeline
    .flatMap((event) => (event.event === "cross-referenced" && event.source?.issue !== undefined ? [event.source.issue] : []))
    .filter((source) => source.pull_request !== undefined && source.state === "open")
    .map((source) => source.number);
  return [...new Set(numbers)].sort((a, b) => a - b);
}

export function dispatchTarget(
  issue: IssueForDispatch,
  comments: readonly Comment[],
  timeline: readonly TimelineEvent[],
): DispatchVerdict {
  if (issue.pull_request !== undefined) {
    return skip("is a pull request");
  }
  if (issue.state !== "open") {
    return skip(`is ${issue.state}`);
  }
  const labels = issue.labels.map((label) => label.name);
  const missing = [BUG_LABEL, CLEAR_LABEL].filter((label) => !labels.includes(label));
  if (missing.length > 0) {
    return skip(`missing ${missing.join(" and ")}`);
  }
  const blocking = labels.filter(
    (label) => BLOCKING_LABELS.includes(label) || BLOCKING_PREFIXES.some((prefix) => label.startsWith(prefix)),
  );
  if (blocking.length > 0) {
    return skip(`carries ${blocking.join(", ")}`);
  }
  if (comments.some((comment) => comment.body.includes(DISPATCH_MARKER))) {
    return skip("was already dispatched once");
  }
  const pulls = openPullsReferencing(timeline);
  if (pulls.length === 1) {
    return skip(`open pull request #${pulls[0]} already references it`);
  }
  if (pulls.length > 1) {
    return skip(`open pull requests ${pulls.map((n) => `#${n}`).join(", ")} already reference it`);
  }
  return { kind: "dispatch" };
}

export async function dispatchIssue(api: GitHubApi, config: DispatchConfig): Promise<DispatchVerdict> {
  const issuePath = `/repos/${config.repo}/issues/${config.issueNumber}`;
  const issue = await api.request<IssueForDispatch>("GET", issuePath);
  const comments = await listAll<Comment>(api, `${issuePath}/comments`);
  const timeline = await listAll<TimelineEvent>(api, `${issuePath}/timeline`);
  const verdict = dispatchTarget(issue, comments, timeline);
  if (verdict.kind === "skip" || config.dryRun) {
    return verdict;
  }
  await api.request("POST", `${issuePath}/labels`, { labels: [RUNNING_LABEL] });
  await api.request("POST", `${issuePath}/comments`, { body: DISPATCH_COMMENT });
  return verdict;
}

function describe(config: DispatchConfig, verdict: DispatchVerdict): string {
  if (verdict.kind === "skip") {
    return `#${config.issueNumber}: skipped, ${verdict.reason}`;
  }
  if (config.dryRun) {
    return `#${config.issueNumber}: DRY RUN, set the TRIAGE_DISPATCH_ENABLED repo variable to true to add ${RUNNING_LABEL} and comment`;
  }
  return `#${config.issueNumber}: added ${RUNNING_LABEL} and commented, Devin's automation takes it from here`;
}

if (import.meta.main) {
  const { token, ...config } = readConfig(process.env);
  const verdict = await dispatchIssue(githubApi(token), config);
  console.log(describe(config, verdict));
}
